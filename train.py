import os
import json
import time
import argparse
import tempfile
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar import build_cifar10_dataloaders, build_cifar_train_val_test_dataloaders
from models.convmixer import ConvMixer
from models.rowcol_convmixer import RowColOnlyConvMixer, AdditiveRowColConvMixer, WeightedRowColConvMixer, WeightedMultiplicativeRowColConvMixer
from utils.seed import set_seed
from utils.metrics import count_parameters, evaluate


def build_model(args):
    """
    Build model by name.

    Supported models:
        M0: convmixer
        M1: rowcol_only
        M2: rowcol_add
        M3: rowcol_weighted
    """
    n_classes = 100 if getattr(args, "dataset", "CIFAR10") == "CIFAR100" else 10
    if args.model == "convmixer":
        return ConvMixer(
            dim=args.hdim,
            depth=args.depth,
            kernel_size=args.conv_ks,
            patch_size=args.psize,
            n_classes=n_classes,
            in_channels=3,
        )

    if args.model == "rowcol_only":
        return RowColOnlyConvMixer(
            dim=args.hdim,
            depth=args.depth,
            patch_size=args.psize,
            image_size=args.image_size,
            n_classes=n_classes,
            in_channels=3,
            rowcol_combine=args.rowcol_combine,
        )

    if args.model == "rowcol_add":
        return AdditiveRowColConvMixer(
            dim=args.hdim,
            depth=args.depth,
            kernel_size=args.conv_ks,
            patch_size=args.psize,
            image_size=args.image_size,
            n_classes=n_classes,
            in_channels=3,
            rowcol_combine=args.rowcol_combine,
        )

    if args.model == "rowcol_weighted":
        return WeightedRowColConvMixer(
            dim=args.hdim,
            depth=args.depth,
            kernel_size=args.conv_ks,
            patch_size=args.psize,
            image_size=args.image_size,
            n_classes=n_classes,
            in_channels=3,
            rowcol_combine=args.rowcol_combine,
            local_init=args.local_scale_init,
            rowcol_init=args.rowcol_scale_init,
            rowcol_pre_norm=getattr(args, "rowcol_pre_norm", False),
        )

    if args.model == "rowcol_mul_weighted":
        return WeightedMultiplicativeRowColConvMixer(
            dim=args.hdim,
            depth=args.depth,
            kernel_size=args.conv_ks,
            patch_size=args.psize,
            image_size=args.image_size,
            n_classes=n_classes,
            in_channels=3,
            local_init=args.local_scale_init,
            rowcol_init=args.rowcol_scale_init,
            channel_specific_pos=args.mul_pos_channel_specific,
            mul_pre_norm=not args.no_mul_pre_norm,
            mul_pos_prior=not getattr(args, "no_mul_pos_prior", False),
        )

    raise ValueError(f"Unknown model: {args.model}")


def build_lr_schedule(args):
    """
    Learning-rate schedule from the user's original code.

    It warms up to lr_max, decays to lr_max/20, then goes to 0.
    """
    def lr_schedule(t):
        return np.interp(
            [t],
            [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
            [0, args.lr_max, args.lr_max / 20.0, 0],
        )[0]

    return lr_schedule


def get_rowcol_metadata(args):
    """Return the effective RowCol condition represented by the model."""
    if args.model == "convmixer":
        return {
            "rowcol_fusion": None,
            "rowcol_pre_norm": None,
            "rowcol_pos_prior": None,
            "rowcol_branch_weight": None,
        }

    if args.model == "rowcol_mul_weighted":
        return {
            "rowcol_fusion": "multiply",
            "rowcol_pre_norm": not args.no_mul_pre_norm,
            "rowcol_pos_prior": not getattr(args, "no_mul_pos_prior", False),
            "rowcol_branch_weight": False,
        }

    return {
        "rowcol_fusion": "additive",
        "rowcol_pre_norm": (
            getattr(args, "rowcol_pre_norm", False)
            if args.model == "rowcol_weighted"
            else False
        ),
        "rowcol_pos_prior": False,
        "rowcol_branch_weight": args.rowcol_combine in ["learnable", "softmax"],
    }


def condition_token(value):
    if value is None:
        return "na"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def build_result_filename(args, rowcol_metadata, collision_suffix=None):
    filename = (
        f"{args.name}"
        f"_model_{args.model}"
        f"_seed_{args.seed}"
        f"_dim_{args.hdim}"
        f"_depth_{args.depth}"
        f"_psize_{args.psize}"
        f"_ks_{args.conv_ks}"
        f"_fusion_{condition_token(rowcol_metadata['rowcol_fusion'])}"
        f"_prenorm_{condition_token(rowcol_metadata['rowcol_pre_norm'])}"
        f"_posprior_{condition_token(rowcol_metadata['rowcol_pos_prior'])}"
        f"_branchweight_{condition_token(rowcol_metadata['rowcol_branch_weight'])}"
        f"_epochs_{args.epochs}"
        f"_lr_{args.lr_max:g}"
        f"_wd_{args.wd:g}"
    )
    if args.eval_protocol == "train_val_test":
        filename += f"_eval_train_val_test_val_{args.val_ratio:g}_split_{args.split_seed}"
    if collision_suffix is not None:
        filename += f"_run_{collision_suffix}"
    return f"{filename}.json"


def save_results_without_overwrite(results, args, rowcol_metadata):
    """Write results to a new file, preserving any existing experiment output."""
    collision_suffix = None
    collision_index = 0

    while True:
        filename = build_result_filename(args, rowcol_metadata, collision_suffix)
        result_path = os.path.join(args.results_dir, filename)
        try:
            with open(result_path, "x") as f:
                json.dump(results, f, indent=4)
            return result_path
        except FileExistsError:
            collision_index += 1
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            collision_suffix = f"{timestamp}-{collision_index}"


def clone_state_dict_to_cpu(state_dict):
    """Recursively clone a (possibly nested) state dict with all tensors moved to CPU.

    Used for both model and optimizer state dicts so the saved/retained copy
    cannot be mutated by ongoing training (no reference aliasing) and does not
    hold extra GPU memory.
    """
    if torch.is_tensor(state_dict):
        return state_dict.detach().cpu().clone()
    if isinstance(state_dict, dict):
        return {key: clone_state_dict_to_cpu(value) for key, value in state_dict.items()}
    if isinstance(state_dict, list):
        return [clone_state_dict_to_cpu(value) for value in state_dict]
    if isinstance(state_dict, tuple):
        return tuple(clone_state_dict_to_cpu(value) for value in state_dict)
    return state_dict


def reserve_checkpoint_path(args):
    """Atomically claim a checkpoint filename unique to this run.

    Creating the file with O_CREAT|O_EXCL guarantees two runs started with the
    same args never end up writing to the same path, even if they race.
    """
    checkpoint_name = f"{args.name}_seed_{args.seed}"
    if args.eval_protocol == "train_val_test":
        checkpoint_name += f"_eval_train_val_test_val_{args.val_ratio:g}_split_{args.split_seed}"
    filename = f"{checkpoint_name}_best.pt"
    suffix = 0
    while True:
        checkpoint_path = os.path.join(args.checkpoint_dir, filename)
        try:
            fd = os.open(checkpoint_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return checkpoint_path
        except FileExistsError:
            suffix += 1
            filename = f"{checkpoint_name}_best_run_{suffix}.pt"


def atomic_save_checkpoint(checkpoint, checkpoint_path, reserved_file=False):
    """Write checkpoint to a temp file in the same directory, then atomically replace checkpoint_path.

    On a Python exception, discard the temp file. A failed first save also
    removes this run's unchanged, empty reservation; later failures leave the
    previously saved checkpoint untouched. SIGKILL cannot run this cleanup.
    """
    directory = os.path.dirname(checkpoint_path) or "."
    tmp_path = None
    reservation = None
    try:
        if reserved_file:
            reservation = os.stat(checkpoint_path)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_ckpt_", suffix=".pt")
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            torch.save(checkpoint, tmp_file)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if reservation is not None:
            try:
                current = os.stat(checkpoint_path)
                if (current.st_dev, current.st_ino, current.st_size) == (
                    reservation.st_dev, reservation.st_ino, 0
                ):
                    os.remove(checkpoint_path)
            except FileNotFoundError:
                pass
        raise


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment identity
    parser.add_argument("--model", type=str, default="convmixer")
    parser.add_argument("--name", type=str, default="M0_ConvMixer_Baseline")
    parser.add_argument("--seed", type=int, default=0)

    # Directories
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--eval-protocol", default="legacy", choices=["legacy", "train_val_test"])
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--split-seed", default=42, type=int)

    # Data augmentation
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--scale", default=0.75, type=float)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--ra-m", default=8, type=int)
    parser.add_argument("--ra-n", default=1, type=int)
    parser.add_argument("--jitter", default=0.1, type=float)
    parser.add_argument("--workers", default=2, type=int)

    # Model hyperparameters
    parser.add_argument("--hdim", default=256, type=int)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--psize", default=2, type=int)
    parser.add_argument("--conv-ks", default=5, type=int)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--rowcol-combine", default="learnable", choices=["sum", "learnable", "softmax"])
    parser.add_argument("--rowcol-pre-norm", action="store_true")
    parser.add_argument("--local-scale-init", default=1.0, type=float)
    parser.add_argument("--rowcol-scale-init", default=0.1, type=float)
    parser.add_argument("--mul-pos-channel-specific", action="store_true")
    parser.add_argument("--no-mul-pre-norm", action="store_true")
    parser.add_argument("--no-mul-pos-prior", action="store_true")

    # Optimization
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr-max", default=0.03, type=float)
    parser.add_argument("--wd", default=0.01, type=float)
    parser.add_argument("--clip-norm", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1", help="Comma-separated GPU ids for DataParallel, e.g., 0,1")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)

    if torch.cuda.is_available():
        requested_gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        available_gpu_count = torch.cuda.device_count()
        gpu_ids = [i for i in requested_gpu_ids if i < available_gpu_count]

        if len(gpu_ids) == 0:
            raise RuntimeError(
                f"No valid GPU id found. Requested {requested_gpu_ids}, "
                f"but available GPU count is {available_gpu_count}."
            )

        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        requested_gpu_ids = []
        gpu_ids = []
        device = torch.device("cpu")

    amp_enabled = device.type == "cuda"

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Using GPU ids: {gpu_ids}")

    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")

    if args.eval_protocol == "train_val_test":
        trainloader, valloader, testloader = build_cifar_train_val_test_dataloaders(args)
    else:
        if args.dataset != "CIFAR10":
            raise ValueError("CIFAR100 requires --eval-protocol train_val_test")
        trainloader, testloader = build_cifar10_dataloaders(args)

    model = build_model(args)
    model = model.to(device)

    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    num_params = count_parameters(model)
    print(f"Trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.wd,
    )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=amp_enabled,
    )

    lr_schedule = build_lr_schedule(args)

    best_test_acc = 0.0
    best_val_acc = -1.0
    best_epoch = -1
    best_model_state = None
    checkpoint_path = None
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_test_acc = 0.0

    total_start = time.time()

    for epoch in range(args.epochs):
        model.train()

        start = time.time()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for step, (X, y) in enumerate(trainloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = lr_schedule(epoch + (step + 1) / len(trainloader))
            optimizer.param_groups[0]["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()

            if args.clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            train_correct += (output.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        if args.eval_protocol == "train_val_test":
            val_acc = evaluate(model, valloader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = clone_state_dict_to_cpu(model.state_dict())

                if args.save_checkpoint:
                    checkpoint = {
                        "epoch": best_epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": clone_state_dict_to_cpu(optimizer.state_dict()),
                        "best_val_acc": best_val_acc,
                        "args": vars(args),
                    }
                    first_save = checkpoint_path is None
                    if first_save:
                        checkpoint_path = reserve_checkpoint_path(args)
                    atomic_save_checkpoint(checkpoint, checkpoint_path, reserved_file=first_save)
        else:
            test_acc = evaluate(model, testloader, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch

                if args.save_checkpoint:
                    checkpoint = {
                        "epoch": best_epoch,
                        "model_state_dict": clone_state_dict_to_cpu(model.state_dict()),
                        "optimizer_state_dict": clone_state_dict_to_cpu(optimizer.state_dict()),
                        "best_test_acc": best_test_acc,
                        "args": vars(args),
                    }
                    first_save = checkpoint_path is None
                    if first_save:
                        checkpoint_path = reserve_checkpoint_path(args)
                    atomic_save_checkpoint(checkpoint, checkpoint_path, reserved_file=first_save)

        last_train_loss = train_loss
        last_train_acc = train_acc
        if args.eval_protocol == "legacy":
            last_test_acc = test_acc

        elapsed = time.time() - start

        evaluation_status = (
            f"Val Acc {val_acc:.4f} | Best {best_val_acc:.4f} @ {best_epoch}"
            if args.eval_protocol == "train_val_test"
            else f"Test Acc {test_acc:.4f} | Best {best_test_acc:.4f} @ {best_epoch}"
        )
        print(
            f"[{args.name}] "
            f"Epoch {epoch:03d}/{args.epochs - 1:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"{evaluation_status} | "
            f"LR {lr:.6f} | "
            f"Time {elapsed:.1f}s"
        )

    total_time = time.time() - total_start
    if args.eval_protocol == "train_val_test":
        if best_model_state is None:
            raise ValueError("train_val_test requires at least one epoch")
        model.load_state_dict(best_model_state)
        final_test_acc = evaluate(model, testloader, device)
        print(f"Final Test Acc {final_test_acc:.4f} (best epoch {best_epoch})")

    if checkpoint_path is not None:
        print(f"Best checkpoint (epoch {best_epoch}) is at {checkpoint_path}")
    rowcol_metadata = get_rowcol_metadata(args)

    results = {
        "model": args.model,
        "name": args.name,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hdim,
        "depth": args.depth,
        "patch_size": args.psize,
        "kernel_size": args.conv_ks,
        "image_size": args.image_size,
        "rowcol_combine": args.rowcol_combine,
        "local_scale_init": args.local_scale_init,
        "rowcol_scale_init": args.rowcol_scale_init,
        "mul_pos_channel_specific": args.mul_pos_channel_specific,
        "mul_pre_norm": not args.no_mul_pre_norm,
        **rowcol_metadata,
        "optimizer": "AdamW",
        "lr_max": args.lr_max,
        "weight_decay": args.wd,
        "num_params": num_params,
        "gpu_ids": gpu_ids,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "eval_protocol": args.eval_protocol,
        "best_epoch": best_epoch,
        "last_train_loss": last_train_loss,
        "last_train_acc": last_train_acc,
        "total_train_time_sec": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    if args.eval_protocol == "train_val_test":
        results.update({
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "train_samples": len(trainloader.dataset),
            "val_samples": len(valloader.dataset),
            "test_samples": len(testloader.dataset),
            "best_val_acc": best_val_acc,
            "final_test_acc": final_test_acc,
        })
    else:
        results.update({"best_test_acc": best_test_acc, "last_test_acc": last_test_acc})

    branch_scales = None
    model_for_analysis = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(model_for_analysis, "get_branch_scales"):
        branch_scales = model_for_analysis.get_branch_scales()
    results["branch_scales"] = branch_scales

    result_path = save_results_without_overwrite(results, args, rowcol_metadata)

    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
