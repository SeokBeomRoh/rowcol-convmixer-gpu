import os
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar import build_cifar10_dataloaders
from models.convmixer import ConvMixer
from utils.seed import set_seed
from utils.metrics import count_parameters, evaluate


def build_model(args):
    """
    Build model by name.

    Current stage:
        M0: convmixer
    """
    if args.model == "convmixer":
        return ConvMixer(
            dim=args.hdim,
            depth=args.depth,
            kernel_size=args.conv_ks,
            patch_size=args.psize,
            n_classes=10,
            in_channels=3,
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
    best_epoch = -1
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

        test_acc = evaluate(model, testloader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

            if args.save_checkpoint:
                ckpt_name = (
                    f"{args.name}"
                    f"_seed_{args.seed}"
                    f"_best.pt"
                )
                ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_test_acc": best_test_acc,
                        "args": vars(args),
                    },
                    ckpt_path,
                )

        last_train_loss = train_loss
        last_train_acc = train_acc
        last_test_acc = test_acc

        elapsed = time.time() - start

        print(
            f"[{args.name}] "
            f"Epoch {epoch:03d}/{args.epochs - 1:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Test Acc {test_acc:.4f} | "
            f"Best {best_test_acc:.4f} @ {best_epoch} | "
            f"LR {lr:.6f} | "
            f"Time {elapsed:.1f}s"
        )

    total_time = time.time() - total_start

    results = {
        "model": args.model,
        "name": args.name,
        "dataset": "CIFAR10",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hdim,
        "depth": args.depth,
        "patch_size": args.psize,
        "kernel_size": args.conv_ks,
        "optimizer": "AdamW",
        "lr_max": args.lr_max,
        "weight_decay": args.wd,
        "num_params": num_params,
        "gpu_ids": gpu_ids,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "last_train_loss": last_train_loss,
        "last_train_acc": last_train_acc,
        "last_test_acc": last_test_acc,
        "total_train_time_sec": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    filename = (
        f"{args.name}"
        f"_model_{args.model}"
        f"_seed_{args.seed}"
        f"_dim_{args.hdim}"
        f"_depth_{args.depth}"
        f"_psize_{args.psize}"
        f"_ks_{args.conv_ks}.json"
    )

    result_path = os.path.join(args.results_dir, filename)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
