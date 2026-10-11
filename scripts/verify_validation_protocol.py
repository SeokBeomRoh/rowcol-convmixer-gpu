#!/usr/bin/env python3
"""Small synthetic checks for the validation evaluation protocol."""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from datasets.cifar import build_cifar_train_val_test_dataloaders, stratified_split_indices


class FakeCifar:
    def __init__(self, root, train, download, transform):
        self.targets = [label for label in range(10) for _ in range(100)] if train else [0] * 20
        self.transform = transform

    def __len__(self):
        return len(self.targets)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(inputs)


class TinyBatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(self.bn(inputs))


class ValidationProtocolChecks(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_state_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def run_three_epoch_checkpoint(self):
        data = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10,
                             torch.tensor([0, 1] * 4))
        loader = DataLoader(data, batch_size=4)
        snapshots = []
        saved = []
        optimizer_ref = []
        original_adamw = optim.AdamW
        original_save = train.atomic_save_checkpoint

        def capture_optimizer(*args, **kwargs):
            optimizer = original_adamw(*args, **kwargs)
            optimizer_ref.append(optimizer)
            return optimizer

        def fake_evaluate(model, eval_loader, device):
            self.assertIs(eval_loader, loader)
            snapshots.append((train.clone_state_dict_to_cpu(model.state_dict()),
                              train.clone_state_dict_to_cpu(optimizer_ref[0].state_dict())))
            return (0.60, 0.75, 0.70, 0.50)[len(snapshots) - 1]

        def capture_save(checkpoint, path, **kwargs):
            original_save(checkpoint, path, **kwargs)
            saved.append((path, torch.load(path, map_location="cpu", weights_only=False)))

        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = [
                "train.py", "--eval-protocol", "train_val_test", "--epochs", "3",
                "--workers", "0", "--results-dir", temporary_dir,
                "--checkpoint-dir", temporary_dir, "--save-checkpoint",
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar_train_val_test_dataloaders", return_value=(loader, loader, loader)
            ), patch.object(train, "build_model", return_value=nn.DataParallel(TinyBatchNormModel())), patch.object(
                train.optim, "AdamW", side_effect=capture_optimizer
            ), patch.object(train, "evaluate", side_effect=fake_evaluate), patch.object(
                train, "atomic_save_checkpoint", side_effect=capture_save
            ):
                train.main()
            checkpoint_files = list(Path(temporary_dir).glob("*.pt"))
            final_checkpoint = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
        return snapshots, saved, checkpoint_files, final_checkpoint

    def test_stratified_split(self):
        for class_count, class_size in ((10, 5000), (100, 500)):
            targets = [label for label in range(class_count) for _ in range(class_size)]
            train_indices, val_indices = stratified_split_indices(targets, 0.1, 42)
            self.assertEqual(len(train_indices), 45000)
            self.assertEqual(len(val_indices), 5000)
            self.assertFalse(set(train_indices) & set(val_indices))
            self.assertEqual((train_indices, val_indices), stratified_split_indices(targets, 0.1, 42))
            self.assertEqual(set(Counter(targets[index] for index in val_indices).values()), {class_size // 10})

    def test_dataset_transforms(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train.parse_args()
        args.workers = 0
        for dataset_name in ("CIFAR10", "CIFAR100"):
            args.dataset = dataset_name
            with patch("datasets.cifar.torchvision.datasets.CIFAR10", FakeCifar), patch(
                "datasets.cifar.torchvision.datasets.CIFAR100", FakeCifar
            ):
                trainloader, valloader, testloader = build_cifar_train_val_test_dataloaders(args)
            self.assertEqual((len(trainloader.dataset), len(valloader.dataset), len(testloader.dataset)), (900, 100, 20))
            self.assertFalse(set(trainloader.dataset.indices) & set(valloader.dataset.indices))
            train_ops = trainloader.dataset.dataset.transform.transforms
            val_ops = valloader.dataset.dataset.transform.transforms
            test_ops = testloader.dataset.transform.transforms
            self.assertIsInstance(train_ops[0], transforms.RandomResizedCrop)
            self.assertEqual([type(op) for op in val_ops], [transforms.ToTensor, transforms.Normalize])
            self.assertEqual([type(op) for op in test_ops], [transforms.ToTensor, transforms.Normalize])
            self.assertEqual(val_ops[1].mean, test_ops[1].mean)
            self.assertEqual(val_ops[1].std, test_ops[1].std)

    def test_best_epoch_restore_checkpoint_and_json(self):
        data = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
        trainloader = DataLoader(data, batch_size=4)
        valloader = DataLoader(data, batch_size=4)
        testloader = DataLoader(data, batch_size=4)
        snapshots = []

        def fake_evaluate(model, loader, device):
            self.assertIs(loader, valloader if len(snapshots) < 2 else testloader)
            snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            return (0.8, 0.8, 0.3)[len(snapshots) - 1]

        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = [
                "train.py", "--eval-protocol", "train_val_test", "--epochs", "2",
                "--workers", "0", "--results-dir", temporary_dir,
                "--checkpoint-dir", temporary_dir, "--save-checkpoint",
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar_train_val_test_dataloaders", return_value=(trainloader, valloader, testloader)
            ), patch.object(train, "build_model", return_value=nn.DataParallel(TinyModel())), patch.object(
                train, "evaluate", side_effect=fake_evaluate
            ):
                train.main()

            self.assertEqual(len(snapshots), 3)
            for key in snapshots[0]:
                torch.testing.assert_close(snapshots[0][key], snapshots[2][key])
            result = json.loads(next(Path(temporary_dir).glob("*.json")).read_text())
            self.assertEqual((result["best_epoch"], result["best_val_acc"], result["final_test_acc"]), (0, 0.8, 0.3))
            self.assertEqual((result["train_samples"], result["val_samples"], result["test_samples"]), (8, 8, 8))
            self.assertEqual(result["eval_protocol"], "train_val_test")
            self.assertNotIn("best_test_acc", result)
            checkpoint = torch.load(next(Path(temporary_dir).glob("*.pt")), map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 0)
            for key in snapshots[0]:
                torch.testing.assert_close(checkpoint["model_state_dict"][key], snapshots[0][key])

            checkpoint_paths = set(Path(temporary_dir).glob("*.pt"))
            result_paths = set(Path(temporary_dir).glob("*.json"))
            snapshots.clear()
            arguments.remove("--save-checkpoint")
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar_train_val_test_dataloaders", return_value=(trainloader, valloader, testloader)
            ), patch.object(train, "build_model", return_value=TinyModel()), patch.object(
                train, "evaluate", side_effect=fake_evaluate
            ):
                train.main()
            self.assertEqual(len(snapshots), 3)
            for key in snapshots[0]:
                torch.testing.assert_close(snapshots[0][key], snapshots[2][key])
            self.assertEqual(set(Path(temporary_dir).glob("*.pt")), checkpoint_paths)
            self.assertEqual(len(set(Path(temporary_dir).glob("*.json")) - result_paths), 1)

            # A later run with the same name/seed/split must not reuse (or overwrite)
            # a checkpoint path already claimed by a previous run.
            with patch.object(sys, "argv", arguments):
                third_run_args = train.parse_args()
            third_run_path = train.reserve_checkpoint_path(third_run_args)
            self.assertNotIn(Path(third_run_path), checkpoint_paths)

    def test_legacy_protocol(self):
        data = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
        loader = DataLoader(data, batch_size=4)
        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = [
                "train.py", "--epochs", "2", "--workers", "0",
                "--results-dir", temporary_dir, "--checkpoint-dir", temporary_dir,
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar10_dataloaders", return_value=(loader, loader)
            ), patch.object(train, "build_model", return_value=TinyModel()), patch.object(
                train, "evaluate", side_effect=[0.4, 0.5]
            ):
                train.main()
            result = json.loads(next(Path(temporary_dir).glob("*.json")).read_text())
            self.assertEqual((result["best_epoch"], result["best_test_acc"], result["last_test_acc"]), (1, 0.5, 0.5))
            self.assertEqual(result["eval_protocol"], "legacy")
            self.assertNotIn("final_test_acc", result)

    def test_checkpoint_saved_immediately_and_survives_crash(self):
        """The checkpoint for the best-so-far epoch must already be on disk before
        training finishes, so a crash in a later epoch cannot destroy it."""
        data = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
        trainloader = DataLoader(data, batch_size=4)
        valloader = DataLoader(data, batch_size=4)
        testloader = DataLoader(data, batch_size=4)

        call_count = [0]

        def crashing_evaluate(model, loader, device):
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.8  # Epoch 0: new best, must be checkpointed immediately.
            raise RuntimeError("simulated crash during epoch 1")

        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = [
                "train.py", "--eval-protocol", "train_val_test", "--epochs", "2",
                "--workers", "0", "--results-dir", temporary_dir,
                "--checkpoint-dir", temporary_dir, "--save-checkpoint",
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar_train_val_test_dataloaders", return_value=(trainloader, valloader, testloader)
            ), patch.object(train, "build_model", return_value=TinyModel()), patch.object(
                train, "evaluate", side_effect=crashing_evaluate
            ):
                with self.assertRaises(RuntimeError):
                    train.main()

            checkpoint_files = list(Path(temporary_dir).glob("*.pt"))
            self.assertEqual(len(checkpoint_files), 1)
            self.assertEqual(list(Path(temporary_dir).glob(".tmp_ckpt_*")), [])
            checkpoint = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertEqual(checkpoint["best_val_acc"], 0.8)

    def test_atomic_save_checkpoint_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = os.path.join(temporary_dir, "ckpt.pt")
            train.atomic_save_checkpoint({"epoch": 0, "value": "good"}, checkpoint_path)
            self.assertEqual(torch.load(checkpoint_path, weights_only=False)["value"], "good")

            with patch.object(torch, "save", side_effect=RuntimeError("simulated write failure")):
                with self.assertRaises(RuntimeError):
                    train.atomic_save_checkpoint({"epoch": 1, "value": "bad"}, checkpoint_path)

            self.assertEqual(torch.load(checkpoint_path, weights_only=False)["value"], "good")
            self.assertEqual(list(Path(temporary_dir).glob(".tmp_ckpt_*")), [])

    def test_first_save_failure_removes_only_its_empty_reservation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.object(sys, "argv", ["train.py", "--checkpoint-dir", temporary_dir, "--name", "X"]):
                args = train.parse_args()
            other_run_path = train.reserve_checkpoint_path(args)
            train.atomic_save_checkpoint({"epoch": 3}, other_run_path, reserved_file=True)
            first_path = train.reserve_checkpoint_path(args)

            def fail_after_partial_write(checkpoint, file):
                file.write(b"partial checkpoint")
                raise RuntimeError("simulated write failure")

            with patch.object(torch, "save", side_effect=fail_after_partial_write):
                with self.assertRaises(RuntimeError):
                    train.atomic_save_checkpoint({"epoch": 0}, first_path, reserved_file=True)
            self.assertFalse(os.path.exists(first_path))
            self.assertEqual(torch.load(other_run_path, weights_only=False)["epoch"], 3)
            self.assertEqual(list(Path(temporary_dir).glob(".tmp_ckpt_*")), [])

    def test_second_best_replaces_checkpoint_without_extra_file(self):
        snapshots, saved, files, checkpoint = self.run_three_epoch_checkpoint()
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0][0], saved[1][0])
        self.assertEqual(len(files), 1)
        self.assertEqual((checkpoint["epoch"], checkpoint["best_val_acc"]), (1, 0.75))
        self.assert_state_equal(checkpoint["model_state_dict"], snapshots[1][0])
        self.assertFalse(torch.equal(checkpoint["model_state_dict"]["module.bn.num_batches_tracked"],
                                     snapshots[0][0]["module.bn.num_batches_tracked"]))

    def test_checkpoint_states_match_best_epoch(self):
        snapshots, saved, _, checkpoint = self.run_three_epoch_checkpoint()
        self.assertEqual((saved[0][1]["epoch"], saved[0][1]["best_val_acc"]), (0, 0.60))
        self.assert_state_equal(saved[0][1]["model_state_dict"], snapshots[0][0])
        self.assert_state_equal(saved[0][1]["optimizer_state_dict"], snapshots[0][1])
        self.assertEqual((checkpoint["epoch"], checkpoint["best_val_acc"]), (1, 0.75))
        self.assert_state_equal(checkpoint["model_state_dict"], snapshots[1][0])
        self.assert_state_equal(checkpoint["optimizer_state_dict"], snapshots[1][1])

    def test_best_restore_preserves_batchnorm_buffers(self):
        snapshots, _, _, checkpoint = self.run_three_epoch_checkpoint()
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            key = f"module.bn.{buffer_name}"
            self.assert_state_equal(snapshots[3][0][key], snapshots[1][0][key])
            self.assert_state_equal(checkpoint["model_state_dict"][key], snapshots[1][0][key])
        self.assertFalse(torch.equal(snapshots[3][0]["module.bn.num_batches_tracked"],
                                     snapshots[2][0]["module.bn.num_batches_tracked"]))

    def test_reserve_checkpoint_path_is_unique_across_runs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.object(sys, "argv", ["train.py", "--checkpoint-dir", temporary_dir, "--name", "X"]):
                args = train.parse_args()
            first_path = train.reserve_checkpoint_path(args)
            second_path = train.reserve_checkpoint_path(args)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(os.path.exists(first_path))
            self.assertTrue(os.path.exists(second_path))

    def test_clone_state_dict_to_cpu_is_independent_copy(self):
        model = TinyModel()
        cloned_model_state = train.clone_state_dict_to_cpu(model.state_dict())
        for tensor in cloned_model_state.values():
            self.assertEqual(tensor.device.type, "cpu")

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        for key, tensor in cloned_model_state.items():
            self.assertFalse(torch.equal(tensor, model.state_dict()[key]))

        optimizer = optim.AdamW(model.parameters(), lr=0.1)
        loss = model(torch.randn(2, 3)).sum()
        loss.backward()
        optimizer.step()
        cloned_optimizer_state = train.clone_state_dict_to_cpu(optimizer.state_dict())
        for param_state in cloned_optimizer_state["state"].values():
            for value in param_state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device.type, "cpu")

    def test_clone_state_dict_to_cpu_handles_nested_tuple(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = {"list": [torch.arange(3.0, device=device)],
                  "tuple": (torch.arange(2.0, device=device), {"x": torch.ones(2, device=device)})}
        copied = train.clone_state_dict_to_cpu(source)
        self.assertIsInstance(copied["tuple"], tuple)
        with torch.no_grad():
            source["list"][0].add_(10)
            source["tuple"][0].add_(10)
            source["tuple"][1]["x"].add_(10)
        for actual, original in ((copied["list"][0], source["list"][0]),
                                 (copied["tuple"][0], source["tuple"][0]),
                                 (copied["tuple"][1]["x"], source["tuple"][1]["x"])):
            self.assertEqual(actual.device.type, "cpu")
            self.assertNotEqual(actual.data_ptr(), original.data_ptr())
            self.assertFalse(torch.equal(actual, original))

    def test_optimizer_state_not_cloned_when_checkpoint_disabled(self):
        """Without --save-checkpoint, optimizer.state_dict() must never be cloned,
        since it is only needed for the checkpoint file. clone_state_dict_to_cpu
        recurses into its own (patched) name for nested tensors/dicts, so this
        checks for the optimizer's distinctive top-level signature rather than
        a raw call count."""
        data = TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4))
        trainloader = DataLoader(data, batch_size=4)
        valloader = DataLoader(data, batch_size=4)
        testloader = DataLoader(data, batch_size=4)

        optimizer_state_clone_attempts = []
        original_clone = train.clone_state_dict_to_cpu

        def spy_clone(state_dict):
            if isinstance(state_dict, dict) and "param_groups" in state_dict:
                optimizer_state_clone_attempts.append(state_dict)
            return original_clone(state_dict)

        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = [
                "train.py", "--eval-protocol", "train_val_test", "--epochs", "1",
                "--workers", "0", "--results-dir", temporary_dir,
                "--checkpoint-dir", temporary_dir,
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                train, "build_cifar_train_val_test_dataloaders", return_value=(trainloader, valloader, testloader)
            ), patch.object(train, "build_model", return_value=TinyModel()), patch.object(
                train, "evaluate", side_effect=[0.8, 0.8]
            ), patch.object(train, "clone_state_dict_to_cpu", side_effect=spy_clone):
                train.main()

        self.assertEqual(optimizer_state_clone_attempts, [])


if __name__ == "__main__":
    unittest.main()
