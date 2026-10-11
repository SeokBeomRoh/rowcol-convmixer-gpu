#!/usr/bin/env python3
"""Small synthetic checks for the validation evaluation protocol."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
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


class ValidationProtocolChecks(unittest.TestCase):
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

            with patch.object(sys, "argv", arguments):
                checkpoint_path = train.save_checkpoint_without_overwrite(checkpoint, train.parse_args())
            self.assertNotEqual(Path(checkpoint_path), next(iter(checkpoint_paths)))

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
            self.assertNotIn("final_test_acc", result)


if __name__ == "__main__":
    unittest.main()
