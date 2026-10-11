import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def stratified_split_indices(targets, val_ratio, split_seed):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    targets = np.asarray(targets)
    generator = np.random.default_rng(split_seed)
    train_indices = []
    val_indices = []
    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        val_count = round(len(class_indices) * val_ratio)
        if not 0 < val_count < len(class_indices):
            raise ValueError("val_ratio leaves an empty train or validation class")
        shuffled = generator.permutation(class_indices)
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
    return train_indices, val_indices


def build_cifar10_dataloaders(args):
    """
    Build CIFAR-10 train/test dataloaders.

    The augmentation policy follows the user's original experiment code:
        - RandomResizedCrop
        - RandomHorizontalFlip
        - RandAugment
        - ColorJitter
        - RandomErasing
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            32,
            scale=(args.scale, 1.0),
            ratio=(1.0, 1.0),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(
            num_ops=args.ra_n,
            magnitude=args.ra_m,
        ),
        transforms.ColorJitter(
            args.jitter,
            args.jitter,
            args.jitter,
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=args.reprob),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    return trainloader, testloader


def build_cifar_train_val_test_dataloaders(args):
    dataset_class = {
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
    }[args.dataset]
    mean, std = (
        (CIFAR10_MEAN, CIFAR10_STD)
        if args.dataset == "CIFAR10"
        else (CIFAR100_MEAN, CIFAR100_STD)
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=args.reprob),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
    val_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=eval_transform)
    train_indices, val_indices = stratified_split_indices(train_dataset.targets, args.val_ratio, args.split_seed)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, train_indices), shuffle=True, **loader_options
    )
    valloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, val_indices), shuffle=False, **loader_options
    )
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_options)
    return trainloader, valloader, testloader
