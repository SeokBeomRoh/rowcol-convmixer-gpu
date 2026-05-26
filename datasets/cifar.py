import torch
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


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
