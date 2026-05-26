import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device) -> float:
    """
    Evaluate classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0

    amp_enabled = device.type == "cuda"

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(X)

        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total
