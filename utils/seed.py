import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility.

    deterministic=False is usually faster.
    deterministic=True may improve reproducibility but can slow training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
