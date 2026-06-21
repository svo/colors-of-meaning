import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
