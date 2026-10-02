import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed()
    print(f"Seed set to {SEED}")