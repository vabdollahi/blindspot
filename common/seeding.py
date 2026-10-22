from collections.abc import Sequence

import numpy as np

# Roadmap §3 seed ranges: train 0-99,999, validation 100,000-109,999.
TRAIN_SEED_START = 0
TRAIN_SEED_STOP = 100_000
VALIDATION_SEED_START = 100_000


def named_generators(seed: int, names: Sequence[str]) -> dict[str, np.random.Generator]:
    """Create reproducible, independent NumPy generators for named purposes."""
    purpose_names = tuple(names)
    if any(not isinstance(name, str) or not name for name in purpose_names):
        raise ValueError("Generator names must be non-empty strings")
    if len(set(purpose_names)) != len(purpose_names):
        raise ValueError("Generator names must be unique")

    child_sequences = np.random.SeedSequence(seed).spawn(len(purpose_names))
    return {
        name: np.random.default_rng(child_sequence)
        for name, child_sequence in zip(purpose_names, child_sequences, strict=True)
    }
