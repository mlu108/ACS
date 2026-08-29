# Adapted from ryeii/Representational-Homomorphism-for-Transformer-Language-Models
# (github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models),
# he_probe/gen_data_scan.py.

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple

Example = Tuple[List[str], List[str]]

PRIMITIVES: List[str] = ["walk", "run", "jump", "look"]
DIRECTIONS: List[str] = ["left", "right"]
MANNER_MODIFIERS: List[str] = ["opposite", "around"]
REPEATERS: List[str] = ["twice", "thrice"]
BINARY_CONNECTORS: List[str] = ["and", "after"]

_SCAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SCAN", "simple_split")


def _parse_scan_file(path: str) -> List[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            in_part, out_part = line.split(" OUT: ")
            inp = in_part[len("IN: "):].split()
            out = out_part.split()
            examples.append((inp, out))
    return examples


def random_scan_split(
    seed: int = 0,
    max_depth: int = 3,
    max_commands_per_depth: int | None = None,
    train_fraction: float = 0.8,
) -> Tuple[List[Example], List[Example]]:
    train_data = _parse_scan_file(os.path.join(_SCAN_DIR, "tasks_train_simple.txt"))
    test_data = _parse_scan_file(os.path.join(_SCAN_DIR, "tasks_test_simple.txt"))
    return train_data, test_data


def subsample_train_by_exposure(
    train_data: Sequence[Example],
    exposure_ratio: float = 1.0,
    seed: int = 0,
) -> List[Example]:
    if not (0.0 < exposure_ratio <= 1.0):
        raise ValueError("exposure_ratio must be in (0,1]")

    train_data = list(train_data)
    if exposure_ratio == 1.0:
        return train_data

    rng = random.Random(seed)
    rng.shuffle(train_data)
    k = max(1, min(int(round(len(train_data) * exposure_ratio)), len(train_data)))
    return train_data[:k]


def build_vocab_from_datasets(
    *datasets: Sequence[Example],
    add_pad: bool = True,
) -> List[str]:
    vocab: set = set()
    for dataset in datasets:
        for inp, out in dataset:
            vocab.update(inp)
            vocab.update(out)
    vocab_list = sorted(vocab)
    if add_pad:
        return [""] + vocab_list
    return vocab_list


def load_size_variation_split(p: int) -> Tuple[List[Example], List[Example]]:
    """Load official SCAN size-variation split for p% training data (p in {1,2,4,8,16,32,64,80}).
    p=80 uses the main simple split (tasks_train/test_simple.txt)."""
    if p == 80:
        return random_scan_split()
    size_dir = os.path.join(_SCAN_DIR, "size_variations")
    train_data = _parse_scan_file(os.path.join(size_dir, f"tasks_train_simple_p{p}.txt"))
    test_data = _parse_scan_file(os.path.join(size_dir, f"tasks_test_simple_p{p}.txt"))
    return train_data, test_data


def summarize_dataset(dataset: Sequence[Example]) -> Dict[str, float]:
    if len(dataset) == 0:
        return {"num_examples": 0, "avg_input_len": 0.0, "avg_output_len": 0.0, "max_input_len": 0, "max_output_len": 0}
    return {
        "num_examples": len(dataset),
        "avg_input_len": float(sum(len(x) for x, _ in dataset) / len(dataset)),
        "avg_output_len": float(sum(len(y) for _, y in dataset) / len(dataset)),
        "max_input_len": max(len(x) for x, _ in dataset),
        "max_output_len": max(len(y) for _, y in dataset),
    }
