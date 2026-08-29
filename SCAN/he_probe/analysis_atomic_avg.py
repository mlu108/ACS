# Adapted from ryeii/Representational-Homomorphism-for-Transformer-Language-Models
# (github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models),
# he_probe/analysis_atomic_avg.py.

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as _scipy_stats
from tqdm import tqdm

try:
    from .gen_data_scan import (
        PRIMITIVES,
        DIRECTIONS,
        MANNER_MODIFIERS,
        REPEATERS,
        BINARY_CONNECTORS,
        random_scan_split,
        subsample_train_by_exposure,
        load_size_variation_split,
        build_vocab_from_datasets,
    )
    from .transformers import DecoderOnlyTransformer
except ImportError:
    from gen_data_scan import (
        PRIMITIVES,
        DIRECTIONS,
        MANNER_MODIFIERS,
        REPEATERS,
        BINARY_CONNECTORS,
        random_scan_split,
        subsample_train_by_exposure,
        load_size_variation_split,
        build_vocab_from_datasets,
    )
    from transformers import DecoderOnlyTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All atomic concept tokens in canonical order
ATOMIC_TOKENS: List[str] = (
    list(PRIMITIVES)
    + list(DIRECTIONS)
    + list(MANNER_MODIFIERS)
    + list(REPEATERS)
    + list(BINARY_CONNECTORS)
)

# Partition for mean-centering: action = movement primitives (excl. "turn" if present)
ACTION_TOKENS: List[str] = [tok for tok in PRIMITIVES if tok != "turn"]
NON_ACTION_TOKENS: List[str] = [tok for tok in ATOMIC_TOKENS if tok not in set(ACTION_TOKENS)]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exposure_tag(exposure_ratio: float) -> str:
    return f"exposure_{str(exposure_ratio).replace('.', 'p')}"


# ============================================================
# Vocab / prompt helpers
# ============================================================

def make_shared_vocab(train_data, test_data):
    base_vocab = build_vocab_from_datasets(train_data, test_data, add_pad=False)
    vocab = ["<pad>", "<bos>", "<sep>", "<eos>"] + [
        tok for tok in base_vocab if tok not in {"<pad>", "<bos>", "<sep>", "<eos>"}
    ]
    token2id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, token2id


def build_prefix_ids(inp_tokens: List[str], token2id: Dict[str, int]) -> torch.Tensor:
    seq = ["<bos>"] + list(inp_tokens) + ["<sep>"]
    ids = [token2id[t] for t in seq]
    return torch.tensor(ids, dtype=torch.long)


def trim_at_eos_or_pad(seq, eos_id, pad_id):
    out = []
    for tok in seq:
        if tok == eos_id or tok == pad_id:
            break
        out.append(tok)
    return out


# ============================================================
# Data reconstruction
# ============================================================

def reconstruct_train_dev_test(
    seed: int,
    max_depth: int,
    max_commands_per_depth: int | None,
    train_fraction: float,
    exposure_ratio: float,
    dev_fraction: float,
    data_mode: str = "exposure",
    size_variation_p: int | None = None,
    dev_source: str = "from_train_split",
):
    if data_mode == "size_variation":
        full_train_data, test_data = load_size_variation_split(size_variation_p)
        exposed_train_data = list(full_train_data)
    else:
        full_train_data, test_data = random_scan_split(
            seed=seed,
            max_depth=max_depth,
            max_commands_per_depth=max_commands_per_depth,
            train_fraction=train_fraction,
        )
        exposed_train_data = subsample_train_by_exposure(
            full_train_data,
            exposure_ratio=exposure_ratio,
            seed=seed,
        )

    rng = random.Random(seed)
    exposed_train_data = list(exposed_train_data)
    rng.shuffle(exposed_train_data)

    if dev_source == "from_test_split":
        test_list = list(test_data)
        rng.shuffle(test_list)
        n_dev = max(1, int(len(test_list) * dev_fraction))
        dev_data = test_list[:n_dev]
        test_data = test_list[n_dev:]
        train_data = exposed_train_data
    else:
        n_dev = max(1, int(len(exposed_train_data) * dev_fraction))
        dev_data = exposed_train_data[:n_dev]
        train_data = exposed_train_data[n_dev:]
        if len(train_data) == 0:
            train_data = dev_data
            dev_data = exposed_train_data[:1]

    return full_train_data, train_data, dev_data, test_data


# ============================================================
# Model loading
# ============================================================

def infer_latest_checkpoint(results_dir: str, seed: int, d_model: int, n_heads: int, d_ff: int, n_layers: int) -> str:
    pattern = os.path.join(results_dir, f"seed{seed}_dmodel{d_model}_nheads{n_heads}_dff{d_ff}_layer{n_layers}.pt")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    pts = sorted(glob.glob(os.path.join(results_dir, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {results_dir}")
    return pts[-1]


def load_model(
    checkpoint_path: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    pad_id: int,
    max_len: int = 128,
):
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1,
        pad_id=pad_id,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and not all(isinstance(k, str) for k in state.keys()):
        # Try common string keys first
        extracted = False
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state:
                state = state[key]
                extracted = True
                break
        if not extracted:
            # Find first value that is a string-keyed dict (the actual state dict)
            for v in state.values():
                if isinstance(v, dict) and len(v) > 0 and all(isinstance(k, str) for k in v.keys()):
                    state = v
                    break
            else:
                raise ValueError(
                    f"Cannot find state_dict in checkpoint. Top-level keys: {list(state.keys())}"
                )
    model.load_state_dict(state)
    model.eval()
    return model


def extract_atomic_representations(
    model,
    train_data,
    token2id: Dict[str, int],
    n_layers: int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, int]]]:
    sums_by_layer = {layer: defaultdict(lambda: None) for layer in range(n_layers)}
    counts_by_layer = {layer: defaultdict(int) for layer in range(n_layers)}

    for inp, out in train_data:
        prefix_ids = build_prefix_ids(inp, token2id).unsqueeze(0).to(device)
        padding_mask = (prefix_ids != token2id["<pad>"])

        hidden_states = model.get_hidden_states(prefix_ids, padding_mask=padding_mask)

        # input-side positions inside [BOS] + inp + [SEP] are 1..len(inp)
        for layer in range(n_layers):
            h = hidden_states[layer][0]
            for j, tok in enumerate(inp):
                pos = 1 + j
                vec = h[pos].detach().cpu()

                prev = sums_by_layer[layer][tok]
                if prev is None:
                    sums_by_layer[layer][tok] = vec.clone()
                else:
                    sums_by_layer[layer][tok] = prev + vec
                counts_by_layer[layer][tok] += 1

    reps_by_layer = {layer: {} for layer in range(n_layers)}
    for layer in range(n_layers):
        for tok, vec_sum in sums_by_layer[layer].items():
            cnt = counts_by_layer[layer][tok]
            reps_by_layer[layer][tok] = vec_sum / max(cnt, 1)

    return reps_by_layer, counts_by_layer


@torch.no_grad()
def extract_atomic_representations_single_tok(
    model,
    token2id: Dict[str, int],
    n_layers: int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, int]]]:
    """Extract each atomic concept's representation via a single <sep> concept <sep> forward pass.

    One pass per concept token; hidden state at position 1 (the concept) is used for every layer.
    Returns a counts_by_layer where every present token has count 1.
    """
    reps_by_layer: Dict[int, Dict[str, torch.Tensor]] = {layer: {} for layer in range(n_layers)}
    counts_by_layer: Dict[int, Dict[str, int]] = {layer: {} for layer in range(n_layers)}

    sep_id = token2id.get("<sep>")
    if sep_id is None:
        raise ValueError("'<sep>' not found in vocab — cannot use single_tok_inference extraction")

    for tok in ATOMIC_TOKENS:
        if tok not in token2id:
            print(f"  warning: atomic token '{tok}' not in vocab — skipping")
            continue

        ids = torch.tensor(
            [sep_id, token2id[tok], sep_id],
            dtype=torch.long,
        ).unsqueeze(0).to(device)
        padding_mask = torch.ones_like(ids, dtype=torch.bool)

        hidden_states = model.get_hidden_states(ids, padding_mask=padding_mask)

        for layer in range(n_layers):
            h = hidden_states[layer][0]          # (seq_len=3, d_model)
            reps_by_layer[layer][tok] = h[1].detach().cpu()   # position 1 = concept token
            counts_by_layer[layer][tok] = 1

    return reps_by_layer, counts_by_layer


# ============================================================
# Atomic rep cache helpers
# ============================================================

def _atomic_reps_cache_path(checkpoint_path: str, extract_method: str) -> str:
    stem = os.path.splitext(checkpoint_path)[0]
    return f"{stem}_atomic_reps_{extract_method}.pt"


def save_atomic_reps(
    reps_by_layer: Dict[int, Dict[str, torch.Tensor]],
    path: str,
) -> None:
    cpu_reps = {
        layer: {tok: v.cpu() for tok, v in reps.items()}
        for layer, reps in reps_by_layer.items()
    }
    torch.save(cpu_reps, path)
    print(f"Saved atomic reps cache: {path}")


def load_atomic_reps(path: str) -> Dict[int, Dict[str, torch.Tensor]]:
    reps = torch.load(path, map_location="cpu")
    print(f"Loaded atomic reps cache: {path}")
    return reps


# ============================================================
# Correctness (generation) cache
#
# model.generate() per example (evaluate_examples_batched) is the single
# most expensive step in a run — but "correct" only depends on
# (checkpoint, exact example list), never on aggregation/dedup/mean_center/
# select_best_layer/etc. Those knobs all vary the SAME checkpoint's dev/test
# split across many separate script invocations (one per results_subdir), so
# without this cache each one re-runs generation from scratch on identical
# examples. Cached alongside the checkpoint, mirroring _atomic_reps_cache_path.
# ============================================================

def _correct_flags_cache_path(checkpoint_path: str, split_name: str) -> str:
    stem = os.path.splitext(checkpoint_path)[0]
    return f"{stem}_correct_{split_name}.json"


def _examples_fingerprint(examples: List[Tuple[List[str], List[str]]]) -> str:
    """Cheap order-sensitive fingerprint of an (input, output) example list, so a
    correctness cache is only reused when it was computed for this EXACT split in
    this EXACT order (index-aligned with the cached correct/[i] flags) — any change
    in data-generation args, split ordering, etc. invalidates it automatically."""
    h = hashlib.md5()
    for inp, out in examples:
        h.update(" ".join(inp).encode())
        h.update(b"\t")
        h.update(" ".join(out).encode())
        h.update(b"\n")
    return h.hexdigest()


def save_correct_flags_cache(
    path: str, examples: List[Tuple[List[str], List[str]]], correct_flags: List[bool],
) -> None:
    with open(path, "w") as f:
        json.dump({
            "n_examples": len(examples),
            "fingerprint": _examples_fingerprint(examples),
            "correct": [int(c) for c in correct_flags],
        }, f)
    print(f"Saved correctness cache: {path}")


def load_correct_flags_cache(
    path: str, examples: List[Tuple[List[str], List[str]]],
) -> Optional[List[bool]]:
    with open(path) as f:
        payload = json.load(f)
    if payload.get("n_examples") != len(examples) or payload.get("fingerprint") != _examples_fingerprint(examples):
        print(f"[correctness cache] stale (example list changed) — recomputing: {path}")
        return None
    print(f"Loaded correctness cache: {path}")
    return [bool(c) for c in payload["correct"]]


# ============================================================
# Test evaluation
# ============================================================

@torch.no_grad()
def evaluate_examples_batched(
    model,
    examples: List[Tuple[List[str], List[str]]],
    token2id: Dict[str, int],
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> List[bool]:
    """Batch correctness evaluation grouped by prefix length (no padding needed)."""
    groups: dict = {}
    for idx, (inp, out) in enumerate(examples):
        prefix_ids = build_prefix_ids(inp, token2id)
        plen = prefix_ids.size(0)
        gold = [token2id[t] for t in out]
        groups.setdefault(plen, []).append((idx, prefix_ids, gold))

    correct = [False] * len(examples)
    for plen, items in groups.items():
        idxs = [item[0] for item in items]
        prefixes = torch.stack([item[1] for item in items]).to(device)
        golds = [item[2] for item in items]
        generated = model.generate(
            prefix_ids=prefixes,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
        ).cpu().tolist()
        for orig_idx, pred_full, gold in zip(idxs, generated, golds):
            pred_trim = trim_at_eos_or_pad(pred_full[plen:], eos_id, pad_id)
            gold_trim = trim_at_eos_or_pad(gold, eos_id, pad_id)
            correct[orig_idx] = (pred_trim == gold_trim)
    return correct


def orthogonality(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> float:
    un = torch.norm(u).item()
    vn = torch.norm(v).item()
    if un < eps or vn < eps:
        return 0.0
    cos = torch.dot(u, v).item() / (un * vn + eps)
    cos = max(min(cos, 1.0), -1.0)
    return 1.0 - abs(cos)


def classify_scan_example(inp_tokens: List[str]) -> str:
    toks = set(inp_tokens)
    has_connector = any(tok in BINARY_CONNECTORS for tok in inp_tokens)
    has_repeater = any(tok in REPEATERS for tok in inp_tokens)
    has_manner = any(tok in MANNER_MODIFIERS for tok in inp_tokens)
    has_direction = any(tok in DIRECTIONS for tok in inp_tokens)

    if has_connector:
        return "binary"
    if has_repeater or has_manner or has_direction:
        return "unary"
    return "primitive"


_ATOMIC_SET = set(ATOMIC_TOKENS)
_ACTION_SET = set(ACTION_TOKENS)
_NON_ACTION_SET = set(NON_ACTION_TOKENS)


def mean_center_atomic_reps(
    reps_by_layer: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Mean-center atomic representations by action / non-action category.

    For each layer:
      - Action tokens (PRIMITIVES minus 'turn'): centered by mean of action reps
      - Non-action tokens (directions, manner, repeaters, connectors): centered by
        mean of non-action reps
      - Unknown tokens: passed through unchanged
    """
    centered: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer, reps in reps_by_layer.items():
        action_vecs = [reps[tok] for tok in ACTION_TOKENS if tok in reps]
        non_action_vecs = [reps[tok] for tok in NON_ACTION_TOKENS if tok in reps]

        mean_action = torch.stack(action_vecs).mean(dim=0) if action_vecs else None
        mean_non_action = torch.stack(non_action_vecs).mean(dim=0) if non_action_vecs else None

        centered[layer] = {}
        for tok, vec in reps.items():
            if tok in _ACTION_SET and mean_action is not None:
                centered[layer][tok] = vec - mean_action
            elif tok in _NON_ACTION_SET and mean_non_action is not None:
                centered[layer][tok] = vec - mean_non_action
            else:
                centered[layer][tok] = vec

    return centered


def count_primitives(inp_tokens: List[str], dedup: bool = False) -> int:
    """Count atomic-token occurrences in input.

    dedup=True mirrors no_duplicate_in_example_ortho: counts unique atomic tokens only,
    so "jump jump" → 1 instead of 2.
    """
    toks = [tok for tok in inp_tokens if tok in _ATOMIC_SET]
    return len(set(toks)) if dedup else len(toks)


def example_orthogonality_score(
    inp_tokens: List[str],
    atomic_reps: Dict[str, torch.Tensor],
    aggregation: str = "min",
    no_duplicate_in_example_ortho: bool = True,
) -> Tuple[float, int]:
    # Only consider input tokens that are known atomic concepts
    atomic_tok_set = set(ATOMIC_TOKENS)
    seen: set = set()
    vecs = []
    for tok in inp_tokens:
        if tok in atomic_tok_set:
            if no_duplicate_in_example_ortho and tok in seen:
                continue
            seen.add(tok)
            if tok not in atomic_reps:
                raise KeyError(f"Token {tok} missing from atomic representations")
            vecs.append(atomic_reps[tok])

    if aggregation == "cumulative_coherence":
        # For each concept i, compute its mean pairwise orthogonality with all other
        # concepts in the example.  The per-concept mean equals 1 - mean|cos(i,j)|,
        # so min(per-concept means) = 1 - max(mean |cos| per concept) = 1 - CI.
        # Storing (1 - CI) keeps the same convention as other aggregations:
        # low value = concepts are coherent = hard for compositional generalization.
        n = len(vecs)
        if n < 2:
            return 1.0, 0
        per_concept_means = []
        for i in range(n):
            others = [orthogonality(vecs[i], vecs[j]) for j in range(n) if j != i]
            per_concept_means.append(float(np.mean(others)))
        return float(np.min(per_concept_means)), n

    if aggregation == "cumulative_coherence_l2":
        # L2 variant: per-concept score = sqrt(mean_j |cos(i,j)|^4) across j≠i.
        # |cos(i,j)| = 1 - orthogonality(i,j).
        # Result = 1 - max_i(sqrt(mean_j |cos(i,j)|^4)).
        # Low value = concepts are coherent (concentrated alignment).
        n = len(vecs)
        if n < 2:
            return 1.0, 0
        per_concept_l2 = []
        for i in range(n):
            abs_cos = np.array([1.0 - orthogonality(vecs[i], vecs[j]) for j in range(n) if j != i])
            per_concept_l2.append(float(np.sqrt(np.mean(abs_cos ** 4))))
        return float(1.0 - max(per_concept_l2)), n

    pair_scores = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pair_scores.append(orthogonality(vecs[i], vecs[j]))

    if len(pair_scores) == 0:
        return 1.0, 0

    if aggregation == "mean":
        return float(np.mean(pair_scores)), len(pair_scores)
    if aggregation == "sum":
        return float(np.sum(pair_scores)), len(pair_scores)
    if aggregation == "min":
        return float(np.min(pair_scores)), len(pair_scores)
    if aggregation == "max":
        return float(np.max(pair_scores)), len(pair_scores)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def analyze_examples(
    model,
    examples,
    token2id: Dict[str, int],
    atomic_reps_by_layer: Dict[int, Dict[str, torch.Tensor]],
    max_new_tokens: int,
    aggregation: str = "mean",
    no_duplicate_in_example_ortho: bool = True,
    checkpoint_path: Optional[str] = None,
    split_name: Optional[str] = None,
    use_correctness_cache: bool = True,
):
    eos_id = token2id["<eos>"]
    pad_id = token2id["<pad>"]
    examples = list(examples)

    # correct only depends on (checkpoint, exact example list) — never on aggregation/
    # dedup/mean_center/etc. — so callers that pass checkpoint_path/split_name skip
    # model.generate() entirely on any re-run of this checkpoint's dev/test split
    # (see the "Correctness (generation) cache" section above for why this matters).
    cache_path = (
        _correct_flags_cache_path(checkpoint_path, split_name)
        if use_correctness_cache and checkpoint_path is not None and split_name is not None
        else None
    )
    correct_flags = load_correct_flags_cache(cache_path, examples) if cache_path and os.path.exists(cache_path) else None

    if correct_flags is None:
        correct_flags = evaluate_examples_batched(
            model=model,
            examples=examples,
            token2id=token2id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
        )
        if cache_path is not None:
            save_correct_flags_cache(cache_path, examples, correct_flags)

    results_by_layer = {}

    for layer, atomic_reps in tqdm(atomic_reps_by_layer.items(), desc="layers", leave=False):
        rows = []
        for idx, (inp, out) in enumerate(tqdm(examples, desc=f"layer {layer}", leave=False)):
            n_prim = count_primitives(inp, dedup=no_duplicate_in_example_ortho)
            if n_prim <= 1:
                continue

            ex_type = classify_scan_example(inp)

            ortho_score, n_pairs = example_orthogonality_score(
                inp_tokens=inp,
                atomic_reps=atomic_reps,
                aggregation=aggregation,
                no_duplicate_in_example_ortho=no_duplicate_in_example_ortho,
            )

            rows.append(
                {
                    "index": idx,
                    "input": " ".join(inp),
                    "output": " ".join(out),
                    "type": ex_type,
                    "n_primitives": n_prim,
                    "correct": int(correct_flags[idx]),
                    "orthogonality": ortho_score,
                    "n_pairs": n_pairs,
                }
            )

        results_by_layer[layer] = rows

    return results_by_layer


# keep old name as alias for backward compatibility
analyze_test_examples = analyze_examples


# ============================================================
# Metrics / binning
# ============================================================

# ── Percentile binning (original) ────────────────────────────────────────────

def _cumulative_bins_pct(rows: List[Dict], sort_key, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=sort_key)
    n = len(rows)
    percentiles = np.arange(n) / n
    edges = np.arange(0.0, 1.0 + 1e-9, bin_width)
    centers, ys = [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        idx = np.where(percentiles <= hi)[0] if i == len(edges) - 2 else np.where(percentiles < hi)[0]
        if len(idx) == 0:
            continue
        centers.append((edges[i] + hi) / 2.0)
        ys.append(np.mean([rows[j]["correct"] for j in idx]))
    return np.array(centers), np.array(ys)


def cumulative_mean_accuracy(sorted_rows: List[Dict], bin_width: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    if not sorted_rows:
        return np.array([]), np.array([])
    return _cumulative_bins_pct(sorted_rows, lambda r: r["orthogonality"], bin_width)


def perfect_cumulative_mean_accuracy(rows: List[Dict], bin_width: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([]), np.array([])
    return _cumulative_bins_pct(rows, lambda r: r["correct"], bin_width)


def cumulative_auc(sorted_rows: List[Dict]) -> float:
    if not sorted_rows:
        return float("nan")
    xs, ys = cumulative_mean_accuracy(sorted_rows)
    return float(np.trapz(ys, xs))


def perfect_cumulative_auc(rows: List[Dict]) -> float:
    if not rows:
        return float("nan")
    xs, ys = perfect_cumulative_mean_accuracy(rows)
    return float(np.trapz(ys, xs))


def noncumulative_fixedwidth_accuracy(
    sorted_rows: List[Dict], bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sorted_rows:
        return np.array([]), np.array([]), np.array([])
    rows = sorted(sorted_rows, key=lambda r: r["orthogonality"])
    n = len(rows)
    percentiles = np.arange(n) / n
    edges = np.arange(0.0, 1.0 + 1e-9, bin_width)
    centers, mean_acc, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idx = (np.where((percentiles >= lo) & (percentiles <= hi))[0]
               if i == len(edges) - 2
               else np.where((percentiles >= lo) & (percentiles < hi))[0])
        if len(idx) == 0:
            continue
        chunk = [rows[j] for j in idx]
        centers.append((lo + hi) / 2.0)
        mean_acc.append(np.mean([r["correct"] for r in chunk]))
        counts.append(len(chunk))
    return np.array(centers), np.array(mean_acc), np.array(counts)


# ── Fixed-range binning (actual orthogonality value bins) ────────────────────

def _cumulative_bins_range(rows: List[Dict], bin_width: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Cumulative accuracy with n_bins equal-width bins over the actual ortho range.
    Returns (xs, ys, min_ortho, max_ortho) so callers can set xlim correctly."""
    rows = sorted(rows, key=lambda r: r["orthogonality"])
    if not rows:
        return np.array([]), np.array([]), 0.0, 1.0
    ortho_vals = np.array([r["orthogonality"] for r in rows], dtype=float)
    min_o, max_o = float(ortho_vals[0]), float(ortho_vals[-1])
    n_bins = max(1, round(1.0 / bin_width))
    if max_o - min_o < 1e-9:
        return (np.array([(min_o + max_o) / 2.0]),
                np.array([np.mean([r["correct"] for r in rows])]),
                min_o, max_o)
    edges = np.linspace(min_o, max_o, n_bins + 1)
    centers, ys = [], []
    for i in range(n_bins):
        hi = float(edges[i + 1])
        mask = ortho_vals <= hi + 1e-9 if i == n_bins - 1 else ortho_vals < hi
        if not np.any(mask):
            continue
        centers.append(hi)
        ys.append(np.mean([rows[j]["correct"] for j in np.where(mask)[0]]))
    return np.array(centers), np.array(ys), min_o, max_o


def cumulative_mean_accuracy_fixedrange(
    sorted_rows: List[Dict], bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if not sorted_rows:
        return np.array([]), np.array([]), 0.0, 1.0
    return _cumulative_bins_range(sorted_rows, bin_width)


def perfect_cumulative_mean_accuracy_fixedrange(
    rows: List[Dict], bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same ortho thresholds as the fixedrange curve; y = accuracy of first-k from
    correctness-ascending sort (wrong-first lower-bound reference)."""
    if not rows:
        return np.array([]), np.array([])
    n_bins = max(1, round(1.0 / bin_width))
    rows_sorted_ortho = sorted(rows, key=lambda r: r["orthogonality"])
    ortho_vals = np.array([r["orthogonality"] for r in rows_sorted_ortho], dtype=float)
    min_o, max_o = float(ortho_vals[0]), float(ortho_vals[-1])
    if max_o - min_o < 1e-9:
        return np.array([(min_o + max_o) / 2.0]), np.array([np.mean([r["correct"] for r in rows])])
    rows_sorted_correct = sorted(rows, key=lambda r: r["correct"])
    edges = np.linspace(min_o, max_o, n_bins + 1)
    centers, ys = [], []
    for i in range(n_bins):
        hi = float(edges[i + 1])
        mask = ortho_vals <= hi + 1e-9 if i == n_bins - 1 else ortho_vals < hi
        k = int(np.sum(mask))
        if k == 0:
            continue
        centers.append(hi)
        ys.append(np.mean([rows_sorted_correct[j]["correct"] for j in range(k)]))
    return np.array(centers), np.array(ys)


def noncumulative_fixedrange_accuracy(
    sorted_rows: List[Dict], bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Non-cumulative bins over actual ortho value range.
    Returns (centers, mean_acc, counts, min_ortho, max_ortho)."""
    if not sorted_rows:
        return np.array([]), np.array([]), np.array([]), 0.0, 1.0
    rows = sorted(sorted_rows, key=lambda r: r["orthogonality"])
    ortho_vals = np.array([r["orthogonality"] for r in rows], dtype=float)
    min_o, max_o = float(ortho_vals[0]), float(ortho_vals[-1])
    n_bins = max(1, round(1.0 / bin_width))
    if max_o - min_o < 1e-9:
        return (np.array([(min_o + max_o) / 2.0]),
                np.array([np.mean([r["correct"] for r in rows])]),
                np.array([len(rows)]), min_o, max_o)
    edges = np.linspace(min_o, max_o, n_bins + 1)
    centers, mean_acc, counts = [], [], []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = ((ortho_vals >= lo) & (ortho_vals <= hi + 1e-9) if i == n_bins - 1
                else (ortho_vals >= lo) & (ortho_vals < hi))
        if not np.any(mask):
            continue
        chunk = [rows[j] for j in np.where(mask)[0]]
        centers.append((lo + hi) / 2.0)
        mean_acc.append(np.mean([r["correct"] for r in chunk]))
        counts.append(len(chunk))
    return np.array(centers), np.array(mean_acc), np.array(counts), min_o, max_o


def noncumulative_percentile_at_ortho(
    rows: List[Dict], bin_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count percentile bins placed at their actual mean orthogonality x-coordinate.

    Divides sorted rows into n_bins equal-count groups (n_bins = round(1/bin_width)).
    Each point: x = mean orthogonality of the bin, y = mean accuracy of the bin.
    This keeps every bin the same size while preserving the orthogonality axis scale.
    Returns (xs, ys, counts).
    """
    if not rows:
        return np.array([]), np.array([]), np.array([])
    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    n = len(rows_sorted)
    n_bins = max(1, round(1.0 / bin_width))
    xs, ys, counts = [], [], []
    for i in range(n_bins):
        lo_idx = int(round(i * n / n_bins))
        hi_idx = int(round((i + 1) * n / n_bins))
        if hi_idx <= lo_idx:
            continue
        chunk = rows_sorted[lo_idx:hi_idx]
        xs.append(float(np.mean([r["orthogonality"] for r in chunk])))
        ys.append(float(np.mean([r["correct"] for r in chunk])))
        counts.append(len(chunk))
    return np.array(xs), np.array(ys), np.array(counts)


def _cumulative_bins_pct_scaled(rows: List[Dict], bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative accuracy at equal-percentile boundaries; x = actual orthogonality at that boundary.

    Like cumulative_mean_accuracy (percentile bins, same count per bin) but x is placed at the
    real orthogonality value of the boundary row instead of the percentile rank.
    """
    rows = sorted(rows, key=lambda r: r["orthogonality"])
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])
    ortho_vals = np.array([r["orthogonality"] for r in rows], dtype=float)
    percentiles = np.arange(n) / n
    edges = np.arange(0.0, 1.0 + 1e-9, bin_width)
    xs_ortho, ys = [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        idx = np.where(percentiles <= hi)[0] if i == len(edges) - 2 else np.where(percentiles < hi)[0]
        if len(idx) == 0:
            continue
        xs_ortho.append(float(ortho_vals[idx[-1]]))
        ys.append(float(np.mean([rows[j]["correct"] for j in idx])))
    return np.array(xs_ortho), np.array(ys)


def _perfect_cumulative_bins_pct_scaled(rows: List[Dict], bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Same x positions as _cumulative_bins_pct_scaled; y from correctness-ascending sort."""
    rows_ortho = sorted(rows, key=lambda r: r["orthogonality"])
    n = len(rows_ortho)
    if n == 0:
        return np.array([]), np.array([])
    ortho_vals = np.array([r["orthogonality"] for r in rows_ortho], dtype=float)
    percentiles = np.arange(n) / n
    rows_correct = sorted(rows, key=lambda r: r["correct"])
    edges = np.arange(0.0, 1.0 + 1e-9, bin_width)
    xs_ortho, ys = [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        idx = np.where(percentiles <= hi)[0] if i == len(edges) - 2 else np.where(percentiles < hi)[0]
        if len(idx) == 0:
            continue
        k = len(idx)
        xs_ortho.append(float(ortho_vals[idx[-1]]))
        ys.append(float(np.mean([rows_correct[j]["correct"] for j in range(k)])))
    return np.array(xs_ortho), np.array(ys)


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    # scipy.pearsonr raises/warns on constant input (zero variance) — degenerate
    # bins (e.g. a single-example or all-same-accuracy n_primitives group) hit
    # this in practice, same failure mode pointbiserial_r_p already guards below.
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = _scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def pointbiserial_r_p(correct: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Point-biserial correlation between binary correct (0/1) and continuous x."""
    mask = np.isfinite(x) & np.isfinite(correct)
    correct, x = correct[mask], x[mask]
    if len(x) < 3 or len(np.unique(correct)) < 2:
        return float("nan"), float("nan")
    r, p = _scipy_stats.pointbiserialr(correct, x)
    return float(r), float(p)


# ============================================================
# IO helpers
# ============================================================

def save_rows_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plot helpers
# ============================================================

def _run_permutation_test(
    rows: List[Dict],
    bin_width: float,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Permutation test: shuffle orthogonality labels B times, build null AUC distributions.

    p-value = fraction of null AUCs <= observed AUC (one-tailed; lower AUC = better predictor).
    Null curve envelopes (mean +/- std across permutations) are returned for percentile and
    fixedrange variants. pct_scaled only gets a p-value since its x-positions shift per permutation.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    ortho_vals  = np.array([r["orthogonality"] for r in rows])
    correct_vals = np.array([r["correct"] for r in rows])

    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    obs_xs_pct, _ = cumulative_mean_accuracy(rows_sorted, bin_width=bin_width)
    obs_auc_pct    = cumulative_auc(rows_sorted)
    obs_xs_r, obs_ys_r_tmp, _, _ = cumulative_mean_accuracy_fixedrange(rows_sorted, bin_width=bin_width)
    obs_auc_r      = float(np.trapz(obs_ys_r_tmp, obs_xs_r)) if len(obs_xs_r) > 1 else float("nan")
    obs_xs_s, obs_ys_s = _cumulative_bins_pct_scaled(rows_sorted, bin_width)
    obs_auc_s = float(np.trapz(obs_ys_s, obs_xs_s)) if len(obs_xs_s) > 1 else float("nan")

    null_aucs_pct, null_aucs_r, null_aucs_s = [], [], []
    null_ys_pct_list, null_ys_r_list = [], []

    for _ in tqdm(range(n_permutations), desc="perm test", leave=False, disable=n_permutations == 0):
        perm_ortho = rng.permutation(ortho_vals)
        perm_rows  = sorted(
            [{"orthogonality": float(perm_ortho[i]), "correct": int(correct_vals[i])} for i in range(n)],
            key=lambda r: r["orthogonality"],
        )

        xp, yp = cumulative_mean_accuracy(perm_rows, bin_width=bin_width)
        ap = float(np.trapz(yp, xp)) if len(xp) > 1 else float("nan")
        null_aucs_pct.append(ap)
        if len(yp) == len(obs_xs_pct):
            null_ys_pct_list.append(yp)

        xrp, yrp, _, _ = cumulative_mean_accuracy_fixedrange(perm_rows, bin_width=bin_width)
        arp = float(np.trapz(yrp, xrp)) if len(xrp) > 1 else float("nan")
        null_aucs_r.append(arp)
        if len(yrp) == len(obs_xs_r):
            null_ys_r_list.append(yrp)

        xsp, ysp = _cumulative_bins_pct_scaled(perm_rows, bin_width)
        asp = float(np.trapz(ysp, xsp)) if len(xsp) > 1 else float("nan")
        null_aucs_s.append(asp)

    null_aucs_pct = np.array(null_aucs_pct)
    null_aucs_r   = np.array(null_aucs_r)
    null_aucs_s   = np.array(null_aucs_s)

    def _pval(null_arr, obs):
        fin = null_arr[np.isfinite(null_arr)]
        return float(np.mean(fin <= obs)) if (len(fin) > 0 and np.isfinite(obs)) else float("nan")

    n_pct = len(obs_xs_pct)
    n_r   = len(obs_xs_r)
    null_ys_pct_arr = np.array(null_ys_pct_list) if null_ys_pct_list else np.empty((0, max(n_pct, 1)))
    null_ys_r_arr   = np.array(null_ys_r_list)   if null_ys_r_list   else np.empty((0, max(n_r, 1)))

    return {
        "n_permutations": n_permutations,
        # percentile
        "null_xs_pct":   obs_xs_pct.tolist(),
        "null_mean_pct": np.nanmean(null_ys_pct_arr, axis=0).tolist() if null_ys_pct_list else [],
        "null_std_pct":  np.nanstd(null_ys_pct_arr,  axis=0).tolist() if null_ys_pct_list else [],
        "null_aucs_pct": null_aucs_pct.tolist(),
        "p_value_pct":   _pval(null_aucs_pct, obs_auc_pct),
        # fixedrange
        "null_xs_r":     obs_xs_r.tolist(),
        "null_mean_r":   np.nanmean(null_ys_r_arr, axis=0).tolist() if null_ys_r_list else [],
        "null_std_r":    np.nanstd(null_ys_r_arr,  axis=0).tolist() if null_ys_r_list else [],
        "null_aucs_r":   null_aucs_r.tolist(),
        "p_value_r":     _pval(null_aucs_r, obs_auc_r),
        # pct_scaled (p-value only; x-positions shift per permutation so no envelope)
        "null_aucs_s":   null_aucs_s.tolist(),
        "p_value_s":     _pval(null_aucs_s, obs_auc_s),
    }


def _pval_label(p: float, n: int) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p < 1.0 / n:
        return f"p<{1.0/n:.3f}"
    return f"p={p:.3f}"


def _add_null_envelope(ax, null_xs, null_mean, null_std):
    null_xs   = np.array(null_xs)
    null_mean = np.array(null_mean)
    null_std  = np.array(null_std)
    if null_xs.size == 0 or null_mean.size != null_xs.size:
        return
    ax.fill_between(null_xs, null_mean - 2 * null_std, null_mean + 2 * null_std,
                    alpha=0.15, color="gray", label="Random ±2σ")
    ax.plot(null_xs, null_mean, color="gray", linewidth=1, linestyle="--", label="Random mean")


def plot_cumulative_curve(
    rows: List[Dict],
    title: str,
    savepath: str,
    bin_width: float = 0.1,
    n_permutations: int = 1000,
    perm_seed: int = 0,
):
    out_dir = os.path.dirname(savepath)
    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    avg_acc = float(np.mean([r["correct"] for r in rows_sorted])) if rows_sorted else float("nan")

    _x_ortho  = np.array([r["orthogonality"] for r in rows], dtype=np.float64)
    _y_correct = np.array([r["correct"]      for r in rows], dtype=np.float64)
    pbc, pbc_p = pointbiserial_r_p(_y_correct, _x_ortho)
    _pbc_str = f"rpb={pbc:+.3f} {'p<0.01' if pbc_p < 0.01 else f'p={pbc_p:.3f}'}"

    perm = _run_permutation_test(rows, bin_width, n_permutations=n_permutations, seed=perm_seed)

    # ── percentile version ────────────────────────────────────────────────────
    xs, ys = cumulative_mean_accuracy(rows_sorted, bin_width=bin_width)
    auc = cumulative_auc(rows_sorted)
    xs_perf, ys_perf = perfect_cumulative_mean_accuracy(rows, bin_width=bin_width)
    perf_auc = perfect_cumulative_auc(rows)
    denom = perf_auc - avg_acc
    recovery = (auc - avg_acc) / denom if abs(denom) > 1e-9 else float("nan")
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    pp = 1.0 - auc / avg_acc if avg_acc > 1e-9 else float("nan")
    p_lbl = _pval_label(perm["p_value_pct"], n_permutations)

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_null_envelope(ax, perm["null_xs_pct"], perm["null_mean_pct"], perm["null_std_pct"])
    ax.plot(xs, ys, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc:.3f}, PP={pp:.3f}, {_pbc_str})")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Orthogonality rank percentile (0=lowest, 1=highest)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative.svg")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_null_envelope(ax, perm["null_xs_pct"], perm["null_mean_pct"], perm["null_std_pct"])
    ax.plot(xs, ys, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc:.3f}, PP={pp:.3f}, slope={slope:.3f}, {_pbc_str})")
    ax.plot(xs_perf, ys_perf, marker="o", markersize=2, linestyle="--",
            label=f"Perfect upper bound (AUC={perf_auc:.3f})")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Rank percentile (0=lowest, 1=highest)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_with_perfect.svg")); plt.close(fig)

    # ── fixed-range version ───────────────────────────────────────────────────
    xs_r, ys_r, min_o, max_o = cumulative_mean_accuracy_fixedrange(rows_sorted, bin_width=bin_width)
    auc_r = float(np.trapz(ys_r, xs_r)) if len(xs_r) > 1 else float("nan")
    xs_rp, ys_rp = perfect_cumulative_mean_accuracy_fixedrange(rows, bin_width=bin_width)
    perf_auc_r = float(np.trapz(ys_rp, xs_rp)) if len(xs_rp) > 1 else float("nan")
    x_range_r = max_o - min_o if max_o > min_o else float("nan")
    baseline_area_r = avg_acc * x_range_r if not np.isnan(x_range_r) else float("nan")
    denom_r = perf_auc_r - baseline_area_r
    recovery_r = (auc_r - baseline_area_r) / denom_r if abs(denom_r) > 1e-9 else float("nan")
    pp_r = 1.0 - auc_r / baseline_area_r if baseline_area_r > 1e-9 else float("nan")
    p_lbl_r = _pval_label(perm["p_value_r"], n_permutations)

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_null_envelope(ax, perm["null_xs_r"], perm["null_mean_r"], perm["null_std_r"])
    ax.plot(xs_r, ys_r, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc_r:.3f}, PP={pp_r:.3f}, {_pbc_str})")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl_r} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Orthogonality threshold (actual value)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_xlim(min_o, max_o)
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_fixedrange.svg")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _add_null_envelope(ax, perm["null_xs_r"], perm["null_mean_r"], perm["null_std_r"])
    ax.plot(xs_r, ys_r, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc_r:.3f}, PP={pp_r:.3f}, {_pbc_str})")
    ax.plot(xs_rp, ys_rp, marker="o", markersize=2, linestyle="--",
            label=f"Correct-asc. reference (AUC={perf_auc_r:.3f})")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl_r} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Orthogonality threshold (actual value)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_xlim(min_o, max_o)
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_with_perfect_fixedrange.svg")); plt.close(fig)

    # ── percentile-scaled-by-ortho version ───────────────────────────────────
    xs_s, ys_s   = _cumulative_bins_pct_scaled(rows_sorted, bin_width)
    xs_sp, ys_sp = _perfect_cumulative_bins_pct_scaled(rows, bin_width)
    auc_s        = float(np.trapz(ys_s, xs_s)) if len(xs_s) > 1 else float("nan")
    x_range_s    = float(xs_s[-1] - xs_s[0]) if len(xs_s) > 1 else float("nan")
    baseline_area_s = avg_acc * x_range_s if not np.isnan(x_range_s) else float("nan")
    pp_s = 1.0 - auc_s / baseline_area_s if baseline_area_s > 1e-9 else float("nan")
    p_lbl_s = _pval_label(perm["p_value_s"], n_permutations)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_s, ys_s, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc_s:.3f}, PP={pp_s:.3f}, {_pbc_str})")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl_s} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Orthogonality (actual value; points at equal-percentile boundaries)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_pct_scaled.svg")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_s, ys_s, marker="o", markersize=2,
            label=f"Orthogonality (AUC={auc_s:.3f}, PP={pp_s:.3f}, {_pbc_str})")
    ax.plot(xs_sp, ys_sp, marker="o", markersize=2, linestyle="--",
            label="Perfect upper bound")
    ax.axhline(avg_acc, linestyle=":", color="black", label=f"Avg acc = {avg_acc:.3f}")
    ax.text(0.98, 0.03, f"perm. test: {p_lbl_s} (n={n_permutations})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel("Orthogonality (actual value; points at equal-percentile boundaries)")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_with_perfect_pct_scaled.svg")); plt.close(fig)

    return {
        "auc": auc,
        "auc_fixedrange": auc_r,
        "auc_pct_scaled": auc_s,
        "predictive_power": pp,
        "predictive_power_fixedrange": pp_r,
        "predictive_power_pct_scaled": pp_s,
        "avg_acc": avg_acc,
        "perfect_auc": perf_auc,
        "recovery": recovery,
        "slope": slope,
        "permutation_test": perm,
        "pbc": pbc,
        "pbc_p": pbc_p,
    }


# ── Failure PR-AUC ────────────────────────────────────────────────────────────

def compute_failure_pr_auc_from_rows(
    rows: List[Dict],
    n_perm: int = 1000,
    n_boot: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """
    PR-AUC for failure prediction from rows with 'orthogonality' and 'correct'.

    Positive class = failure (correct == 0).
    failure_score = 1 - orthogonality, so lower orthogonality predicts failure.

    Statistical significance (both off by default via n_perm=0/n_boot=0 — pass
    the shell script's DO_PERMUTATION_TEST-derived value to enable):
      - perm_p: permutation test. Randomly permutes failure_score against the
        fixed correct/incorrect labels (i.e. shuffles which CI score goes with
        which example) n_perm times, recomputing AP each time as the null
        distribution; perm_p = fraction of null >= observed pr_auc.
      - *_ci_lo/_ci_hi: bootstrap CI (`ci`, default 95%) for the improvement in
        PR-AUC over the failure-rate baseline. Resamples (score, label) pairs
        WITH replacement n_boot times; failure_rate is recomputed per-resample
        along with pr_auc (both covary under resampling, which is correct since
        the baseline is a property of the resampled failure distribution, not a
        fixed external constant), giving CIs for both the raw improvement
        (pr_auc - failure_rate) and normalized_pr_auc.

    Returns scalar metrics only. Precision/recall arrays are NOT stored —
    recompute from rows if needed.
    """
    from sklearn.metrics import average_precision_score as _aps

    x = np.array([r["orthogonality"] for r in rows], dtype=np.float64)
    y_correct = np.array([r["correct"] for r in rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y_correct)
    x, y_correct = x[mask], y_correct[mask]

    nan = float("nan")
    if len(x) == 0 or len(np.unique(y_correct)) < 2:
        failure_rate = float(np.mean(1 - y_correct)) if len(y_correct) > 0 else nan
        return {
            "pr_auc": nan, "normalized_pr_auc": nan, "failure_rate": failure_rate, "perm_p": nan,
            "pr_auc_improvement_ci_lo": nan, "pr_auc_improvement_ci_hi": nan,
            "normalized_pr_auc_ci_lo": nan, "normalized_pr_auc_ci_hi": nan,
            "n_boot": n_boot, "ci": ci,
        }

    y_failure = (1 - y_correct).astype(int)
    failure_rate = float(np.mean(y_failure))
    failure_score = 1.0 - x
    n = len(y_failure)

    pr_auc = float(_aps(y_failure, failure_score))
    normalized_pr_auc = (
        float((pr_auc - failure_rate) / (1.0 - failure_rate)) if failure_rate < 1.0 else nan
    )

    rng = np.random.default_rng(seed)
    perm_aucs = np.array([
        float(_aps(y_failure, rng.permutation(failure_score)))
        for _ in tqdm(range(n_perm), desc="pr perm test", leave=False, disable=n_perm == 0)
    ])
    perm_p = float(np.mean(perm_aucs >= pr_auc)) if n_perm > 0 else nan

    boot_rng = np.random.default_rng(seed + 1)
    boot_improvement = np.full(n_boot, nan)
    boot_normalized = np.full(n_boot, nan)
    for i in tqdm(range(n_boot), desc="pr bootstrap CI", leave=False, disable=n_boot == 0):
        idx = boot_rng.integers(0, n, size=n)
        yb, sb = y_failure[idx], failure_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        fr_b = float(np.mean(yb))
        pr_b = float(_aps(yb, sb))
        boot_improvement[i] = pr_b - fr_b
        if fr_b < 1.0:
            boot_normalized[i] = (pr_b - fr_b) / (1.0 - fr_b)

    alpha_pct = 100.0 * (1.0 - ci) / 2.0

    def _pctile_ci(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return nan, nan
        return float(np.percentile(finite, alpha_pct)), float(np.percentile(finite, 100.0 - alpha_pct))

    imp_lo, imp_hi = _pctile_ci(boot_improvement) if n_boot > 0 else (nan, nan)
    norm_lo, norm_hi = _pctile_ci(boot_normalized) if n_boot > 0 else (nan, nan)

    return {
        "pr_auc": pr_auc,
        "normalized_pr_auc": normalized_pr_auc,
        "failure_rate": failure_rate,
        "perm_p": perm_p,
        "pr_auc_improvement_ci_lo": imp_lo,
        "pr_auc_improvement_ci_hi": imp_hi,
        "normalized_pr_auc_ci_lo": norm_lo,
        "normalized_pr_auc_ci_hi": norm_hi,
        "n_boot": n_boot,
        "ci": ci,
    }


def plot_failure_pr_curve_from_rows(
    rows: List[Dict],
    title: str,
    out_dir: str,
    n_perm: int = 1000,
    n_boot: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """
    Compute failure PR-AUC, save the curve to out_dir/failure_pr_curve.svg, and return
    scalar metrics (pr_auc, normalized_pr_auc, failure_rate, perm_p, and — when
    n_boot > 0 — the bootstrap CI fields; see compute_failure_pr_auc_from_rows).
    """
    from sklearn.metrics import precision_recall_curve as _prc

    metrics = compute_failure_pr_auc_from_rows(rows, n_perm=n_perm, n_boot=n_boot, ci=ci, seed=seed)

    x = np.array([r["orthogonality"] for r in rows], dtype=np.float64)
    y_correct = np.array([r["correct"] for r in rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y_correct)
    x, y_correct = x[mask], y_correct[mask]

    if len(x) == 0 or len(np.unique(y_correct)) < 2:
        return metrics

    y_failure = (1 - y_correct).astype(int)
    precision, recall, _ = _prc(y_failure, 1.0 - x)

    pr_auc = metrics["pr_auc"]
    norm_pr_auc = metrics["normalized_pr_auc"]
    failure_rate = metrics["failure_rate"]
    perm_p = metrics["perm_p"]
    p_str = f", p(perm)={perm_p:.3f}" if np.isfinite(perm_p) else ""
    ci_lo, ci_hi = metrics["normalized_pr_auc_ci_lo"], metrics["normalized_pr_auc_ci_hi"]
    ci_str = f"\nnorm {int(ci * 100)}% CI=[{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else ""

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, linewidth=2.5,
            label=f"Geometry ranking\nPR-AUC={pr_auc:.3f}, norm={norm_pr_auc:.3f}{p_str}{ci_str}")
    ax.axhline(failure_rate, linestyle="--", linewidth=1.8, color="black",
               label=f"Random baseline={failure_rate:.3f}")
    ax.set_xlabel("Recall: fraction of all failures found", fontsize=12)
    ax.set_ylabel("Precision: fraction of predicted failures that fail", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "failure_pr_curve.svg"), bbox_inches="tight")
    plt.close(fig)

    return metrics


def plot_failure_pr_curve_by_len(
    rows: List[Dict],
    title: str,
    out_dir: str,
    n_perm: int = 1000,
    n_boot: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """
    Failure PR-AUC grouped by n_primitives.
    Saves one curve per group + an overlay plot.
    Returns {n_prim: {pr_auc, normalized_pr_auc, failure_rate, perm_p, ...}} (plus the
    bootstrap CI fields when n_boot > 0; see compute_failure_pr_auc_from_rows).
    """
    from sklearn.metrics import precision_recall_curve as _prc

    os.makedirs(out_dir, exist_ok=True)
    groups: Dict = defaultdict(list)
    for r in rows:
        groups[r["n_primitives"]].append(r)

    metrics = {}
    overlay_data = {}

    for n_prim in sorted(groups.keys()):
        grp = groups[n_prim]
        scalars = compute_failure_pr_auc_from_rows(grp, n_perm=n_perm, n_boot=n_boot, ci=ci, seed=seed)
        metrics[n_prim] = {**scalars, "n": len(grp)}

        # per-group curve
        x = np.array([r["orthogonality"] for r in grp], dtype=np.float64)
        y_correct = np.array([r["correct"] for r in grp], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y_correct)
        if mask.sum() > 0 and len(np.unique(y_correct[mask])) >= 2:
            precision, recall, _ = _prc((1 - y_correct[mask]).astype(int), 1.0 - x[mask])
            overlay_data[n_prim] = (recall, precision, scalars["pr_auc"])

            p_str = f", p={scalars['perm_p']:.3f}" if np.isfinite(scalars["perm_p"]) else ""
            _ci_lo, _ci_hi = scalars["normalized_pr_auc_ci_lo"], scalars["normalized_pr_auc_ci_hi"]
            ci_str = f", CI=[{_ci_lo:.3f},{_ci_hi:.3f}]" if np.isfinite(_ci_lo) else ""
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(recall, precision, linewidth=2.0,
                    label=f"PR-AUC={scalars['pr_auc']:.3f}, norm={scalars['normalized_pr_auc']:.3f}{p_str}{ci_str}")
            ax.axhline(scalars["failure_rate"], linestyle="--", color="black",
                       label=f"Baseline={scalars['failure_rate']:.3f}")
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title(f"{title} — n_prim={n_prim} (n={len(grp)})", fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"failure_pr_curve_nprim{n_prim}.svg"), bbox_inches="tight")
            plt.close(fig)

    # overlay: all groups on one plot
    if overlay_data:
        fig, ax = plt.subplots(figsize=(6, 5))
        for n_prim, (recall, precision, pr_auc) in overlay_data.items():
            ax.plot(recall, precision, linewidth=1.8,
                    label=f"n_prim={n_prim} (PR-AUC={pr_auc:.3f}, n={groups[n_prim].__len__()})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "failure_pr_curve_by_len.svg"), bbox_inches="tight")
        plt.close(fig)

    return metrics


def plot_cumulative_curve_by_subset(
    rows: List[Dict],
    title: str,
    savepath: str,
    bin_width: float = 0.1,
):
    subsets = [
        ("all", rows),
        ("primitive", [r for r in rows if r["type"] == "primitive"]),
        ("unary", [r for r in rows if r["type"] == "unary"]),
        ("binary", [r for r in rows if r["type"] == "binary"]),
    ]
    metrics = {}

    # ── percentile version ────────────────────────────────────────────────────
    plt.figure(figsize=(7, 5))
    for label, subset_rows in subsets:
        if not subset_rows:
            continue
        subset_sorted = sorted(subset_rows, key=lambda r: r["orthogonality"])
        xs, ys = cumulative_mean_accuracy(subset_sorted, bin_width=bin_width)
        auc = cumulative_auc(subset_sorted)
        avg_acc = float(np.mean([r["correct"] for r in subset_sorted]))
        plt.plot(xs, ys, marker="o", markersize=2, label=f"{label} (AUC={auc:.3f})")
        xs_perf, ys_perf = perfect_cumulative_mean_accuracy(subset_rows, bin_width=bin_width)
        perf_auc = perfect_cumulative_auc(subset_rows)
        plt.plot(xs_perf, ys_perf, markersize=2, linestyle="--", label=f"{label} perfect (AUC={perf_auc:.3f})")
        metrics[label] = {"auc": auc, "avg_acc": avg_acc, "perfect_auc": perf_auc}
    if rows:
        plt.axhline(np.mean([r["correct"] for r in rows]), linestyle=":", color="black", label="Overall avg acc")
    plt.xlabel("Rank percentile (0=lowest, 1=highest)")
    plt.ylabel("Cumulative exact-match accuracy")
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(savepath, dpi=200); plt.close()

    # ── fixed-range version ───────────────────────────────────────────────────
    all_ortho = [r["orthogonality"] for r in rows] if rows else [0.0, 1.0]
    global_min_o, global_max_o = min(all_ortho), max(all_ortho)
    fr_savepath = savepath.replace(".svg", "_fixedrange.svg")
    plt.figure(figsize=(7, 5))
    for label, subset_rows in subsets:
        if not subset_rows:
            continue
        subset_sorted = sorted(subset_rows, key=lambda r: r["orthogonality"])
        xs_r, ys_r, _, _ = cumulative_mean_accuracy_fixedrange(subset_sorted, bin_width=bin_width)
        auc_r = float(np.trapz(ys_r, xs_r)) if len(xs_r) > 1 else float("nan")
        plt.plot(xs_r, ys_r, marker="o", markersize=2, label=f"{label} (AUC={auc_r:.3f})")
        xs_rp, ys_rp = perfect_cumulative_mean_accuracy_fixedrange(subset_rows, bin_width=bin_width)
        perf_auc_r = float(np.trapz(ys_rp, xs_rp)) if len(xs_rp) > 1 else float("nan")
        plt.plot(xs_rp, ys_rp, markersize=2, linestyle="--", label=f"{label} ref (AUC={perf_auc_r:.3f})")
    if rows:
        plt.axhline(np.mean([r["correct"] for r in rows]), linestyle=":", color="black", label="Overall avg acc")
    plt.xlabel("Orthogonality threshold (actual value)")
    plt.ylabel("Cumulative exact-match accuracy")
    plt.xlim(global_min_o, global_max_o)
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(fr_savepath, dpi=200); plt.close()

    return metrics


def plot_noncumulative_curve(
    rows: List[Dict],
    title: str,
    savepath: str,
    bin_width: float = 0.1,
):
    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    avg_acc = float(np.mean([r["correct"] for r in rows_sorted])) if rows_sorted else float("nan")

    # ── percentile version ────────────────────────────────────────────────────
    xs, ys, counts = noncumulative_fixedwidth_accuracy(rows_sorted, bin_width=bin_width)
    r, p = pearson_r_p(xs, ys)
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o", label=f"Bin acc (R={r:.3f}, p={p:.3g})")
    plt.axhline(avg_acc, linestyle="--", label=f"Avg acc = {avg_acc:.3f}")
    plt.xlabel("Orthogonality rank percentile (bin center)")
    plt.ylabel("Mean exact-match accuracy")
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(savepath); plt.close()

    # ── fixed-range version ───────────────────────────────────────────────────
    xs_r, ys_r, counts_r, min_o, max_o = noncumulative_fixedrange_accuracy(rows_sorted, bin_width=bin_width)
    r_r, p_r = pearson_r_p(xs_r, ys_r)
    fr_savepath = savepath.replace(".svg", "_fixedrange.svg")
    plt.figure(figsize=(7, 5))
    plt.plot(xs_r, ys_r, marker="o", label=f"Bin acc (R={r_r:.3f}, p={p_r:.3g})")
    plt.axhline(avg_acc, linestyle="--", label=f"Avg acc = {avg_acc:.3f}")
    plt.xlabel("Orthogonality bin center (actual value)")
    plt.ylabel("Mean exact-match accuracy")
    plt.xlim(min_o, max_o)
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(fr_savepath); plt.close()

    return {"r": r, "p": p, "avg_acc": avg_acc, "n_nonempty_bins": int(len(xs))}


def plot_noncumulative_pct_at_ortho(
    rows: List[Dict],
    title: str,
    savepath: str,
    bin_width: float = 0.1,
) -> dict:
    """Noncumulative accuracy: equal-count percentile bins placed at actual orthogonality values.

    Saves two plots:
      *_pct_at_ortho.{ext}          – the new curve alone
      *_fixedrange_vs_pct.{ext}     – comparison overlay with fixed-range bins
    """
    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    avg_acc = float(np.mean([r["correct"] for r in rows_sorted])) if rows_sorted else float("nan")

    xs_p, ys_p, _ = noncumulative_percentile_at_ortho(rows_sorted, bin_width=bin_width)
    r_p, p_p = pearson_r_p(xs_p, ys_p)

    xs_r, ys_r, _, min_o, max_o = noncumulative_fixedrange_accuracy(rows_sorted, bin_width=bin_width)
    r_r, p_r = pearson_r_p(xs_r, ys_r)

    def _swap_ext(path, suffix):
        if path.endswith(".svg"):
            return path[:-4] + suffix + ".svg"
        return path + suffix

    # standalone percentile-at-ortho plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_p, ys_p, marker="o", label=f"Pct-at-ortho bins (R={r_p:.3f}, p={p_p:.3g})")
    ax.axhline(avg_acc, linestyle="--", color="gray", label=f"Avg acc = {avg_acc:.3f}")
    ax.set_xlabel("Orthogonality (mean of percentile bin)")
    ax.set_ylabel("Mean exact-match accuracy")
    ax.set_title(title)
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(_swap_ext(savepath, "_pct_at_ortho"))
    plt.close(fig)

    # comparison overlay: fixed-range vs. percentile-at-ortho
    all_x = np.concatenate([xs_r, xs_p]) if len(xs_r) and len(xs_p) else np.array([min_o, max_o])
    xlim = (float(all_x.min()), float(all_x.max()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_r, ys_r, marker="o", label=f"Fixed-range bins (R={r_r:.3f}, p={p_r:.3g})")
    ax.plot(xs_p, ys_p, marker="s", linestyle="--", label=f"Percentile-at-ortho bins (R={r_p:.3f}, p={p_p:.3g})")
    ax.axhline(avg_acc, linestyle=":", color="gray", label=f"Avg acc = {avg_acc:.3f}")
    ax.set_xlabel("Orthogonality (actual value)")
    ax.set_ylabel("Mean exact-match accuracy")
    ax.set_xlim(*xlim)
    ax.set_title(title); ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(_swap_ext(savepath, "_fixedrange_vs_pct"))
    plt.close(fig)

    return {"r_pct_at_ortho": r_p, "p_pct_at_ortho": p_p, "r_fixedrange": r_r, "p_fixedrange": p_r, "avg_acc": avg_acc}


def plot_noncumulative_curve_by_subset(
    rows: List[Dict],
    title: str,
    savepath: str,
    bin_width: float = 0.1,
):
    subsets = [
        ("all", rows),
        ("primitive", [r for r in rows if r["type"] == "primitive"]),
        ("unary", [r for r in rows if r["type"] == "unary"]),
        ("binary", [r for r in rows if r["type"] == "binary"]),
    ]
    metrics = {}

    # ── percentile version ────────────────────────────────────────────────────
    plt.figure(figsize=(7, 5))
    for label, subset_rows in subsets:
        if not subset_rows:
            continue
        subset_sorted = sorted(subset_rows, key=lambda r: r["orthogonality"])
        xs, ys, _ = noncumulative_fixedwidth_accuracy(subset_sorted, bin_width=bin_width)
        if len(xs) == 0:
            continue
        r, p = pearson_r_p(xs, ys)
        plt.plot(xs, ys, marker="o", label=f"{label} (R={r:.3f}, p={p:.3g})")
        metrics[label] = {"r": r, "p": p, "n_nonempty_bins": int(len(xs))}
    if rows:
        plt.axhline(np.mean([r["correct"] for r in rows]), linestyle="--", color="black", label="Overall avg acc")
    plt.xlabel("Orthogonality rank percentile (bin center)")
    plt.ylabel("Mean exact-match accuracy")
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(savepath); plt.close()

    # ── fixed-range version ───────────────────────────────────────────────────
    all_ortho = [r["orthogonality"] for r in rows] if rows else [0.0, 1.0]
    global_min_o, global_max_o = min(all_ortho), max(all_ortho)
    fr_savepath = savepath.replace(".svg", "_fixedrange.svg")
    plt.figure(figsize=(7, 5))
    for label, subset_rows in subsets:
        if not subset_rows:
            continue
        subset_sorted = sorted(subset_rows, key=lambda r: r["orthogonality"])
        xs_r, ys_r, _, _, _ = noncumulative_fixedrange_accuracy(subset_sorted, bin_width=bin_width)
        if len(xs_r) == 0:
            continue
        r_r, p_r = pearson_r_p(xs_r, ys_r)
        plt.plot(xs_r, ys_r, marker="o", label=f"{label} (R={r_r:.3f}, p={p_r:.3g})")
    if rows:
        plt.axhline(np.mean([r["correct"] for r in rows]), linestyle="--", color="black", label="Overall avg acc")
    plt.xlabel("Orthogonality bin center (actual value)")
    plt.ylabel("Mean exact-match accuracy")
    plt.xlim(global_min_o, global_max_o)
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(fr_savepath); plt.close()

    return metrics


def plot_cumulative_curve_by_len(
    rows: List[Dict],
    title: str,
    out_dir: str,
    bin_width: float = 0.1,
):
    """Overlay cumulative curves grouped by n_primitives; also save per-group individual plots."""
    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[r["n_primitives"]].append(r)

    metrics = {}

    # pre-compute per-group data once, reuse for overlay and individual plots
    group_data = {}
    for n_prim in sorted(groups.keys()):
        grp = groups[n_prim]
        if not grp:
            continue
        grp_sorted = sorted(grp, key=lambda r: r["orthogonality"])
        xs,    ys    = cumulative_mean_accuracy(grp_sorted, bin_width=bin_width)
        xs_r,  ys_r, _, _ = cumulative_mean_accuracy_fixedrange(grp_sorted, bin_width=bin_width)
        xs_s,  ys_s  = _cumulative_bins_pct_scaled(grp_sorted, bin_width)
        auc   = float(np.trapz(ys,   xs))   if len(xs)   > 1 else float("nan")
        auc_r = float(np.trapz(ys_r, xs_r)) if len(xs_r) > 1 else float("nan")
        auc_s = float(np.trapz(ys_s, xs_s)) if len(xs_s) > 1 else float("nan")
        avg_acc = float(np.mean([r["correct"] for r in grp]))
        xs_perf, ys_perf = perfect_cumulative_mean_accuracy(grp, bin_width=bin_width)
        perf_auc = perfect_cumulative_auc(grp)
        group_data[n_prim] = dict(
            grp=grp, grp_sorted=grp_sorted,
            xs=xs, ys=ys, xs_r=xs_r, ys_r=ys_r, xs_s=xs_s, ys_s=ys_s,
            auc=auc, auc_r=auc_r, auc_s=auc_s,
            avg_acc=avg_acc, xs_perf=xs_perf, ys_perf=ys_perf, perf_auc=perf_auc,
        )
        metrics[n_prim] = {"auc": auc, "auc_fixedrange": auc_r, "auc_pct_scaled": auc_s,
                           "avg_acc": avg_acc, "n": len(grp)}

    # overlay: percentile
    plt.figure(figsize=(8, 5))
    for n_prim, d in group_data.items():
        plt.plot(d["xs"], d["ys"], marker="o", markersize=2,
                 label=f"n_prim={n_prim} (AUC={d['auc']:.3f}, n={len(d['grp'])})")
    plt.xlabel("Orthogonality rank percentile (0=lowest, 1=highest)")
    plt.ylabel("Cumulative exact-match accuracy")
    plt.title(title); plt.legend(fontsize=8); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_by_len.svg")); plt.close()

    # overlay: fixedrange
    plt.figure(figsize=(8, 5))
    for n_prim, d in group_data.items():
        plt.plot(d["xs_r"], d["ys_r"], marker="o", markersize=2,
                 label=f"n_prim={n_prim} (AUC={d['auc_r']:.3f}, n={len(d['grp'])})")
    plt.xlabel("Orthogonality threshold (actual value)")
    plt.ylabel("Cumulative exact-match accuracy")
    plt.title(title); plt.legend(fontsize=8); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_fixedrange_by_len.svg")); plt.close()

    # overlay: pct_scaled
    plt.figure(figsize=(8, 5))
    for n_prim, d in group_data.items():
        plt.plot(d["xs_s"], d["ys_s"], marker="o", markersize=2,
                 label=f"n_prim={n_prim} (AUC={d['auc_s']:.3f}, n={len(d['grp'])})")
    plt.xlabel("Orthogonality (actual value; points at equal-percentile boundaries)")
    plt.ylabel("Cumulative exact-match accuracy")
    plt.title(title); plt.legend(fontsize=8); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_pct_scaled_by_len.svg")); plt.close()

    # per-group individual plots (all three variants)
    for n_prim, d in group_data.items():
        grp_label = f"{title} | n_primitives={n_prim} (n={len(d['grp'])})"

        plt.figure(figsize=(7, 5))
        plt.plot(d["xs"], d["ys"], marker="o", markersize=2, label=f"Orthogonality (AUC={d['auc']:.3f})")
        plt.plot(d["xs_perf"], d["ys_perf"], marker="o", markersize=2, linestyle="--",
                 label=f"Perfect upper bound (AUC={d['perf_auc']:.3f})")
        plt.axhline(d["avg_acc"], linestyle=":", color="gray", label=f"Avg acc = {d['avg_acc']:.3f}")
        plt.xlabel("Rank percentile (0=lowest, 1=highest)")
        plt.ylabel("Cumulative exact-match accuracy")
        plt.title(grp_label); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cumulative_nprim{n_prim}.svg")); plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(d["xs_r"], d["ys_r"], marker="o", markersize=2, label=f"Orthogonality (AUC={d['auc_r']:.3f})")
        plt.axhline(d["avg_acc"], linestyle=":", color="gray", label=f"Avg acc = {d['avg_acc']:.3f}")
        plt.xlabel("Orthogonality threshold (actual value)")
        plt.ylabel("Cumulative exact-match accuracy")
        plt.title(grp_label); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cumulative_fixedrange_nprim{n_prim}.svg")); plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(d["xs_s"], d["ys_s"], marker="o", markersize=2, label=f"Orthogonality (AUC={d['auc_s']:.3f})")
        plt.axhline(d["avg_acc"], linestyle=":", color="gray", label=f"Avg acc = {d['avg_acc']:.3f}")
        plt.xlabel("Orthogonality (actual value; points at equal-percentile boundaries)")
        plt.ylabel("Cumulative exact-match accuracy")
        plt.title(grp_label); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cumulative_pct_scaled_nprim{n_prim}.svg")); plt.close()

    return metrics


def plot_noncumulative_curve_by_len(
    rows: List[Dict],
    title: str,
    out_dir: str,
    bin_width: float = 0.1,
):
    """Overlay noncumulative curves grouped by n_primitives; also save per-group individual plots."""
    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[r["n_primitives"]].append(r)

    metrics = {}

    # overlay figure
    plt.figure(figsize=(8, 5))
    for n_prim in sorted(groups.keys()):
        grp = groups[n_prim]
        if not grp:
            continue
        grp_sorted = sorted(grp, key=lambda r: r["orthogonality"])
        xs, ys, _ = noncumulative_fixedwidth_accuracy(grp_sorted, bin_width=bin_width)
        if len(xs) == 0:
            continue
        r, p = pearson_r_p(xs, ys)
        plt.plot(xs, ys, marker="o", markersize=2, label=f"n_prim={n_prim} (R={r:.3f}, p={p:.3g}, n={len(grp)})")
        metrics[n_prim] = {"r": r, "p": p, "n": len(grp)}

    plt.xlabel("Orthogonality bin center")
    plt.ylabel("Mean exact-match accuracy")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noncumulative_by_len.svg"))
    plt.close()

    # per-group individual plots
    for n_prim in sorted(groups.keys()):
        grp = groups[n_prim]
        if not grp:
            continue
        grp_sorted = sorted(grp, key=lambda r: r["orthogonality"])
        xs, ys, _ = noncumulative_fixedwidth_accuracy(grp_sorted, bin_width=bin_width)
        if len(xs) == 0:
            continue
        r, p = pearson_r_p(xs, ys)
        avg_acc = float(np.mean([row["correct"] for row in grp]))

        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, marker="o", label=f"Bin acc (R={r:.3f}, p={p:.3g})")
        plt.axhline(avg_acc, linestyle="--", label=f"Avg acc = {avg_acc:.3f}")
        plt.xlabel("Orthogonality bin center")
        plt.ylabel("Mean exact-match accuracy")
        plt.title(f"{title} | n_primitives={n_prim} (n={len(grp)})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"noncumulative_nprim{n_prim}.svg"))
        plt.close()

    return metrics


def plot_mean_orthogonality_by_layer(
    results_by_layer: Dict[int, List[Dict]],
    title: str,
    savepath: str,
):
    layers = sorted(results_by_layer.keys())
    means = [
        float(np.mean([r["orthogonality"] for r in results_by_layer[l]]))
        if results_by_layer[l] else float("nan")
        for l in layers
    ]

    plt.figure(figsize=(7, 5))
    plt.plot(layers, means, marker="o")
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Mean orthogonality")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_orthogonality_heatmap(
    atomic_reps: Dict[str, torch.Tensor],
    title: str,
    savepath: str,
):
    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)

    if len(ordered) < 2:
        return

    tokens = [t for t, _ in ordered]
    categories = [c for _, c in ordered]
    n = len(tokens)

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = orthogonality(atomic_reps[tokens[i]], atomic_reps[tokens[j]])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Orthogonality")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    # draw boundaries when category changes
    for i in range(1, n):
        if categories[i] != categories[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def _ordered_concept_tokens(atomic_reps=None):
    """Return tokens in canonical category order, optionally filtering to those in atomic_reps."""
    groups = [
        [(tok, "primitive") for tok in PRIMITIVES],
        [(tok, "direction") for tok in DIRECTIONS],
        [(tok, "manner") for tok in MANNER_MODIFIERS],
        [(tok, "repeater") for tok in REPEATERS],
        [(tok, "connector") for tok in BINARY_CONNECTORS],
    ]
    ordered = [item for group in groups for item in group]
    if atomic_reps is not None:
        ordered = [(tok, cat) for tok, cat in ordered if tok in atomic_reps]
    return ordered


def plot_cooccurrence_heatmap(
    train_data: List,
    title: str,
    savepath: str,
):
    ordered = _ordered_concept_tokens()
    tokens = [t for t, _ in ordered]
    categories = [c for _, c in ordered]
    n = len(tokens)
    tok_set = set(tokens)

    matrix = np.zeros((n, n), dtype=np.int64)
    tok_idx = {tok: i for i, tok in enumerate(tokens)}

    for inp, _ in train_data:
        present = [tok for tok in inp if tok in tok_set]
        for i in range(len(present)):
            for j in range(len(present)):
                if i != j:
                    matrix[tok_idx[present[i]], tok_idx[present[j]]] += 1

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="Co-occurrence count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    for i in range(1, n):
        if categories[i] != categories[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_orthogonality_vs_cooccurrence(
    atomic_reps: Dict[str, torch.Tensor],
    train_data: List,
    title: str,
    savepath: str,
):
    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)
    tokens = [t for t, _ in ordered]
    tok_set = set(tokens)
    tok_idx = {tok: i for i, tok in enumerate(tokens)}
    n = len(tokens)

    # co-occurrence counts
    cooc = np.zeros((n, n), dtype=np.int64)
    for inp, _ in train_data:
        present = [tok for tok in inp if tok in tok_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = tok_idx[present[i]], tok_idx[present[j]]
                cooc[a, b] += 1
                cooc[b, a] += 1

    # collect upper-triangle pairs
    ortho_vals, cooc_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ortho_vals.append(orthogonality(atomic_reps[tokens[i]], atomic_reps[tokens[j]]))
            cooc_vals.append(int(cooc[i, j]))

    ortho_arr = np.array(ortho_vals)
    cooc_arr = np.array(cooc_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ortho_arr, cooc_arr, alpha=0.6, s=20)

    # regression line
    if len(ortho_arr) >= 2:
        coeffs = np.polyfit(ortho_arr, cooc_arr, 1)
        xs_line = np.linspace(ortho_arr.min(), ortho_arr.max(), 200)
        ax.plot(xs_line, np.polyval(coeffs, xs_line), color="black", linewidth=1.5, label="regression")
        r = np.corrcoef(ortho_arr, cooc_arr)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                verticalalignment="top", fontsize=10)

    ax.set_xlabel("Orthogonality")
    ax.set_ylabel("Co-occurrence count (training corpus)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def _build_neighbor_cooc(train_data: List, tok_set: set, tok_idx: dict, n: int) -> np.ndarray:
    """Count symmetric neighbor co-occurrences: adjacent positions where both tokens are concept tokens."""
    matrix = np.zeros((n, n), dtype=np.int64)
    for inp, _ in train_data:
        for k in range(len(inp) - 1):
            a, b = inp[k], inp[k + 1]
            if a in tok_set and b in tok_set:
                ia, ib = tok_idx[a], tok_idx[b]
                matrix[ia, ib] += 1
                matrix[ib, ia] += 1
    return matrix


def plot_neighbor_cooccurrence_heatmap(
    train_data: List,
    title: str,
    savepath: str,
):
    ordered = _ordered_concept_tokens()
    tokens = [t for t, _ in ordered]
    categories = [c for _, c in ordered]
    n = len(tokens)
    tok_set = set(tokens)
    tok_idx = {tok: i for i, tok in enumerate(tokens)}

    matrix = _build_neighbor_cooc(train_data, tok_set, tok_idx, n)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="Neighbor co-occurrence count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    for i in range(1, n):
        if categories[i] != categories[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_orthogonality_vs_neighbor_cooccurrence(
    atomic_reps: Dict[str, torch.Tensor],
    train_data: List,
    title: str,
    savepath: str,
):
    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)
    tokens = [t for t, _ in ordered]
    tok_set = set(tokens)
    tok_idx = {tok: i for i, tok in enumerate(tokens)}
    n = len(tokens)

    matrix = _build_neighbor_cooc(train_data, tok_set, tok_idx, n)

    ortho_vals, cooc_vals, labels = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            ortho_vals.append(orthogonality(atomic_reps[tokens[i]], atomic_reps[tokens[j]]))
            cooc_vals.append(int(matrix[i, j]))
            labels.append(f"{tokens[i]}-{tokens[j]}")

    ortho_arr = np.array(ortho_vals)
    cooc_arr = np.array(cooc_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ortho_arr, cooc_arr, alpha=0.6, s=20)

    if len(ortho_arr) >= 2:
        coeffs = np.polyfit(ortho_arr, cooc_arr, 1)
        xs_line = np.linspace(ortho_arr.min(), ortho_arr.max(), 200)
        ax.plot(xs_line, np.polyval(coeffs, xs_line), color="black", linewidth=1.5, label="regression")
        r = np.corrcoef(ortho_arr, cooc_arr)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                verticalalignment="top", fontsize=10)

    ax.set_xlabel("Orthogonality")
    ax.set_ylabel("Neighbor co-occurrence count (training corpus)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_concept_occurrence_counts(
    train_data: List,
    title: str,
    savepath: str,
):
    """Bar chart of total occurrence count per atomic concept token across all training examples."""
    ordered = _ordered_concept_tokens()
    tokens = [t for t, _ in ordered]
    categories = [c for _, c in ordered]
    tok_set = set(tokens)

    counts = defaultdict(int)
    for inp, _ in train_data:
        for tok in inp:
            if tok in tok_set:
                counts[tok] += 1

    vals = [counts[tok] for tok in tokens]
    category_colors = {
        "primitive": "steelblue", "direction": "darkorange",
        "manner": "green", "repeater": "purple", "connector": "red",
    }
    colors = [category_colors.get(cat, "gray") for cat in categories]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), 5))
    bars = ax.bar(tokens, vals, color=colors)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_ylabel("Total occurrences in training")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()
                       if cat in categories]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_occurrence_vs_mean_orthogonality(
    atomic_reps: Dict[str, torch.Tensor],
    train_data: List,
    title: str,
    savepath: str,
):
    """Scatter: total training occurrences of concept x vs sum of orthogonality(x, y) for all y != x."""
    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)
    tokens = [t for t, _ in ordered]
    tok_set = set(tokens)

    counts = defaultdict(int)
    for inp, _ in train_data:
        for tok in inp:
            if tok in tok_set and tok in atomic_reps:
                counts[tok] += 1

    sum_ortho = {}
    for tok in tokens:
        sum_ortho[tok] = sum(
            orthogonality(atomic_reps[tok], atomic_reps[other])
            for other in tokens if other != tok
        )

    xs = np.array([counts[tok] for tok in tokens], dtype=float)
    ys = np.array([sum_ortho[tok] for tok in tokens], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.7, s=40)
    for tok, x, y in zip(tokens, xs, ys):
        ax.annotate(tok, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    if len(xs) >= 2:
        coeffs = np.polyfit(xs, ys, 1)
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(xs_line, np.polyval(coeffs, xs_line), color="black", linewidth=1.5, label="regression")
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                verticalalignment="top", fontsize=10)

    ax.set_xlabel("Total occurrences in training (duplicates counted)")
    ax.set_ylabel("Sum of orthogonality to all other concepts")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_pca_2d_svg(
    atomic_reps: Dict[str, torch.Tensor],
    title: str,
    savepath: str,
):
    """2-D PCA vector plot: arrows from the PCA origin to each concept's projection.

    The origin is the centroid of all concept vectors (sklearn PCA centers the data),
    so each arrow shows the displacement of that concept from the mean in PC space.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"Skipping 2D PCA plot (missing dependency: {e})")
        return

    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)
    if len(ordered) < 2:
        return

    tokens     = [t for t, _ in ordered]
    categories = [c for _, c in ordered]

    matrix = np.stack([atomic_reps[t].numpy() for t in tokens])
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(matrix)          # (n_tokens, 2)
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / np.where(norms > 0, norms, 1)   # unit vectors
    var    = pca.explained_variance_ratio_

    category_colors = {
        "primitive": "#4e79a7",
        "direction": "#f28e2b",
        "manner":    "#59a14f",
        "repeater":  "#9467bd",
        "connector": "#e15759",
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    seen_cats: set = set()
    for i, (tok, cat) in enumerate(zip(tokens, categories)):
        x, y  = coords[i, 0], coords[i, 1]
        color = category_colors.get(cat, "gray")
        label = cat if cat not in seen_cats else "_nolegend_"
        seen_cats.add(cat)

        ax.annotate(
            "",
            xy=(x, y), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.5,
                mutation_scale=12,
            ),
            label=label,
        )
        # dot at tip
        ax.scatter(x, y, color=color, s=30, zorder=3)
        # token label slightly offset from tip
        offset_x = 0.015 * (coords[:, 0].max() - coords[:, 0].min())
        offset_y = 0.015 * (coords[:, 1].max() - coords[:, 1].min())
        ax.text(x + offset_x, y + offset_y, tok, fontsize=7.5, color=color,
                ha="left", va="bottom")

    # draw origin
    ax.scatter(0, 0, color="black", s=40, zorder=4, marker="+")

    # manual legend entries for categories
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=category_colors.get(cat, "gray"),
               lw=2, label=cat)
        for cat in ["primitive", "direction", "manner", "repeater", "connector"]
        if cat in seen_cats
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best")

    ax.axhline(0, color="lightgray", lw=0.6, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def save_plot_data_json(
    rows: List[Dict],
    atomic_reps: Dict[str, torch.Tensor],
    cumulative_title: str,
    cumulative_savepath: str,
    heatmap_title: str,
    heatmap_savepath: str,
    pca_title: str,
    pca_savepath: str,
    bin_width: float,
    out_json: str,
    cum_metrics: Dict = None,
    noncumulative_title: str = None,
    noncumulative_savepath: str = None,
    by_len_title: str = None,
    by_len_out_dir: str = None,
):
    """Serialize all data needed to regenerate cumulative, noncumulative, by_len, heatmap, and pca_2d plots."""
    ordered = _ordered_concept_tokens(atomic_reps=atomic_reps)
    tokens = [t for t, _ in ordered]
    categories = [c for _, c in ordered]
    n = len(tokens)

    matrix = [
        [orthogonality(atomic_reps[tokens[i]], atomic_reps[tokens[j]]) for j in range(n)]
        for i in range(n)
    ]
    vectors = [atomic_reps[t].numpy().tolist() for t in tokens]

    saved_rows = [
        {
            "orthogonality": float(r["orthogonality"]),
            "correct": int(r["correct"]),
            "type": str(r.get("type", "")),
            "n_primitives": int(r["n_primitives"]) if "n_primitives" in r else None,
        }
        for r in rows
    ]

    data = {
        "cumulative": {
            "rows": saved_rows,
            "title": cumulative_title,
            "savepath": cumulative_savepath,
            "bin_width": bin_width,
        },
        "heatmap": {
            "tokens": tokens,
            "categories": categories,
            "matrix": matrix,
            "title": heatmap_title,
            "savepath": heatmap_savepath,
        },
        "pca_2d": {
            "tokens": tokens,
            "categories": categories,
            "vectors": vectors,
            "title": pca_title,
            "savepath": pca_savepath,
        },
    }

    if cum_metrics is not None and "permutation_test" in cum_metrics:
        data["permutation_test"] = cum_metrics["permutation_test"]

    if noncumulative_title is not None:
        data["noncumulative"] = {
            "title": noncumulative_title,
            "savepath": noncumulative_savepath,
            "bin_width": bin_width,
        }

    if by_len_title is not None:
        data["by_len"] = {
            "title": by_len_title,
            "out_dir": by_len_out_dir,
            "bin_width": bin_width,
        }

    parent = os.path.dirname(out_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved plot data to: {out_json}")


def plot_accuracy_bar(
    rows: List[Dict],
    title: str,
    savepath: str,
):
    acc = float(np.mean([r["correct"] for r in rows])) if rows else float("nan")
    n = len(rows)

    plt.figure(figsize=(3, 5))
    bar = plt.bar(["all"], [acc], color="steelblue")
    if not np.isnan(acc):
        plt.text(
            bar[0].get_x() + bar[0].get_width() / 2,
            acc + 0.02,
            f"{acc:.3f}\nn={n}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.ylabel("Exact-match accuracy")
    plt.ylim(0, 1.15)
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


# keep old name as alias
plot_accuracy_by_subset_bar = plot_accuracy_bar


# ============================================================
# Error recall curve
# ============================================================

def plot_error_recall_curve(
    rows: List[Dict],
    title: str,
    savepath: str,
) -> Dict:
    """Cumulative error recall curve.

    Sort examples by 'orthogonality' ascending (lowest score = predicted hardest first).
    X = fraction of test examples included (0..1).
    Y = fraction of total errors captured among those examples.

    Random baseline: Y = X (diagonal) — including x% of examples at random captures x% of errors.
    A curve above the diagonal means errors are concentrated at low-score examples,
    i.e. the predictor successfully front-loads failures.

    Also writes a "plot_data_error_recall.json" file next to `savepath` containing
    the rows (orthogonality, correct) and metrics needed to regenerate this figure.
    """
    parent = os.path.dirname(savepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_json = os.path.join(parent, "plot_data_error_recall.json") if parent else "plot_data_error_recall.json"

    def _saved_rows(rs):
        return [{"orthogonality": float(r["orthogonality"]), "correct": int(r["correct"])} for r in rs]

    if not rows:
        with open(out_json, "w") as f:
            json.dump({"title": title, "savepath": savepath, "rows": []}, f, indent=2)
        return {"error_recall_auc_above_diagonal": float("nan"), "error_rate": float("nan")}

    rows_sorted = sorted(rows, key=lambda r: r["orthogonality"])
    n = len(rows_sorted)
    total_errors = sum(1 - r["correct"] for r in rows_sorted)

    if total_errors == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline (x=y)")
        ax.set_xlabel("Fraction of test examples included")
        ax.set_ylabel("Fraction of total errors captured")
        ax.set_title(title + " [no errors in this set]")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(savepath); plt.close(fig)
        with open(out_json, "w") as f:
            json.dump({
                "title": title,
                "savepath": savepath,
                "rows": _saved_rows(rows_sorted),
                "error_rate": 0.0,
            }, f, indent=2)
        return {"error_recall_auc_above_diagonal": float("nan"), "error_rate": 0.0}

    xs = np.arange(1, n + 1) / n
    ys = np.cumsum([1 - r["correct"] for r in rows_sorted]) / total_errors
    error_rate = total_errors / n

    # perfect predictor: all errors come first
    perfect_xs = np.array([0.0, error_rate, 1.0])
    perfect_ys = np.array([0.0, 1.0, 1.0])

    auc_above = float(np.trapz(ys - xs, xs))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5,
            label="Random baseline (x=y)")
    ax.plot(perfect_xs, perfect_ys, linestyle=":", color="green", linewidth=1.5,
            label=f"Perfect predictor (error rate={error_rate:.3f})")
    ax.plot(xs, ys, marker="o", markersize=2,
            label=f"Predictor (AUC above diag={auc_above:.3f})")
    ax.set_xlabel("Fraction of test examples included (sorted by score ↑)")
    ax.set_ylabel("Fraction of total errors captured")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(savepath); plt.close(fig)

    with open(out_json, "w") as f:
        json.dump({
            "title": title,
            "savepath": savepath,
            "rows": _saved_rows(rows_sorted),
            "error_rate": error_rate,
            "error_recall_auc_above_diagonal": auc_above,
            "n": n,
            "total_errors": int(total_errors),
        }, f, indent=2)

    return {
        "error_recall_auc_above_diagonal": auc_above,
        "error_rate": error_rate,
        "n": n,
        "total_errors": int(total_errors),
    }


def plot_error_recall_curve_by_len(
    rows: List[Dict],
    title: str,
    out_dir: str,
) -> Dict:
    """Error recall curves grouped by n_primitives; overlay + per-group plots."""
    os.makedirs(out_dir, exist_ok=True)
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r["n_primitives"]].append(r)

    metrics = {}
    overlay_data = {}

    for n_prim in sorted(groups.keys()):
        grp = groups[n_prim]
        grp_sorted = sorted(grp, key=lambda r: r["orthogonality"])
        n = len(grp_sorted)
        total_errors = sum(1 - r["correct"] for r in grp_sorted)
        if total_errors == 0:
            continue

        xs = np.arange(1, n + 1) / n
        ys = np.cumsum([1 - r["correct"] for r in grp_sorted]) / total_errors
        error_rate = total_errors / n
        auc_above = float(np.trapz(ys - xs, xs))

        overlay_data[n_prim] = (xs, ys, auc_above, n, error_rate)
        metrics[n_prim] = {
            "error_recall_auc_above_diagonal": auc_above,
            "error_rate": error_rate,
            "n": n,
        }

        perfect_xs = np.array([0.0, error_rate, 1.0])
        perfect_ys = np.array([0.0, 1.0, 1.0])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5,
                label="Random (x=y)")
        ax.plot(perfect_xs, perfect_ys, linestyle=":", color="green", linewidth=1.5,
                label=f"Perfect (error rate={error_rate:.3f})")
        ax.plot(xs, ys, marker="o", markersize=2,
                label=f"Predictor (AUC={auc_above:.3f})")
        ax.set_xlabel("Fraction of test examples included")
        ax.set_ylabel("Fraction of total errors captured")
        ax.set_title(f"{title} | n_primitives={n_prim} (n={n})")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"error_recall_nprim{n_prim}.svg"))
        plt.close(fig)

    if overlay_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5,
                label="Random (x=y)")
        for n_prim, (xs, ys, auc_above, n, error_rate) in overlay_data.items():
            ax.plot(xs, ys, marker="o", markersize=2,
                    label=f"n_prim={n_prim} (AUC={auc_above:.3f}, err={error_rate:.3f}, n={n})")
        ax.set_xlabel("Fraction of test examples included")
        ax.set_ylabel("Fraction of total errors captured")
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "error_recall_by_len.svg"))
        plt.close(fig)

    return metrics


# ============================================================
# Cross-experiment summary
# ============================================================

def plot_predictive_power_barplot(results_dir: str, results_subdir: str = "analysis_results_atomic_avg_train") -> None:
    """Collect all best_layer_metrics.json files under results_dir and plot
    predictive power as a grouped bar chart: x = size_p, bars = d_model."""
    import glob as _glob

    records = []
    pattern = os.path.join(results_dir, "size_p*", results_subdir,
                           "*", "best_*_layer", "best_layer_metrics.json")
    for path in sorted(_glob.glob(pattern)):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("size_variation_p") is None or data.get("d_model") is None:
            continue
        records.append(data)

    if not records:
        return

    savedir = results_dir
    pp_variants = [
        ("predictive_power",            "Percentile AUC"),
        ("predictive_power_fixedrange", "Fixedrange AUC"),
        ("predictive_power_pct_scaled", "Pct-scaled AUC"),
    ]

    aggregations = sorted({r["aggregation"] for r in records if r.get("aggregation")})
    for agg in aggregations:
        agg_recs = [r for r in records if r.get("aggregation") == agg]
        size_ps  = sorted({r["size_variation_p"] for r in agg_recs})
        d_models = sorted({r["d_model"]          for r in agg_recs})
        n_groups, n_bars = len(size_ps), len(d_models)
        width = 0.8 / max(n_bars, 1)

        for pp_key, pp_label in pp_variants:
            fig, ax = plt.subplots(figsize=(max(8, n_groups * n_bars * 0.5 + 2), 5))
            for i, dm in enumerate(d_models):
                vals = []
                for sp in size_ps:
                    matches = [r[pp_key] for r in agg_recs
                               if r["size_variation_p"] == sp and r["d_model"] == dm
                               and r.get(pp_key) is not None]
                    vals.append(float(np.mean(matches)) if matches else float("nan"))
                offsets = np.arange(n_groups) + (i - (n_bars - 1) / 2) * width
                ax.bar(offsets, vals, width=width * 0.9, label=f"d_model={dm}")

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="baseline (PP=1)")
            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels([f"p={sp}%" for sp in size_ps])
            ax.set_xlabel("Training size (% of full dataset)")
            ax.set_ylabel("Predictive power (AUC / baseline AUC)")
            ax.set_title(f"Predictive power — {agg} aggregation ({pp_label})")
            ax.legend(title="d_model", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fname = f"predictive_power_{pp_key}_{agg}.svg"
            fig.savefig(os.path.join(savedir, fname))
            plt.close(fig)
            print(f"Saved: {os.path.join(savedir, fname)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results_scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_commands_per_depth", type=int, default=1000)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--exposure_ratio", type=float, default=1.0)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--aggregation", type=str, default="min", choices=["mean", "sum", "min", "max", "cumulative_coherence", "cumulative_coherence_l2"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--bin_width", type=float, default=0.1)
    parser.add_argument("--data_mode", type=str, default="size_variation", choices=["exposure", "size_variation"])
    parser.add_argument("--size_variation_p", type=int, default=None)
    parser.add_argument("--dev_source", type=str, default="from_train_split",
                        choices=["from_train_split", "from_test_split"])
    parser.add_argument("--split_by_len", action="store_true", default=False,
                        help="Also produce cumulative and noncumulative plots grouped by n_primitives")
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="analysis_results_atomic_avg_train",
        help="Name of the subdirectory inside size_p*/ where analysis results are saved",
    )
    parser.add_argument(
        "--extract_atomic_repr",
        type=str,
        default="avg_train",
        choices=["avg_train", "single_tok_inference"],
        help=(
            "'avg_train': average hidden state over all training occurrences (default). "
            "'single_tok_inference': one forward pass of <sep> concept <sep> per token."
        ),
    )
    parser.add_argument(
        "--select_best_layer",
        type=str,
        default="cumulative",
        choices=["cumulative", "noncumulative", "cumulative_auc_and_direction"],
        help=(
            "Criterion for selecting the best layer using dev set. "
            "'cumulative': lowest dev cumulative AUC. "
            "'noncumulative': highest dev noncumulative R². "
            "'cumulative_auc_and_direction': argmax of normalized(-dev_auc) + normalized(dev_slope), "
            "where dev_slope is the linear slope of the dev cumulative curve."
        ),
    )
    parser.add_argument(
        "--no_duplicate_in_example_ortho",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help=(
            "If True (default), deduplicate atomic tokens per example before computing pairwise "
            "orthogonality.  Prevents the same averaged rep being paired with itself (which always "
            "gives ortho=0) when a token appears multiple times in one input."
        ),
    )
    parser.add_argument(
        "--do_just_aucs",
        action="store_true",
        default=False,
        help=(
            "If set, skip all heavy visualisation (heatmaps, co-occurrence, PCA, etc.) and only "
            "produce cumulative/PR-AUC plots and their JSON/CSV metrics."
        ),
    )
    parser.add_argument(
        "--do_permutation_test",
        action="store_true",
        default=False,
        help=(
            "If set, run permutation tests for cumulative AUC and PR-AUC (1000 iterations each), "
            "AND bootstrap a 95% CI for the improvement in PR-AUC over the failure-rate baseline "
            "(1000 resamples; see compute_failure_pr_auc_from_rows). Disabled by default to speed "
            "up runs; enable when p-values / CIs are needed."
        ),
    )
    parser.add_argument(
        "--mean_center",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=False,
        metavar="BOOL",
        help=(
            "If True, apply mean-centering to atomic representations before computing orthogonality. "
            "Action tokens (PRIMITIVES excl. 'turn') are centered by the mean action rep; "
            "non-action tokens (directions, manner, repeaters, connectors) by the mean non-action rep."
        ),
    )
    args = parser.parse_args()
    DO_JUST_AUCS = args.do_just_aucs
    DO_PERMUTATION_TEST = args.do_permutation_test

    set_seed(args.seed)
    print("starting analysis with config:", vars(args))

    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    max_commands_per_depth = None if args.max_commands_per_depth < 0 else args.max_commands_per_depth

    if args.data_mode == "size_variation":
        variant_dir = os.path.join(args.results_dir, f"size_p{args.size_variation_p}")
        variant_label = f"p={args.size_variation_p}%"
    else:
        variant_dir = os.path.join(args.results_dir, exposure_tag(args.exposure_ratio))
        variant_label = f"exposure{str(args.exposure_ratio).replace('.', 'p')}"

    if args.checkpoint is None:
        checkpoint_path = infer_latest_checkpoint(
            variant_dir, args.seed, args.d_model, args.n_heads, args.d_ff, args.n_layers
        )
    else:
        checkpoint_path = args.checkpoint
    print("using checkpoint:", checkpoint_path)

    print("\n[1/6] Reconstructing data splits...")
    full_train_data, train_data, dev_data, test_data = reconstruct_train_dev_test(
        seed=args.seed,
        max_depth=args.max_depth,
        max_commands_per_depth=max_commands_per_depth,
        train_fraction=args.train_fraction,
        exposure_ratio=args.exposure_ratio,
        dev_fraction=args.dev_fraction,
        data_mode=args.data_mode,
        size_variation_p=args.size_variation_p,
        dev_source=args.dev_source,
    )
    print(f"    train={len(train_data)}  dev={len(dev_data)}  test={len(test_data)}")

    vocab, token2id = make_shared_vocab(full_train_data, test_data)
    pad_id = token2id["<pad>"]
    print(f"    shared vocab: {len(vocab)} tokens")

    print("\n[2/6] Loading model...")
    model = load_model(
        checkpoint_path=checkpoint_path,
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        pad_id=pad_id,
        max_len=128,
    )
    print(f"    loaded from {checkpoint_path}")

    print("\n[3/6] Extracting atomic representations...")
    reps_cache = _atomic_reps_cache_path(checkpoint_path, args.extract_atomic_repr)
    if os.path.exists(reps_cache):
        atomic_reps_by_layer = load_atomic_reps(reps_cache)
        counts_by_layer = {}  # not needed downstream; omit from cache
        print(f"    loaded from cache: {reps_cache}")
    else:
        if args.extract_atomic_repr == "single_tok_inference":
            atomic_reps_by_layer, counts_by_layer = extract_atomic_representations_single_tok(
                model=model,
                token2id=token2id,
                n_layers=args.n_layers,
            )
        else:
            atomic_reps_by_layer, counts_by_layer = extract_atomic_representations(
                model=model,
                train_data=train_data,
                token2id=token2id,
                n_layers=args.n_layers,
            )
        save_atomic_reps(atomic_reps_by_layer, reps_cache)
    print(f"    layers: {sorted(atomic_reps_by_layer.keys())}")
    print(f"    tokens: {sorted(next(iter(atomic_reps_by_layer.values())).keys())}")

    if args.mean_center:
        print("    Applying mean-centering (action vs non-action) to atomic representations...")
        atomic_reps_by_layer = mean_center_atomic_reps(atomic_reps_by_layer)
        print(f"    Action tokens:     {ACTION_TOKENS}")
        print(f"    Non-action tokens: {NON_ACTION_TOKENS}")

    if args.layer is not None:
        atomic_reps_by_layer = {args.layer: atomic_reps_by_layer[args.layer]}

    max_new_tokens = max(len(out) for _, out in (train_data + dev_data + test_data)) + 1

    print(f"\n[4/6] Evaluating test set ({len(test_data)} examples)...")
    results_by_layer = analyze_examples(
        model=model,
        examples=list(test_data),
        token2id=token2id,
        atomic_reps_by_layer=atomic_reps_by_layer,
        max_new_tokens=max_new_tokens,
        aggregation=args.aggregation,
        no_duplicate_in_example_ortho=args.no_duplicate_in_example_ortho,
        checkpoint_path=checkpoint_path,
        split_name="test",
    )

    print(f"\n[5/6] Evaluating dev set ({len(dev_data)} examples)...")
    dev_results_by_layer = analyze_examples(
        model=model,
        examples=list(dev_data),
        token2id=token2id,
        atomic_reps_by_layer=atomic_reps_by_layer,
        max_new_tokens=max_new_tokens,
        aggregation=args.aggregation,
        no_duplicate_in_example_ortho=args.no_duplicate_in_example_ortho,
        checkpoint_path=checkpoint_path,
        split_name="dev",
    )
    print()

    dedup_tag = "_dedup" if args.no_duplicate_in_example_ortho else "_withdup"
    meancenter_tag = "_meancenter" if args.mean_center else ""
    run_name = (
        f"seed{args.seed}_dmodel{args.d_model}_nheads{args.n_heads}_dff{args.d_ff}_layer{args.n_layers}_"
        f"{variant_label}_"
        f"{args.aggregation}_"
        f"sel{args.select_best_layer}"
        f"{dedup_tag}"
        f"{meancenter_tag}"
    )
    analysis_dir = os.path.join(variant_dir, args.results_subdir, run_name)
    os.makedirs(analysis_dir, exist_ok=True)

    # Save which atomic tokens were successfully extracted per layer
    with open(os.path.join(analysis_dir, "atomic_tokens.json"), "w") as f:
        json.dump(
            {
                str(layer): sorted(reps.keys())
                for layer, reps in atomic_reps_by_layer.items()
            },
            f,
            indent=2,
        )

    with open(os.path.join(analysis_dir, "config.json"), "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "seed": args.seed,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_ff": args.d_ff,
                "max_depth": args.max_depth,
                "max_commands_per_depth": max_commands_per_depth,
                "train_fraction": args.train_fraction,
                "exposure_ratio": args.exposure_ratio,
                "dev_fraction": args.dev_fraction,
                "aggregation": args.aggregation,
                "layer": args.layer,
                "bin_width": args.bin_width,
                "data_mode": args.data_mode,
                "size_variation_p": args.size_variation_p,
                "dev_source": args.dev_source,
                "split_by_len": args.split_by_len,
                "no_duplicate_in_example_ortho": args.no_duplicate_in_example_ortho,
                "extraction_method": "isolated_sep_concept_sep",
                "mean_center": args.mean_center,
            },
            f,
            indent=2,
        )

    summary = {}
    dev_auc_by_layer = {}
    dev_r2_by_layer = {}
    dev_slope_by_layer = {}
    n_perm_arg = 1000 if DO_PERMUTATION_TEST else 0
    print(f"[6/6] Per-layer dev-set scoring ({len(results_by_layer)} layers) "
          f"to select the best layer, then full analysis on that layer only "
          f"(perm_test/bootstrap_ci={'on' if DO_PERMUTATION_TEST else 'off'}, "
          f"just_aucs={DO_JUST_AUCS})...")
    for layer, rows in tqdm(results_by_layer.items(), desc="per-layer dev scoring"):
        dev_rows = dev_results_by_layer[layer]
        dev_rows_sorted = sorted(dev_rows, key=lambda r: r["orthogonality"])
        dev_auc = cumulative_auc(dev_rows_sorted)
        dev_auc_by_layer[layer] = dev_auc
        dev_xs, dev_ys, _ = noncumulative_fixedwidth_accuracy(dev_rows_sorted, bin_width=args.bin_width)
        dev_r2, _ = pearson_r_p(dev_xs, dev_ys) if len(dev_xs) >= 2 else (float("nan"), float("nan"))
        dev_r2_by_layer[layer] = dev_r2
        dev_cum_xs, dev_cum_ys = cumulative_mean_accuracy(dev_rows_sorted, bin_width=args.bin_width)
        dev_slope = float(np.polyfit(dev_cum_xs, dev_cum_ys, 1)[0]) if len(dev_cum_xs) >= 2 else float("nan")
        dev_slope_by_layer[layer] = dev_slope

        summary[layer] = {
            "n_examples": len(rows),
            "mean_accuracy": float(np.mean([r["correct"] for r in rows])) if rows else float("nan"),
            "mean_orthogonality": float(np.mean([r["orthogonality"] for r in rows])) if rows else float("nan"),
            "dev_cumulative_auc": float(dev_auc),
            "dev_cumulative_slope": float(dev_slope) if not np.isnan(dev_slope) else None,
            "dev_noncumulative_r2": float(dev_r2) if not np.isnan(dev_r2) else None,
        }

    layers_ordered = sorted(dev_auc_by_layer.keys())

    if args.select_best_layer == "cumulative":
        best_layer = min(dev_auc_by_layer, key=lambda l: dev_auc_by_layer[l])
        print(
            f"\nBest layer by lowest dev cumulative AUC: layer {best_layer} "
            f"(dev AUC={dev_auc_by_layer[best_layer]:.4f})"
        )
    elif args.select_best_layer == "noncumulative":
        best_layer = max(
            dev_r2_by_layer,
            key=lambda l: dev_r2_by_layer[l] if not np.isnan(dev_r2_by_layer[l]) else -np.inf,
        )
        print(
            f"\nBest layer by highest dev noncumulative R²: layer {best_layer} "
            f"(dev R²={dev_r2_by_layer[best_layer]:.4f})"
        )
    else:  # cumulative_auc_and_direction
        aucs = np.array([dev_auc_by_layer[l] for l in layers_ordered], dtype=float)
        slopes = np.array(
            [dev_slope_by_layer[l] if not np.isnan(dev_slope_by_layer[l]) else 0.0 for l in layers_ordered],
            dtype=float,
        )

        def _normalize(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 1e-12 else np.zeros_like(arr)

        combined = _normalize(-aucs) + _normalize(slopes)
        best_layer = layers_ordered[int(np.argmax(combined))]
        print(
            f"\nBest layer by normalized(-dev_auc) + normalized(dev_slope): layer {best_layer} "
            f"(dev AUC={dev_auc_by_layer[best_layer]:.4f}, dev slope={dev_slope_by_layer[best_layer]:.4f}, "
            f"combined={combined[layers_ordered.index(best_layer)]:.4f})"
        )

    summary["best_layer"] = best_layer
    summary["select_best_layer"] = args.select_best_layer

    best_rows = results_by_layer[best_layer]
    best_layer_dir = os.path.join(analysis_dir, f"best_{args.select_best_layer}_layer")
    os.makedirs(best_layer_dir, exist_ok=True)
    print(f"\n[best-layer] Analyzing best layer {best_layer}...")

    # Kept at the legacy layer{N}/test_scores.csv path (not best_layer_dir) since
    # compare_metrics_by_nprim.py's cumulative_coherence_test_scores_path reads from there.
    best_layer_csv_dir = os.path.join(analysis_dir, f"layer{best_layer}")
    os.makedirs(best_layer_csv_dir, exist_ok=True)
    save_rows_csv(best_rows, os.path.join(best_layer_csv_dir, "test_scores.csv"))

    best_cum_metrics = plot_cumulative_curve(
        rows=best_rows,
        title=f"Best layer (layer {best_layer}): cumulative test eval accuracy vs orthogonality",
        savepath=os.path.join(best_layer_dir, "cumulative.svg"),
        bin_width=args.bin_width,
        n_permutations=n_perm_arg,
    )

    best_pr_metrics = plot_failure_pr_curve_from_rows(
        rows=best_rows,
        title=f"Best layer (layer {best_layer}): failure PR-AUC vs orthogonality",
        out_dir=best_layer_dir,
        n_perm=n_perm_arg,
        n_boot=n_perm_arg,
    )
    with open(os.path.join(best_layer_dir, "failure_pr_auc.json"), "w") as _f:
        json.dump(best_pr_metrics, _f, indent=2)

    best_noncum_metrics = plot_noncumulative_curve(
        rows=best_rows,
        title=f"Best layer (layer {best_layer}): noncumulative test eval accuracy vs orthogonality",
        savepath=os.path.join(best_layer_dir, "noncumulative.svg"),
        bin_width=args.bin_width,
    )

    plot_noncumulative_pct_at_ortho(
        rows=best_rows,
        title=f"Best layer (layer {best_layer}): noncumulative test eval accuracy vs orthogonality (pct-at-ortho)",
        savepath=os.path.join(best_layer_dir, "noncumulative.svg"),
        bin_width=args.bin_width,
    )

    best_error_recall_metrics = plot_error_recall_curve(
        rows=best_rows,
        title=f"Best layer (layer {best_layer}): error recall curve",
        savepath=os.path.join(best_layer_dir, "error_recall.svg"),
    )

    if args.split_by_len:
        best_by_len_dir = os.path.join(best_layer_dir, "by_len")
        plot_cumulative_curve_by_len(
            rows=best_rows,
            title=f"Best layer (layer {best_layer}): cumulative test eval accuracy by n_primitives",
            out_dir=best_by_len_dir,
            bin_width=args.bin_width,
        )
        plot_noncumulative_curve_by_len(
            rows=best_rows,
            title=f"Best layer (layer {best_layer}): noncumulative test eval accuracy by n_primitives",
            out_dir=best_by_len_dir,
            bin_width=args.bin_width,
        )
        best_pr_by_len_metrics = plot_failure_pr_curve_by_len(
            rows=best_rows,
            title=f"Best layer (layer {best_layer}): failure PR-AUC by n_primitives",
            out_dir=best_by_len_dir,
            n_perm=n_perm_arg,
            n_boot=n_perm_arg,
        )
        with open(os.path.join(best_by_len_dir, "failure_pr_auc_by_len.json"), "w") as _f:
            json.dump({str(k): v for k, v in best_pr_by_len_metrics.items()}, _f, indent=2)
        plot_error_recall_curve_by_len(
            rows=best_rows,
            title=f"Best layer (layer {best_layer}): error recall by n_primitives",
            out_dir=best_by_len_dir,
        )
    if not DO_JUST_AUCS:
        plot_orthogonality_heatmap(
            atomic_reps=atomic_reps_by_layer[best_layer],
            title=f"Best layer (layer {best_layer}): pairwise orthogonality of atomic concepts",
            savepath=os.path.join(best_layer_dir, "orthogonality_heatmap.svg"),
        )

        plot_cooccurrence_heatmap(
            train_data=train_data,
            title="Training corpus: pairwise co-occurrence frequency of atomic concepts",
            savepath=os.path.join(best_layer_dir, "cooccurrence_heatmap.svg"),
        )

        plot_orthogonality_vs_cooccurrence(
            atomic_reps=atomic_reps_by_layer[best_layer],
            train_data=train_data,
            title=f"Best layer (layer {best_layer}): orthogonality vs co-occurrence",
            savepath=os.path.join(best_layer_dir, "orthogonality_vs_cooccurrence.svg"),
        )

        plot_neighbor_cooccurrence_heatmap(
            train_data=train_data,
            title="Training corpus: neighbor co-occurrence of atomic concepts",
            savepath=os.path.join(best_layer_dir, "neighbor_cooccurrence_heatmap.svg"),
        )

        plot_orthogonality_vs_neighbor_cooccurrence(
            atomic_reps=atomic_reps_by_layer[best_layer],
            train_data=train_data,
            title=f"Best layer (layer {best_layer}): orthogonality vs neighbor co-occurrence",
            savepath=os.path.join(best_layer_dir, "orthogonality_vs_neighbor_cooccurrence.svg"),
        )

        plot_concept_occurrence_counts(
            train_data=train_data,
            title="Training corpus: total occurrence count per atomic concept",
            savepath=os.path.join(best_layer_dir, "concept_occurrence_counts.svg"),
        )

        plot_occurrence_vs_mean_orthogonality(
            atomic_reps=atomic_reps_by_layer[best_layer],
            train_data=train_data,
            title=f"Best layer (layer {best_layer}): occurrence count vs sum orthogonality to others",
            savepath=os.path.join(best_layer_dir, "occurrence_vs_orthogonality.svg"),
        )

        plot_pca_2d_svg(
            atomic_reps=atomic_reps_by_layer[best_layer],
            title=f"Best layer (layer {best_layer}): 2D PCA of atomic concept embeddings",
            savepath=os.path.join(best_layer_dir, "pca_2d_concepts.svg"),
        )
       

        save_plot_data_json(
            rows=best_rows,
            atomic_reps=atomic_reps_by_layer[best_layer],
            cumulative_title=f"Best layer (layer {best_layer}): cumulative test eval accuracy vs orthogonality",
            cumulative_savepath=os.path.join(best_layer_dir, "cumulative.svg"),
            heatmap_title=f"Best layer (layer {best_layer}): pairwise orthogonality of atomic concepts",
            heatmap_savepath=os.path.join(best_layer_dir, "orthogonality_heatmap.svg"),
            pca_title=f"Best layer (layer {best_layer}): 2D PCA of atomic concept embeddings",
            pca_savepath=os.path.join(best_layer_dir, "pca_2d_concepts.svg"),
            bin_width=args.bin_width,
            out_json=os.path.join(best_layer_dir, "plot_data.json"),
            cum_metrics=best_cum_metrics,
            noncumulative_title=f"Best layer (layer {best_layer}): noncumulative test eval accuracy vs orthogonality",
            noncumulative_savepath=os.path.join(best_layer_dir, "noncumulative.svg"),
            by_len_title=f"Best layer (layer {best_layer}): accuracy by n_primitives" if args.split_by_len else None,
            by_len_out_dir=os.path.join(best_layer_dir, "by_len") if args.split_by_len else None,
        )
    else:
        save_plot_data_json(
            rows=best_rows,
            atomic_reps=atomic_reps_by_layer[best_layer],
            cumulative_title=f"Best layer (layer {best_layer}): cumulative test eval accuracy vs orthogonality",
            cumulative_savepath=os.path.join(best_layer_dir, "cumulative.svg"),
            heatmap_title=f"Best layer (layer {best_layer}): pairwise orthogonality of atomic concepts",
            heatmap_savepath=os.path.join(best_layer_dir, "orthogonality_heatmap.svg"),
            pca_title=f"Best layer (layer {best_layer}): 2D PCA of atomic concept embeddings",
            pca_savepath=os.path.join(best_layer_dir, "pca_2d_concepts.svg"),
            bin_width=args.bin_width,
            out_json=os.path.join(best_layer_dir, "plot_data.json"),
            cum_metrics=best_cum_metrics,
            noncumulative_title=f"Best layer (layer {best_layer}): noncumulative test eval accuracy vs orthogonality",
            noncumulative_savepath=os.path.join(best_layer_dir, "noncumulative.svg"),
            by_len_title=f"Best layer (layer {best_layer}): accuracy by n_primitives" if args.split_by_len else None,
            by_len_out_dir=os.path.join(best_layer_dir, "by_len") if args.split_by_len else None,
        )

    best_layer_metrics = {
        "best_layer": best_layer,
        "aggregation": args.aggregation,
        "size_variation_p": args.size_variation_p,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "seed": args.seed,
        "avg_acc": best_cum_metrics["avg_acc"],
        "cumulative_auc": best_cum_metrics["auc"],
        "cumulative_auc_fixedrange": best_cum_metrics["auc_fixedrange"],
        "cumulative_auc_pct_scaled": best_cum_metrics["auc_pct_scaled"],
        "predictive_power": best_cum_metrics["predictive_power"],
        "predictive_power_fixedrange": best_cum_metrics["predictive_power_fixedrange"],
        "predictive_power_pct_scaled": best_cum_metrics["predictive_power_pct_scaled"],
        "noncumulative_r": best_noncum_metrics["r"],
        "noncumulative_p": best_noncum_metrics["p"],
        "pbc": best_cum_metrics["pbc"],
        "pbc_p": best_cum_metrics["pbc_p"],
        "pr_auc_failure": best_pr_metrics["pr_auc"],
        "normalized_pr_auc_failure": best_pr_metrics["normalized_pr_auc"],
        "failure_rate": best_pr_metrics["failure_rate"],
        "perm_p_pr_auc": best_pr_metrics["perm_p"],
        "pr_auc_improvement_ci_lo": best_pr_metrics["pr_auc_improvement_ci_lo"],
        "pr_auc_improvement_ci_hi": best_pr_metrics["pr_auc_improvement_ci_hi"],
        "normalized_pr_auc_ci_lo": best_pr_metrics["normalized_pr_auc_ci_lo"],
        "normalized_pr_auc_ci_hi": best_pr_metrics["normalized_pr_auc_ci_hi"],
        "error_recall_auc_above_diagonal": best_error_recall_metrics["error_recall_auc_above_diagonal"],
        "mean_center": args.mean_center,
    }
    with open(os.path.join(best_layer_dir, "best_layer_metrics.json"), "w") as f:
        json.dump(best_layer_metrics, f, indent=2)

    with open(os.path.join(analysis_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── auto-copy best layer plots into plots_for_paper ──────────────────────
    import shutil as _shutil
    plots_for_paper = os.path.join(args.results_dir, "plots_for_paper")
    if args.data_mode == "size_variation" and args.size_variation_p is not None:
        dest_dir = os.path.join(plots_for_paper, args.aggregation, f"exp{args.size_variation_p}")
    else:
        exp_tag = str(args.exposure_ratio).replace(".", "p")
        dest_dir = os.path.join(plots_for_paper, args.aggregation, f"exp{exp_tag}")
    os.makedirs(dest_dir, exist_ok=True)

    prefix = f"dmodel{args.d_model}__"
    for fname in os.listdir(best_layer_dir):
        src = os.path.join(best_layer_dir, fname)
        if os.path.isfile(src):
            _shutil.copy2(src, os.path.join(dest_dir, prefix + fname))

    print(f"\nSaved analysis outputs to: {analysis_dir}")
    print(f"Best layer plots saved to:  {best_layer_dir}")
    print(f"Copied to plots_for_paper:  {dest_dir}")

    # ── update cross-experiment predictive power bar plot ─────────────────────
    if args.data_mode == "size_variation":
        plot_predictive_power_barplot(args.results_dir, args.results_subdir)


if __name__ == "__main__":
    main()


