# Adapted from ryeii/Representational-Homomorphism-for-Transformer-Language-Models
# (github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models),
# he_probe/experiment_scan.py.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from .gen_data_scan import (
        random_scan_split,
        subsample_train_by_exposure,
        load_size_variation_split,
        build_vocab_from_datasets,
        summarize_dataset,
    )
    from .transformers import DecoderOnlyTransformer
except ImportError:
    from gen_data_scan import (
        random_scan_split,
        subsample_train_by_exposure,
        load_size_variation_split,
        build_vocab_from_datasets,
        summarize_dataset,
    )
    from transformers import DecoderOnlyTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Dataset wrapper for decoder-only LM
# ============================================================

class SeqDataset(Dataset):
    """
    Format:
      [BOS] + input + [SEP] + output + [EOS]

    Loss is applied only to output-side targets.
    """
    def __init__(
        self,
        data,
        vocab,
        bos_token="<bos>",
        sep_token="<sep>",
        eos_token="<eos>",
        pad_token="<pad>",
    ):
        self.data = list(data)
        self.vocab = list(vocab)
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        self.bos_token = bos_token
        self.sep_token = sep_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.bos_id = self.token2id[bos_token]
        self.sep_id = self.token2id[sep_token]
        self.eos_id = self.token2id[eos_token]
        self.pad_id = self.token2id[pad_token]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]

        full_seq = [self.bos_token] + list(inp) + [self.sep_token] + list(out) + [self.eos_token]
        full_ids = torch.tensor([self.token2id[tok] for tok in full_seq], dtype=torch.long)

        prefix_seq = [self.bos_token] + list(inp) + [self.sep_token]
        prefix_ids = torch.tensor([self.token2id[tok] for tok in prefix_seq], dtype=torch.long)

        target_out_ids = torch.tensor([self.token2id[tok] for tok in list(out)], dtype=torch.long)

        output_start_idx = 1 + len(inp) + 1  # BOS + input + SEP

        return {
            "full_ids": full_ids,
            "prefix_ids": prefix_ids,
            "target_out_ids": target_out_ids,
            "output_start_idx": output_start_idx,
        }


def collate_batch(batch, pad_id: int):
    full_ids = nn.utils.rnn.pad_sequence(
        [b["full_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    prefix_ids = nn.utils.rnn.pad_sequence(
        [b["prefix_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    target_out_ids = nn.utils.rnn.pad_sequence(
        [b["target_out_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    output_start_idxs = [b["output_start_idx"] for b in batch]

    return {
        "full_ids": full_ids,
        "prefix_ids": prefix_ids,
        "target_out_ids": target_out_ids,
        "output_start_idx": output_start_idxs,
    }


# ============================================================
# Training / evaluation
# ============================================================

def train_one_epoch(model, train_loader, optimizer, pad_id, epoch=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch["full_ids"].to(device)
        output_start_idxs = batch["output_start_idx"]
        padding_mask = (x != pad_id)

        optimizer.zero_grad()
        logits, _ = model(x, padding_mask=padding_mask)

        logits = logits[:, :-1, :]
        targets = x[:, 1:]

        loss_per_token = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(targets.shape)

        B, Tm1 = targets.shape
        output_mask = torch.zeros((B, Tm1), dtype=torch.float32, device=device)

        for i, s in enumerate(output_start_idxs):
            start_in_targets = s - 1
            if start_in_targets < Tm1:
                output_mask[i, start_in_targets:] = 1.0

        non_pad = (targets != pad_id).float()
        output_mask = output_mask * non_pad

        per_example_loss = (loss_per_token * output_mask).sum(dim=1) / output_mask.sum(dim=1).clamp(min=1.0)
        loss = per_example_loss.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def trim_at_eos_or_pad(seq, eos_id, pad_id):
    out = []
    for tok in seq:
        if tok == eos_id or tok == pad_id:
            break
        out.append(tok)
    return out


@torch.no_grad()
def _evaluate_single(model, loader, eos_id, pad_id, max_new_tokens):
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        prefix_ids = batch["prefix_ids"]
        target_out_ids = batch["target_out_ids"].cpu().tolist()
        for i in range(prefix_ids.size(0)):
            plen = int((prefix_ids[i] != pad_id).sum().item())
            prefix = prefix_ids[i, :plen].unsqueeze(0).to(device)  # (1, plen)
            pred_full = model.generate(
                prefix_ids=prefix,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
            ).cpu().tolist()[0]
            pred_trim = trim_at_eos_or_pad(pred_full[plen:], eos_id, pad_id)
            gold_trim = trim_at_eos_or_pad(target_out_ids[i], eos_id, pad_id)
            total += 1
            correct += int(pred_trim == gold_trim)
    return {"accuracy": correct / total if total > 0 else 0.0}


@torch.no_grad()
def _evaluate_batch(model, loader, eos_id, pad_id, max_new_tokens):
    """Groups samples by prefix length so batches need no padding."""
    model.eval()
    groups: dict = {}
    for batch in loader:
        prefix_ids = batch["prefix_ids"]
        target_out_ids = batch["target_out_ids"].cpu().tolist()
        for i in range(prefix_ids.size(0)):
            plen = int((prefix_ids[i] != pad_id).sum().item())
            prefix = prefix_ids[i, :plen]
            groups.setdefault(plen, []).append((prefix, target_out_ids[i]))

    total = 0
    correct = 0
    for plen, samples in groups.items():
        prefixes = torch.stack([s[0] for s in samples]).to(device)
        golds = [s[1] for s in samples]
        generated = model.generate(
            prefix_ids=prefixes,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
        ).cpu().tolist()
        for pred_full, gold in zip(generated, golds):
            pred_trim = trim_at_eos_or_pad(pred_full[plen:], eos_id, pad_id)
            gold_trim = trim_at_eos_or_pad(gold, eos_id, pad_id)
            total += 1
            correct += int(pred_trim == gold_trim)
    return {"accuracy": correct / total if total > 0 else 0.0}


def evaluate_model(model, loader, eos_id, pad_id, max_new_tokens=64, eval_batch=False):
    if eval_batch:
        return _evaluate_batch(model, loader, eos_id, pad_id, max_new_tokens)
    return _evaluate_single(model, loader, eos_id, pad_id, max_new_tokens)


# ============================================================
# Experiment records
# ============================================================

@dataclass
class EpochRecord:
    seed_split_data: int
    seed_for_training: int
    d_model: int
    n_layers: int
    exposure_ratio: float
    epoch: int
    train_loss: float
    dev_acc: float
    test_acc: float


# ============================================================
# Main experiment
# ============================================================

def make_shared_vocab(train_data, test_data):
    base_vocab = build_vocab_from_datasets(train_data, test_data, add_pad=False)
    vocab = ["<pad>", "<bos>", "<sep>", "<eos>"] + [
        tok for tok in base_vocab if tok not in {"<pad>", "<bos>", "<sep>", "<eos>"}
    ]
    return vocab


def make_loader(dataset, pad_id, batch_size=32, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_batch(b, pad_id),
    )


def build_data_split(
    seed_split_data,
    max_depth=3,
    max_commands_per_depth=None,
    train_fraction=0.8,
    exposure_ratio=1.0,
    dev_fraction=0.1,
    dev_source="from_train_split",
    data_mode="exposure",
    size_variation_p=None,
):
    """
    All randomness here comes from `seed_split_data` via private random.Random
    instances (never the global random/np.random/torch RNG), so this is fully
    independent of seed_for_training / model init / anything training-related.
    """
    if data_mode == "size_variation":
        full_train_data, test_data = load_size_variation_split(size_variation_p)
        exposed_train_data = list(full_train_data)
        exposure_ratio = size_variation_p / 100.0
    else:
        full_train_data, test_data = random_scan_split(
            seed=seed_split_data,
            max_depth=max_depth,
            max_commands_per_depth=max_commands_per_depth,
            train_fraction=train_fraction,
        )
        exposed_train_data = subsample_train_by_exposure(
            full_train_data,
            exposure_ratio=exposure_ratio,
            seed=seed_split_data,
        )

    rng = random.Random(seed_split_data)
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

    return full_train_data, exposed_train_data, train_data, dev_data, test_data


def data_split_fingerprint(train_data, dev_data, test_data) -> str:
    """Short hash identifying the exact train/dev/test contents, for checking
    that two runs sharing the same seed_split_data produced the same split."""
    h = hashlib.sha256()
    for split in (train_data, dev_data, test_data):
        h.update(f"|n={len(split)}|".encode())
        for inp, out in split:
            h.update(("/".join(inp) + ">" + "/".join(out) + ";").encode())
    return h.hexdigest()[:12]


def check_seed_split_data_determinism(seed_split_data=42, n_trials=3, **data_kwargs) -> bool:
    """
    Small sanity check: rebuild the data split `n_trials` times with the same
    seed_split_data (and the rest of the data-related kwargs held fixed) and
    assert every trial produces an identical fingerprint. Since seed_for_training
    is never passed into build_data_split, this also demonstrates the split is
    independent of it.
    """
    fingerprints = []
    for _ in range(n_trials):
        _, _, train_data, dev_data, test_data = build_data_split(seed_split_data, **data_kwargs)
        fingerprints.append(data_split_fingerprint(train_data, dev_data, test_data))

    ok = len(set(fingerprints)) == 1
    status = "PASS" if ok else "FAIL"
    print(f"[check_seed_split_data_determinism] seed_split_data={seed_split_data} "
          f"n_trials={n_trials} fingerprints={fingerprints} -> {status}")
    if not ok:
        raise AssertionError(
            f"Data split is NOT deterministic for seed_split_data={seed_split_data}: {fingerprints}"
        )
    return ok


def run_single_config(
    seed_split_data=42,
    seed_for_training=42,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=None,
    epochs=100,
    lr=1e-3,
    batch_size=32,
    max_depth=3,
    max_commands_per_depth=None,
    train_fraction=0.8,
    exposure_ratio=1.0,
    dev_fraction=0.1,
    dev_source="from_train_split",
    save_weights=False,
    saved_folder="results_scan_random",
    patience=None,
    data_mode="exposure",
    size_variation_p=None,
    eval_batch=False,
):
    # seed_split_data controls only the data split/exposure subsampling/shuffle
    # below (all via private random.Random instances). seed_for_training is
    # applied last so it alone controls model init, dropout, and DataLoader
    # shuffling during training.
    os.makedirs(saved_folder, exist_ok=True)
    log_path = os.path.join(
        saved_folder,
        f"seedsplit{seed_split_data}_seedtrain{seed_for_training}_dmodel{d_model}_nheads{n_heads}_dff{d_ff or 4 * d_model}_layer{n_layers}.txt",
    )
    log_file = open(log_path, "w", buffering=1)

    if d_ff is None:
        d_ff = 4 * d_model

    full_train_data, exposed_train_data, train_data, dev_data, test_data = build_data_split(
        seed_split_data,
        max_depth=max_depth,
        max_commands_per_depth=max_commands_per_depth,
        train_fraction=train_fraction,
        exposure_ratio=exposure_ratio,
        dev_fraction=dev_fraction,
        dev_source=dev_source,
        data_mode=data_mode,
        size_variation_p=size_variation_p,
    )
    if data_mode == "size_variation":
        exposure_ratio = size_variation_p / 100.0

    split_fp = data_split_fingerprint(train_data, dev_data, test_data)
    print("FULL TRAIN:", summarize_dataset(full_train_data))
    print("EXPOSED TRAIN:", summarize_dataset(exposed_train_data))
    print("DEV:", summarize_dataset(dev_data))
    print("TEST:", summarize_dataset(test_data))
    print(f"[data-split check] seed_split_data={seed_split_data} fingerprint={split_fp}")
    log_file.write(f"data_split_fingerprint seed_split_data={seed_split_data} fingerprint={split_fp}\n")

    # Data split/exposure/shuffle above is fully determined by seed_split_data
    # and does not touch global RNG state. Everything from here on (model
    # init, dropout, DataLoader shuffling) is controlled by seed_for_training.
    set_seed(seed_for_training)

    vocab = make_shared_vocab(full_train_data, test_data)

    train_ds = SeqDataset(train_data, vocab)
    dev_ds = SeqDataset(dev_data, vocab)
    test_ds = SeqDataset(test_data, vocab)

    pad_id = train_ds.pad_id
    eos_id = train_ds.eos_id

    train_loader = make_loader(train_ds, pad_id, batch_size=batch_size, shuffle=True)
    dev_loader = make_loader(dev_ds, pad_id, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_ds, pad_id, batch_size=batch_size, shuffle=False)

    model = DecoderOnlyTransformer(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=128,
        dropout=0.1,
        pad_id=pad_id,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_new_tokens = max(len(out) for _, out in (train_data + dev_data + test_data)) + 1
    print(f"max_new_tokens={max_new_tokens}")

    records: List[EpochRecord] = []
    best_dev_acc = -1.0
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            pad_id=pad_id,
            epoch=epoch,
        )

        dev_metrics = evaluate_model(
            model=model,
            loader=dev_loader,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
            eval_batch=eval_batch,
        )

        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
            eval_batch=eval_batch,
        )

        line = (
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )
        print(line)
        log_file.write(line + "\n")

        records.append(
            EpochRecord(
                seed_split_data=seed_split_data,
                seed_for_training=seed_for_training,
                d_model=d_model,
                n_layers=n_layers,
                exposure_ratio=exposure_ratio,
                epoch=epoch,
                train_loss=train_loss,
                dev_acc=dev_metrics["accuracy"],
                test_acc=test_metrics["accuracy"],
            )
        )

        if dev_metrics["accuracy"] > best_dev_acc:
            best_dev_acc = dev_metrics["accuracy"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        elif patience is not None:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no dev improvement for {patience} epochs)")
                break

    if save_weights and best_state_dict is not None:
        weight_path = os.path.join(
            saved_folder,
            f"seedsplit{seed_split_data}_seedtrain{seed_for_training}_dmodel{d_model}_nheads{n_heads}_dff{d_ff}_layer{n_layers}.pt",
        )
        torch.save(best_state_dict, weight_path)
        print(f"Saved best weights (dev_acc={best_dev_acc:.3f}) to {weight_path}")

    log_file.close()
    return records


def plot_single_run(records, run_dir, seed_split_data, seed_for_training, d_model, n_heads, d_ff, n_layers, variant_label):
    recs = [asdict(r) for r in records]
    epochs = [r["epoch"] for r in recs]
    arch = f"d={d_model}, h={n_heads}, ff={d_ff}, L={n_layers}"
    seed_label = f"seed_split={seed_split_data}, seed_train={seed_for_training}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [r["train_loss"] for r in recs], marker="o", markersize=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title(f"Train loss ({arch}, {variant_label}, {seed_label})")
    ax1.grid(True)

    ax2.plot(epochs, [r["dev_acc"] for r in recs], marker="o", markersize=2, label="dev")
    ax2.plot(epochs, [r["test_acc"] for r in recs], marker="o", markersize=2, label="test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy ({arch}, {variant_label}, {seed_label})")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fname = f"seedsplit{seed_split_data}_seedtrain{seed_for_training}_dmodel{d_model}_nheads{n_heads}_dff{d_ff}_layer{n_layers}_curves.png"
    plt.savefig(os.path.join(run_dir, fname), dpi=150)
    plt.close()
    print(f"Saved per-run plots to {run_dir}/{fname}")


def run_sweep(
    seeds_split_data=(42,),
    seeds_for_training=(42,),
    d_models=(128,),
    n_heads_list=(4,),
    d_ffs=(None,),
    n_layers_list=(4,),
    exposure_ratios=(1.0,),
    epochs=100,
    lr=1e-3,
    batch_size=32,
    max_depth=3,
    max_commands_per_depth=None,
    train_fraction=0.8,
    save_weights=False,
    base_results_dir="results_scan_random",
    patience=None,
    data_mode="exposure",
    size_variation_ps=(1, 2, 4, 8, 16, 32, 64),
    eval_batch=False,
    dev_fraction=0.1,
    dev_source="from_train_split",
):
    all_records = []

    if data_mode == "size_variation":
        variants = list(size_variation_ps)
        get_run_dir = lambda v: os.path.join(base_results_dir, f"size_p{v}")
        get_label = lambda v: f"p={v}%"
        get_kwargs = lambda v: {"data_mode": "size_variation", "size_variation_p": v}
    else:
        variants = list(exposure_ratios)
        get_run_dir = lambda v: os.path.join(base_results_dir, exposure_tag(v))
        get_label = lambda v: f"exp={v}"
        get_kwargs = lambda v: {"data_mode": "exposure", "exposure_ratio": v}

    for variant in variants:
        run_dir = get_run_dir(variant)
        os.makedirs(run_dir, exist_ok=True)

        for seed_split_data in seeds_split_data:
            for seed_for_training in seeds_for_training:
                for d_model in d_models:
                    for n_heads in n_heads_list:
                        for d_ff in d_ffs:
                            resolved_d_ff = 4 * d_model if d_ff is None else d_ff
                            for n_layers in n_layers_list:
                                print("=" * 80)
                                print(
                                    f"Running seed_split_data={seed_split_data}, seed_for_training={seed_for_training}, "
                                    f"d_model={d_model}, n_heads={n_heads}, "
                                    f"d_ff={resolved_d_ff}, n_layers={n_layers}, {get_label(variant)}"
                                )
                                records = run_single_config(
                                    seed_split_data=seed_split_data,
                                    seed_for_training=seed_for_training,
                                    d_model=d_model,
                                    n_heads=n_heads,
                                    d_ff=resolved_d_ff,
                                    n_layers=n_layers,
                                    epochs=epochs,
                                    lr=lr,
                                    batch_size=batch_size,
                                    max_depth=max_depth,
                                    max_commands_per_depth=max_commands_per_depth,
                                    train_fraction=train_fraction,
                                    save_weights=save_weights,
                                    saved_folder=run_dir,
                                    patience=patience,
                                    eval_batch=eval_batch,
                                    dev_fraction=dev_fraction,
                                    dev_source=dev_source,
                                    **get_kwargs(variant),
                                )
                                plot_single_run(
                                    records, run_dir, seed_split_data, seed_for_training,
                                    d_model, n_heads, resolved_d_ff, n_layers, get_label(variant),
                                )
                                all_records.extend([asdict(r) for r in records])

    return all_records


# ============================================================
# Plotting
# ============================================================

def best_dev_records(records):
    best = {}
    for r in records:
        key = (r["seed_split_data"], r["seed_for_training"], r["d_model"], r["n_layers"], r["exposure_ratio"])
        if key not in best or r["dev_acc"] > best[key]["dev_acc"]:
            best[key] = r
    return list(best.values())


def plot_train_loss(records, savepath):
    plt.figure(figsize=(8, 5))

    grouped = {}
    for r in records:
        key = (r["d_model"], r["n_layers"], r["exposure_ratio"], r["epoch"])
        grouped.setdefault(key, []).append(r["train_loss"])

    curves = {}
    for (d_model, n_layers, exposure_ratio, epoch), vals in grouped.items():
        curves.setdefault((d_model, n_layers, exposure_ratio), []).append((epoch, np.mean(vals)))

    for (d_model, n_layers, exposure_ratio), pts in sorted(curves.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"d={d_model}, L={n_layers}, exp={exposure_ratio}")

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("SCAN random-split train loss")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_test_accuracy_vs_exposure(records, savepath):
    plt.figure(figsize=(8, 5))
    finals = best_dev_records(records)

    grouped = {}
    for r in finals:
        grouped.setdefault((r["d_model"], r["n_layers"], r["exposure_ratio"]), []).append(r["test_acc"])

    lines = {}
    for (d_model, n_layers, exposure_ratio), vals in grouped.items():
        lines.setdefault((d_model, n_layers), []).append((exposure_ratio, np.mean(vals), np.std(vals)))

    for (d_model, n_layers), pts in sorted(lines.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [p[2] for p in pts]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=f"d={d_model}, L={n_layers}")

    plt.xlabel("Train exposure ratio")
    plt.ylabel("Random-split test exact-match accuracy")
    plt.title("SCAN random-split test accuracy vs exposure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()



# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    # allow any config key to be overridden from the command line
    parser.add_argument("--base_results_dir", type=str)
    parser.add_argument("--seeds_split_data", type=int, nargs="+")
    parser.add_argument("--seeds_for_training", type=int, nargs="+")
    parser.add_argument("--d_models", type=int, nargs="+")
    parser.add_argument("--n_heads_list", type=int, nargs="+")
    parser.add_argument("--d_ffs", type=int, nargs="+")
    parser.add_argument("--n_layers_list", type=int, nargs="+")
    parser.add_argument("--exposure_ratios", type=float, nargs="+")
    parser.add_argument("--data_mode", type=str, choices=["exposure", "size_variation"])
    parser.add_argument("--size_variation_ps", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--max_commands_per_depth", type=int)
    parser.add_argument("--train_fraction", type=float)
    parser.add_argument("--save_weights", action="store_true", default=None)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval_train_every", type=int)  # kept for backward compat with old configs, ignored
    parser.add_argument("--eval_batch", action="store_true", default=None)
    parser.add_argument("--dev_fraction", type=float)
    parser.add_argument("--dev_source", type=str, choices=["from_train_split", "from_test_split"])
    parser.add_argument(
        "--check_split_determinism", action="store_true", default=False,
        help="Just verify build_data_split is deterministic for each seed_split_data, then exit (no training).",
    )
    args = parser.parse_args()

    if args.check_split_determinism:
        cfg_check = {
            "seeds_split_data": [42],
            "max_depth": 3, "max_commands_per_depth": 1000, "train_fraction": 0.8,
            "exposure_ratio": 1.0, "dev_fraction": 0.1, "dev_source": "from_train_split",
            "data_mode": "exposure", "size_variation_p": None,
        }
        if args.config is not None:
            with open(args.config) as f:
                file_cfg = json.load(f)
            for k in ("seeds_split_data", "max_depth", "max_commands_per_depth", "train_fraction",
                      "dev_fraction", "dev_source", "data_mode"):
                if k in file_cfg:
                    cfg_check[k] = file_cfg[k]
            if "exposure_ratios" in file_cfg:
                cfg_check["exposure_ratio"] = file_cfg["exposure_ratios"][0]
            if "size_variation_ps" in file_cfg:
                cfg_check["size_variation_p"] = file_cfg["size_variation_ps"][0]
        for seed_split_data in cfg_check.pop("seeds_split_data"):
            check_seed_split_data_determinism(seed_split_data=seed_split_data, n_trials=3, **cfg_check)
        raise SystemExit(0)

    # defaults
    cfg = {
        "base_results_dir": "results_scan",
        "seeds_split_data": [42],
        "seeds_for_training": [42],
        "d_models": [128],
        "n_heads_list": [4],
        "d_ffs": [None],
        "n_layers_list": [4],
        "data_mode": "exposure",
        "exposure_ratios": [1.0],
        "size_variation_ps": [1, 2, 4, 8, 16, 32, 64],
        "eval_batch": True,
        "epochs": 1000,
        "lr": 1e-3,
        "batch_size": 32,
        "max_depth": 3,
        "max_commands_per_depth": 1000,
        "train_fraction": 0.8,
        "save_weights": True,
        "patience": 300,
        "dev_fraction": 0.1,
        "dev_source": "from_train_split",
    }

    if args.config is not None:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # command-line args override config file
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            cfg[key] = val

    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    records = run_sweep(
        seeds_split_data=cfg["seeds_split_data"],
        seeds_for_training=cfg["seeds_for_training"],
        d_models=cfg["d_models"],
        n_heads_list=cfg["n_heads_list"],
        d_ffs=cfg["d_ffs"],
        n_layers_list=cfg["n_layers_list"],
        exposure_ratios=cfg["exposure_ratios"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        max_depth=cfg["max_depth"],
        max_commands_per_depth=cfg["max_commands_per_depth"],
        train_fraction=cfg["train_fraction"],
        save_weights=cfg["save_weights"],
        base_results_dir=base_results_dir,
        patience=cfg["patience"],
        data_mode=cfg["data_mode"],
        size_variation_ps=cfg["size_variation_ps"],
        eval_batch=cfg["eval_batch"],
        dev_fraction=cfg["dev_fraction"],
        dev_source=cfg["dev_source"],
    )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(base_results_dir, f"scan2_decoder_only_results_{date_time}.json")
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)

    plot_train_loss(records, savepath=os.path.join(base_results_dir, "scan_random_train_loss.png"))
    plot_test_accuracy_vs_exposure(records, savepath=os.path.join(base_results_dir, "scan_random_test_accuracy_vs_exposure.png"))

    print(f"Saved results to {results_path}")
    print(f"Saved outputs in {base_results_dir}/")