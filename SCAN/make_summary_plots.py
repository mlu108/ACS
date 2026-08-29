#!/usr/bin/env python3
"""Aggregate summary plots across all he_probe analysis runs.

Plots produced (all saved as SVG in {results_dir}/Plots/):

  Per fixed size_p
    p{N}_accuracy_by_setup.svg
        Bar chart: overall accuracy per (d_model, d_ff, n_layers) setup.

    p{N}_ortho_by_layer_sel{MODE}.svg  [× 3 select modes]
        Line chart: mean orthogonality per layer, one line per setup.
        Best layer marked with a star (★) per setup.

  Across all size_p values
    accuracy_by_exposure.svg
        Line chart: accuracy vs size_p, one line per setup.

    ortho_by_exposure_sel{MODE}.svg  [× 3 select modes]
        Line chart: best-layer mean orthogonality vs size_p, one line per setup.
        (avg across all layers also drawn as thin dashed line for reference.)
"""

# Adapted from ryeii/Representational-Homomorphism-for-Transformer-Language-Models
# (github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models),
# make_summary_plots.py.

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SELECT_MODES = ["cumulative", "noncumulative", "cumulative_auc_and_direction"]

# ── colour palette ────────────────────────────────────────────────────────────
_CMAP = plt.cm.get_cmap("tab10")

def _color(i: int):
    return _CMAP(i % 10)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_run_name(run_name: str) -> dict | None:
    m = re.match(
        r"seed(\d+)_dmodel(\d+)_nheads(\d+)_dff(\d+)_layer(\d+)"
        r"_p=(\d+)%_\w+_sel(\w+)$",
        run_name,
    )
    if not m:
        return None
    return {
        "seed":              int(m.group(1)),
        "d_model":           int(m.group(2)),
        "n_heads":           int(m.group(3)),
        "d_ff":              int(m.group(4)),
        "n_layers":          int(m.group(5)),
        "size_p":            int(m.group(6)),
        "select_best_layer": m.group(7),
    }


def _setup_label(d_model: int, d_ff: int, n_layers: int) -> str:
    return f"d={d_model} ff={d_ff} L={n_layers}"


def _mean_std(values) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_records(results_dir: str) -> list[dict]:
    records = []
    for size_dir in sorted(Path(results_dir).iterdir()):
        if not size_dir.is_dir() or not size_dir.name.startswith("size_p"):
            continue
        analysis_dir = size_dir / "analysis_results"
        if not analysis_dir.exists():
            continue
        for run_dir in sorted(analysis_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            meta = _parse_run_name(run_dir.name)
            if meta is None:
                print(f"  warning: could not parse run name: {run_dir.name!r}")
                continue
            with open(summary_path) as f:
                summary = json.load(f)

            layer_keys = sorted(
                [k for k in summary if isinstance(k, str) and k.isdigit()],
                key=int,
            )
            if not layer_keys:
                continue

            accuracy        = float(summary[layer_keys[0]]["mean_accuracy"])
            ortho_by_layer  = {int(k): float(summary[k]["mean_orthogonality"]) for k in layer_keys}
            avg_ortho       = float(np.mean(list(ortho_by_layer.values())))
            best_layer      = summary.get("best_layer")
            best_layer_ortho = (
                ortho_by_layer.get(int(best_layer), float("nan"))
                if best_layer is not None else float("nan")
            )

            records.append({
                **meta,
                "accuracy":         accuracy,
                "ortho_by_layer":   ortho_by_layer,
                "avg_ortho":        avg_ortho,
                "best_layer":       int(best_layer) if best_layer is not None else None,
                "best_layer_ortho": best_layer_ortho,
            })

    print(f"Loaded {len(records)} records from {results_dir}")
    return records


def _dedup_for_accuracy(records: list[dict]) -> list[dict]:
    """Return one record per (seed, size_p, setup) — accuracy is layer-independent."""
    seen: set = set()
    out = []
    for r in records:
        key = (r["seed"], r["size_p"], r["d_model"], r["d_ff"], r["n_layers"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ── plot 1 ────────────────────────────────────────────────────────────────────

def plot1_accuracy_by_setup(records: list[dict], size_p: int, plots_dir: str) -> None:
    """Bar: accuracy per (d_model,d_ff,n_layers) setup, fixed size_p."""
    subset = [r for r in records if r["size_p"] == size_p]
    if not subset:
        return

    by_setup: dict[str, list] = defaultdict(list)
    for r in subset:
        by_setup[_setup_label(r["d_model"], r["d_ff"], r["n_layers"])].append(r["accuracy"])

    labels = sorted(by_setup)
    means  = [_mean_std(by_setup[l])[0] for l in labels]
    stds   = [_mean_std(by_setup[l])[1] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[_color(i) for i in range(len(labels))], alpha=0.85)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Accuracy by model setup  |  size_p = {size_p}%")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, f"p{size_p}_accuracy_by_setup.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ── plot 2 ────────────────────────────────────────────────────────────────────

def plot2_ortho_by_layer(
    records: list[dict], size_p: int, select_mode: str, plots_dir: str
) -> None:
    """Line: mean orthogonality vs layer per setup, best layer starred. Fixed size_p."""
    subset = [r for r in records if r["size_p"] == size_p and r["select_best_layer"] == select_mode]
    if not subset:
        return

    by_setup: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    best_layers_by_setup: dict[str, list] = defaultdict(list)
    for r in subset:
        label = _setup_label(r["d_model"], r["d_ff"], r["n_layers"])
        for layer, ortho in r["ortho_by_layer"].items():
            by_setup[label][layer].append(ortho)
        if r["best_layer"] is not None:
            best_layers_by_setup[label].append(r["best_layer"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(sorted(by_setup)):
        color  = _color(i)
        layers = sorted(by_setup[label])
        means  = [np.mean(by_setup[label][l]) for l in layers]
        stds   = [np.std(by_setup[label][l]) for l in layers]
        ax.errorbar(layers, means, yerr=stds, marker="o", markersize=4,
                    color=color, label=label, capsize=2)

        if best_layers_by_setup[label]:
            best_l = int(round(np.median(best_layers_by_setup[label])))
            if best_l in by_setup[label]:
                best_ortho = float(np.mean(by_setup[label][best_l]))
                ax.scatter([best_l], [best_ortho], s=160, marker="*",
                           color=color, zorder=5, edgecolors="black", linewidths=0.5)

    # legend entry for the star
    ax.scatter([], [], s=160, marker="*", color="gray",
               edgecolors="black", linewidths=0.5, label="best layer (★)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean orthogonality")
    ax.set_title(
        f"Mean orthogonality by layer  |  size_p = {size_p}%  |  sel = {select_mode}"
    )
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, f"p{size_p}_ortho_by_layer_sel{select_mode}.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ── plot 3 ────────────────────────────────────────────────────────────────────

def plot3_accuracy_by_exposure(records: list[dict], plots_dir: str) -> None:
    """Line: accuracy vs size_p, one line per (d_model,d_ff,n_layers) setup."""
    by_setup: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        label = _setup_label(r["d_model"], r["d_ff"], r["n_layers"])
        by_setup[label][r["size_p"]].append(r["accuracy"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(sorted(by_setup)):
        sizes  = sorted(by_setup[label])
        means  = [_mean_std(by_setup[label][s])[0] for s in sizes]
        stds   = [_mean_std(by_setup[label][s])[1] for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", markersize=5,
                    color=_color(i), label=label, capsize=3)

    ax.set_xlabel("size_p (%)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Accuracy vs exposure (size_p)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, "accuracy_by_exposure.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ── plot 4 ────────────────────────────────────────────────────────────────────

def plot4_ortho_by_exposure(
    records: list[dict], select_mode: str, plots_dir: str
) -> None:
    """Line: best-layer orthogonality vs size_p per setup, with avg-across-layers as dashed reference."""
    subset = [r for r in records if r["select_best_layer"] == select_mode]
    if not subset:
        return

    best_by_setup: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    avg_by_setup:  dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in subset:
        label = _setup_label(r["d_model"], r["d_ff"], r["n_layers"])
        best_by_setup[label][r["size_p"]].append(r["best_layer_ortho"])
        avg_by_setup[label][r["size_p"]].append(r["avg_ortho"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(sorted(best_by_setup)):
        color  = _color(i)
        sizes  = sorted(best_by_setup[label])

        best_means = [_mean_std(best_by_setup[label][s])[0] for s in sizes]
        best_stds  = [_mean_std(best_by_setup[label][s])[1] for s in sizes]
        avg_means  = [_mean_std(avg_by_setup[label][s])[0] for s in sizes]

        ax.errorbar(sizes, best_means, yerr=best_stds, marker="o", markersize=5,
                    color=color, label=label, capsize=3)
        ax.plot(sizes, avg_means, linestyle="--", linewidth=1,
                color=color, alpha=0.45)  # avg-across-layers reference, no legend entry

    # explain dashed lines once
    ax.plot([], [], linestyle="--", color="gray", alpha=0.6, label="avg across all layers (ref)")
    ax.set_xlabel("size_p (%)")
    ax.set_ylabel("Mean orthogonality at best layer")
    ax.set_title(f"Orthogonality vs exposure  |  sel = {select_mode}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(plots_dir, f"ortho_by_exposure_sel{select_mode}.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregate summary SVG plots from analysis_scan outputs."
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root results directory (e.g. results_scan_paper_new_dev)")
    parser.add_argument("--plot_folder_name", type=str, required=True,
                        help="Folder name to save plots (e.g. results_scan_paper_new_dev/Plots)")
    parser.add_argument("--not_include_dmodel", type=int, nargs="+", default=None,
                        help="d_model values to exclude from all plots (e.g. --not_include_dmodel 6)")
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, args.plot_folder_name)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Output directory: {plots_dir}")

    records = load_all_records(args.results_dir)
    if args.not_include_dmodel:
        exclude = set(args.not_include_dmodel)
        before = len(records)
        records = [r for r in records if r["d_model"] not in exclude]
        print(f"Excluded d_model {sorted(exclude)}: {before} → {len(records)} records")
    if not records:
        print("No records found — check that analysis_scan.py has been run first.")
        return

    all_size_ps = sorted({r["size_p"] for r in records})
    print(f"Found size_p values: {all_size_ps}")
    print(f"Found select modes:  {sorted({r['select_best_layer'] for r in records})}\n")

    dedup = _dedup_for_accuracy(records)

    # ── plot 1: accuracy by setup (per size_p, no best-layer needed) ──────────
    print("=== Plot 1: accuracy by setup ===")
    for size_p in all_size_ps:
        plot1_accuracy_by_setup(dedup, size_p, plots_dir)

    # ── plot 2: ortho vs layer (per size_p × select_mode) ────────────────────
    print("\n=== Plot 2: orthogonality by layer ===")
    for size_p in all_size_ps:
        for mode in SELECT_MODES:
            plot2_ortho_by_layer(records, size_p, mode, plots_dir)

    # ── plot 3: accuracy vs exposure (no best-layer needed) ───────────────────
    print("\n=== Plot 3: accuracy by exposure ===")
    plot3_accuracy_by_exposure(dedup, plots_dir)

    # ── plot 4: best-layer ortho vs exposure (per select_mode) ───────────────
    print("\n=== Plot 4: orthogonality by exposure ===")
    for mode in SELECT_MODES:
        plot4_ortho_by_exposure(records, mode, plots_dir)

    print(f"\nAll summary plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
