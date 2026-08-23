import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


VALID_LANGS = ["en", "es", "fr", "ru", "uk", "hu", "nl", "vi", "zh", "ja", "ko"]


def infer_fixed_prefix(df):
    prefixes = ["fixed_subspace_", "fixed_translation_vector_"]
    labels = list(map(str, df.index)) + list(map(str, df.columns))
    for prefix in prefixes:
        if any(x.startswith(prefix) for x in labels):
            return prefix
    raise ValueError("Could not infer fixed prefix.")


def load_auc_matrix(csv_path, *, valid_langs=VALID_LANGS):
    df = pd.read_csv(csv_path)

    if "language" in df.columns:
        df = df.set_index("language")
    else:
        df = df.set_index(df.columns[0])

    df = df.apply(pd.to_numeric, errors="coerce")

    fixed_prefix = infer_fixed_prefix(df)

    # save en_fact
    en_fact_series = df["en_fact"].copy() if "en_fact" in df.columns else None

    df = df.drop(columns=[c for c in df.columns if "en_fact" in str(c)], errors="ignore")

    row_has_fixed = df.index.astype(str).str.startswith(fixed_prefix).any()
    col_has_fixed = pd.Index(df.columns.astype(str)).str.startswith(fixed_prefix).any()

    if col_has_fixed:
        fixed_cols = [c for c in df.columns if str(c).startswith(fixed_prefix)]
        df = df[fixed_cols]
        df.columns = [str(c).replace(fixed_prefix, "") for c in df.columns]
        df.index = df.index.astype(str)
        df = df.T

    elif row_has_fixed:
        df = df[df.index.astype(str).str.startswith(fixed_prefix)]
        df.index = [str(r).replace(fixed_prefix, "") for r in df.index]
        df.columns = [str(c).replace(fixed_prefix, "") for c in df.columns]

    else:
        raise ValueError("No fixed mode rows or columns found.")

    ordered = [l for l in valid_langs if l in df.index and l in df.columns]
    df = df.loc[ordered, ordered]

    if en_fact_series is not None:
        en_fact_series.index = en_fact_series.index.astype(str)
        en_fact_series = en_fact_series.loc[ordered]

    return df, en_fact_series


def plot_auc_heatmap(
    raw_df,
    en_fact_series=None,
    *,
    add_en_fact_row=True,
    save_path=None,
    title=None,
    figsize=(12, 7),
    annotate=True,
):
    # --- add language-specific row ---
    if add_en_fact_row and en_fact_series is not None:
        raw_df = pd.concat(
            [pd.DataFrame([en_fact_series.values], index=["language-specific"], columns=raw_df.columns),
             raw_df]
        )

    # --- color normalization (column-wise) ---
    col_min = raw_df.min(axis=0)
    col_max = raw_df.max(axis=0)
    color_df = (raw_df - col_min) / (col_max - col_min + 1e-12)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_red", ["#ffffff", "#ff0000"]
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(color_df.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(raw_df.shape[1]))
    ax.set_xticklabels(raw_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(raw_df.shape[0]))
    ax.set_yticklabels(raw_df.index)

    ax.set_xlabel("target language")
    ax.set_ylabel("source (fixed / language-specific)")
    ax.set_title(title or "AUC heatmap")

    if annotate:
        for i in range(raw_df.shape[0]):
            for j in range(raw_df.shape[1]):
                val = raw_df.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    # --- highlight diagonal (skip language-specific row) ---
    offset = 1 if add_en_fact_row and en_fact_series is not None else 0
    for i in range(len(raw_df.columns)):
        if i + offset < raw_df.shape[0]:
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.5, i + offset - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("relative AUC (per column)")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="svg")
        print("Saved:", save_path)

    plt.show()
    plt.close(fig)


def plot_auc_diagsum_and_rowsum_bars(
    raw_df,
    en_fact_series=None,
    *,
    add_en_fact_row=True,
    save_path=None,
    title=None,
):
    vals = raw_df.values.astype(float)

    diag_sum = np.nansum(np.diag(vals))
    row_sums = np.nansum(vals, axis=1)

    labels = []
    values = []

    if add_en_fact_row and en_fact_series is not None:
        labels.append("language-specific")
        values.append(np.nansum(en_fact_series.values))

    labels.append("diagonal_sum")
    values.append(diag_sum)

    for lang, s in zip(raw_df.index, row_sums):
        labels.append(f"fixed_{lang}_sum")
        values.append(s)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("raw AUC sum")
    ax.set_title(title or "AUC summary")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(values)*0.98, max(values) * 1.02)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="svg")
        print("Saved:", save_path)

    plt.show()
    plt.close(fig)

