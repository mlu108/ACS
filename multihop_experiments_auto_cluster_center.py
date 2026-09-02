
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm

from helpers import resolve_model_id

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def safe_norm(x, eps=1e-12):
    return max(float(np.linalg.norm(x)), eps)


def cosine(u, v, eps=1e-12):
    nu = safe_norm(u, eps)
    nv = safe_norm(v, eps)
    return float(np.dot(u, v) / (nu * nv))


def orthogonality(u, v, mode="one_minus_abs_cos"):
    c = float(np.clip(cosine(u, v), -1.0, 1.0))
    if mode == "one_minus_abs_cos":
        return 1.0 - abs(c)
    if mode == "one_minus_cos":
        return 1.0 - c
    if mode == "angle":
        return float(np.arccos(np.clip(abs(c), -1.0, 1.0)) / np.pi)
    raise ValueError(f"Unknown ortho mode: {mode}")




def get_first_target_token_id(tokenizer, target_text):
    ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        return None

    space_id = tokenizer(" ", add_special_tokens=False)["input_ids"]
    space_id = space_id[0] if len(space_id) > 0 else None

    # if first token is space, take the next one (if exists)
    if space_id is not None and ids[0] == space_id:
        if len(ids) > 1:
            return int(ids[1])
        else:
            return None  # only space

    return int(ids[0])


def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s.casefold()


def infer_numpy_dtype(dtype_str):
    return np.float32 if str(dtype_str) == "float32" else np.float16


def compute_layerwise_correlations(xs_by_layer, y_acc_by_layer, min_examples):
    pearson_acc = []
    spearman_acc = []
    n_per_layer = []
    for L in range(len(xs_by_layer)):
        x = np.asarray(xs_by_layer[L], dtype=np.float64)
        ya = np.asarray(y_acc_by_layer[L], dtype=np.float64)
        n_per_layer.append(len(x))
        if len(x) < min_examples or np.std(x) == 0:
            pearson_acc.append(np.nan)
            spearman_acc.append(np.nan)
            continue
        try:
            pearson_acc.append(float(pearsonr(x, ya).statistic))
        except Exception:
            pearson_acc.append(np.nan)
        try:
            spearman_acc.append(float(spearmanr(x, ya).statistic))
        except Exception:
            spearman_acc.append(np.nan)
    return {
        "pearson_acc": np.asarray(pearson_acc, dtype=np.float64),
        "spearman_acc": np.asarray(spearman_acc, dtype=np.float64),
        "n_per_layer": np.asarray(n_per_layer, dtype=np.int32),
    }


def compute_shuffle_correlation_baseline(xs_by_layer, y_acc_by_layer, min_examples, n_trials=5, random_seed=0):
    trial_corrs = []
    for t in range(int(n_trials)):
        ya_trials = []
        for L in range(len(xs_by_layer)):
            rng = np.random.default_rng(int(random_seed) + 9001 * (t + 1) + 101 * L)
            ya = np.asarray(y_acc_by_layer[L], dtype=np.float64)
            ya_trials.append(ya[rng.permutation(len(ya))] if len(ya) else ya)
        trial_corrs.append(compute_layerwise_correlations(xs_by_layer, ya_trials, min_examples))
    out = {}
    for key in ["pearson_acc", "spearman_acc"]:
        out[key] = np.nanmean(np.stack([d[key] for d in trial_corrs], axis=0), axis=0)
    return out


def save_global_correlation_plots(real_corr, shuffle_corr, layers, out_dir, use_random):
    plt.figure(figsize=(9, 5))
    plt.plot(layers, real_corr["pearson_acc"], marker="o", label="Real Pearson")
    plt.plot(layers, real_corr["spearman_acc"], marker="o", label="Real Spearman")
    if use_random and shuffle_corr is not None:
        plt.plot(layers, shuffle_corr["pearson_acc"], linestyle="--", label="Random-shuffle Pearson")
        plt.plot(layers, shuffle_corr["spearman_acc"], linestyle="--", label="Random-shuffle Spearman")
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xlabel("Layer")
    plt.ylabel("Correlation")
    plt.title("Orthogonality vs Qx_GFx accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_acc_by_layer.svg"), format="svg", bbox_inches="tight")
    plt.show()
    plt.close()





def build_global_example_index(runs, summary, filter_same_answer_span=False, filter_same_first_token=False, y_accuracy_using_first_token=False):
    """
    Returns:
      eligible: list of example dicts
      row_lookup: maps global_row_id -> (run_idx, local_row_idx)
    """
    prompt_type2id = summary["prompt_type2id"]
    id_qx_fx = int(prompt_type2id["Qx_Fx"])
    id_qfx_gfx = int(prompt_type2id["QFx_GFx"])
    id_qx_gfx = int(prompt_type2id["Qx_GFx"])

    row_lookup = {}
    row_index = {}
    global_row_id = 0

    task_id_offset = 0
    offsets_by_run = {}

    # build shifted row index without loading memmaps
    for run_idx, r in enumerate(runs):
        current_tasks = list(r["summary"]["tasks"])
        offsets_by_run[r["run_dir"]] = task_id_offset

        shifted_task_id = r["task_id"] + task_id_offset

        for local_i in range(r["N"]):
            gid = global_row_id
            row_lookup[gid] = (run_idx, local_i)

            key = (
                int(shifted_task_id[local_i]),
                int(r["local_query_idx"][local_i]),
                int(r["prompt_type_id"][local_i]),
            )
            row_index[key] = gid
            global_row_id += 1

        task_id_offset += len(current_tasks)

    example_keys = sorted(set((k[0], k[1]) for k in row_index.keys()))

    eligible = []
    skipped_no_target_token = 0
    skipped_same_answer_span = 0
    skipped_same_first_token = 0

    for t_id, q_idx in tqdm(example_keys, desc="Building eligible examples"):
        k1 = (t_id, q_idx, id_qx_fx)
        k2 = (t_id, q_idx, id_qfx_gfx)
        k3 = (t_id, q_idx, id_qx_gfx)
        if k1 not in row_index or k2 not in row_index or k3 not in row_index:
            continue

        g1 = row_index[k1]
        g2 = row_index[k2]
        g3 = row_index[k3]

        run1, i1 = row_lookup[g1]
        run2, i2 = row_lookup[g2]
        run3, i3 = row_lookup[g3]

        r1, r2, r3 = runs[run1], runs[run2], runs[run3]

        if not (r1["match"][i1] == 1 and r2["match"][i2] == 1):
            continue

        fx_text = r1["pred_rows"][i1]["target"]
        gfx_text = r2["pred_rows"][i2]["target"]

        if filter_same_answer_span and normalize_text(fx_text) == normalize_text(gfx_text):
            skipped_same_answer_span += 1
            continue

        fx_first_tok_id = get_first_target_token_id(tokenizer, fx_text)
        gfx_first_tok_id = get_first_target_token_id(tokenizer, gfx_text)
        if (
            filter_same_first_token
            and fx_first_tok_id is not None
            and gfx_first_tok_id is not None
            and fx_first_tok_id == gfx_first_tok_id
        ):

            skipped_same_first_token += 1
            continue

        target_text = r3["pred_rows"][i3]["target"]
        first_tok_id = get_first_target_token_id(tokenizer, target_text)
        if first_tok_id is None:
            skipped_no_target_token += 1
            continue

        if not y_accuracy_using_first_token:
            qx_gfx_acc_val = int(r3["match"][i3])
        else:
            # match_first_token is precomputed in predictions.jsonl (argmax of
            # the first-step logits, decoded and prefix-checked against the
            # gold target) — see update_predictions.py — so this no longer
            # needs the raw first_step_logits.memmap.
            qx_gfx_acc_val = int(r3["pred_rows"][i3]["match_first_token"])

        eligible.append({
            "task": r3["pred_rows"][i3].get("task", f"task_{t_id}"),
            "task_id": int(t_id),
            "local_query_idx": int(q_idx),

            "run_idx_qx_fx": int(run1),
            "run_idx_qfx_gfx": int(run2),
            "run_idx_qx_gfx": int(run3),

            "idx_qx_fx": int(i1),
            "idx_qfx_gfx": int(i2),
            "idx_qx_gfx": int(i3),

            "fx_text": fx_text,
            "gfx_text": gfx_text,
            "target_text": target_text,
            "target_first_tok_id": int(first_tok_id),
            "qx_gfx_acc": int(qx_gfx_acc_val),
        })

    print(f"\nTotal combined eligible examples: {len(eligible)}")
    print(f"Skipped due to empty target tokenization: {skipped_no_target_token}")
    print(f"Skipped due to same answer span: {skipped_same_answer_span}")
    print(f"Skipped due to same first token: {skipped_same_first_token}")

    if len(eligible) == 0:
        raise ValueError("No eligible examples after filtering.")

    return eligible, row_lookup, offsets_by_run

"""
model_name = 'meta-llama/Llama-3.2-3B'
run_dirs = [
    'multihop_functions_resid_logits',
    'multihop_functions_resid_logits_NEW_DATASETS',
    'multihop_functions_resid_logits_NEW_DATASETS2',
    'multihop_functions_resid_logits_NEW_DATASETS3',
]
"""
model_name_dict={
    'allenai/OLMo-7B':'olmo_7b',
    'meta-llama/Llama-3.2-3B':'llama_3b',
    "Qwen/Qwen2.5-14B":'qwen_14b'
}
ortho_mode = 'one_minus_abs_cos'
cumulative_quantile_step = 0.05
per_x_Fx_GFx = True
random = True
min_examples_per_layer = 5
save_prefix = 'multihop_composition_geometry'
random_seed = 0
n_random_trials = 5
filter_same_answer_span = False
filter_same_first_token = False

y_accuracy_using_first_token = False
acc_cond = 'first_tok_acc' if y_accuracy_using_first_token else 'answer_span_acc'




# parse_known_args (not parse_args) so this still runs unmodified under a
# Jupyter kernel, which injects its own argv (e.g. "-f kernel-xxx.json") that
# would otherwise make argparse error out and kill the kernel.
_arg_parser = argparse.ArgumentParser(description="Multihop auto-cluster-center experiment")
_arg_parser.add_argument(
    "--model_name", type=str, default='allenai/OLMo-7B',
    help="Model to run the experiment for (default: allenai/OLMo-7B). Accepts "
         f"a full HF model id ({list(model_name_dict.keys())}), a short "
         f"save-name ({list(model_name_dict.values())}), a hyphenated alias "
         "(e.g. 'olmo-7b', 'llama-3b', 'qwen-14b'), or any other alias "
         "helpers.resolve_model_id understands (e.g. 'qwen-0.5b').",
)
_arg_parser.add_argument(
    "--experiments_dir", type=str, default="multihop_experiments",
    help="Root directory for auto-cluster JSON/plots (default: multihop_experiments).",
)
_arg_parser.add_argument(
    "--manual_cluster", action=argparse.BooleanOptionalAction, default=True,
    help="Use the hand-picked A/B/C dataset groups for cluster-mean-centering "
         "instead of run_auto_clustering's per-layer KMeans (default: True). "
         "Pass --no-manual_cluster to use auto-clustering instead.",
)
_args, _unknown_argv = _arg_parser.parse_known_args()
model_name = _args.model_name
EXPERIMENTS_DIR = _args.experiments_dir

# Allow the full HF id, the short save-name (llama_3b), or a hyphenated
# alias (llama-3b) on the CLI — all normalize to the same short save-name.
_full_name_by_short = {v: k for k, v in model_name_dict.items()}
_normalized_alias = model_name.strip().lower().replace('-', '_')
model_name = _full_name_by_short.get(_normalized_alias, model_name)

if model_name in model_name_dict:
    # a recognized full HF id -> its canonical short save-name
    model_save_name = model_name_dict[model_name]
elif model_name in model_name_dict.values():
    # already a canonical short save-name
    model_save_name = model_name
else:
    # any other model: match stage 1's own output-directory convention
    # (multihop_datasets_save_resid_stream_logits.py: MODEL_NAME.replace('-', '_'))
    # directly, so the two scripts always agree without needing an entry here.
    model_save_name = model_name.strip().replace('-', '_')

run_dirs = [f"multihop_functions_resid_logits_{model_save_name}"]

# Resolve an actual loadable HF id for the tokenizer. The three models above
# already have one (a full HF id, via model_name_dict); for anything else,
# fall back to the same short-alias table helpers.py uses to load models in
# stage 1 (e.g. 'qwen-0.5b' -> 'Qwen/Qwen2-0.5B'). If that alias isn't
# recognized either, assume the user already passed a loadable HF id.
if model_name in model_name_dict:
    tokenizer_model_id = model_name
else:
    try:
        tokenizer_model_id, _ = resolve_model_id(model_name)
    except ValueError:
        tokenizer_model_id = model_name

tokenizer = (
    PreTrainedTokenizerFast.from_pretrained(tokenizer_model_id)
    if tokenizer_model_id == 'allenai/OLMo-7B'
    else AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=False, trust_remote_code=True)
)
if len(run_dirs) == 0:
    raise ValueError("run_dirs must contain at least one run directory.")


def load_run_for_merge(run_dir):
    summary = load_json(os.path.join(run_dir, "summary.json"))
    pred_rows = load_jsonl(os.path.join(run_dir, "predictions.jsonl"))

    task_id = np.load(os.path.join(run_dir, "task_id.npy"))
    prompt_type_id = np.load(os.path.join(run_dir, "prompt_type_id.npy"))
    local_query_idx = np.load(os.path.join(run_dir, "local_query_idx.npy"))
    match = np.load(os.path.join(run_dir, "match.npy"))

    N = int(summary["N"])
    n_layers = int(summary["n_layers"])
    d_model = int(summary["d_model"])
    d_vocab = int(summary["d_vocab"])
    mm_dtype = infer_numpy_dtype(summary.get("dtype", "float16"))

    resid_last = np.memmap(
        os.path.join(run_dir, "resid_last_all_layers.memmap"),
        mode="r",
        dtype=mm_dtype,
        shape=(N, n_layers, d_model),
    )

    return {
        "run_dir": run_dir,
        "summary": summary,
        "pred_rows": pred_rows,
        "task_id": task_id,
        "prompt_type_id": prompt_type_id,
        "local_query_idx": local_query_idx,
        "match": match,
        "N": N,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_vocab": d_vocab,
        "mm_dtype": mm_dtype,
        "resid_last": resid_last,
    }


runs = [load_run_for_merge(rd) for rd in run_dirs]

# ---------------------------
# compatibility checks
# ---------------------------
ref = runs[0]
for r in runs[1:]:
    if ref["n_layers"] != r["n_layers"]:
        raise ValueError(
            f"n_layers mismatch: {ref['run_dir']} has {ref['n_layers']} vs "
            f"{r['run_dir']} has {r['n_layers']}"
        )
    if ref["d_model"] != r["d_model"]:
        raise ValueError(
            f"d_model mismatch: {ref['run_dir']} has {ref['d_model']} vs "
            f"{r['run_dir']} has {r['d_model']}"
        )
    if ref["d_vocab"] != r["d_vocab"]:
        raise ValueError(
            f"d_vocab mismatch: {ref['run_dir']} has {ref['d_vocab']} vs "
            f"{r['run_dir']} has {r['d_vocab']}"
        )
    if ref["mm_dtype"] != r["mm_dtype"]:
        raise ValueError(
            f"dtype mismatch: {ref['run_dir']} has {ref['mm_dtype']} vs "
            f"{r['run_dir']} has {r['mm_dtype']}"
        )
    if ref["summary"]["prompt_type2id"] != r["summary"]["prompt_type2id"]:
        raise ValueError(
            f"prompt_type2id mismatch:\n"
            f"{ref['run_dir']} -> {ref['summary']['prompt_type2id']}\n"
            f"{r['run_dir']} -> {r['summary']['prompt_type2id']}"
        )

n_layers = int(ref["n_layers"])

# ---------------------------
# merged summary metadata only
# ---------------------------
merged_tasks = []
task_id_offset = 0
offsets_by_run = {}

for r in runs:
    current_tasks = list(r["summary"]["tasks"])
    offsets_by_run[r["run_dir"]] = task_id_offset
    merged_tasks.extend(current_tasks)
    task_id_offset += len(current_tasks)

summary = dict(ref["summary"])
summary["tasks"] = merged_tasks
summary["task2id"] = {task: i for i, task in enumerate(merged_tasks)}
summary["id2task"] = {str(i): task for i, task in enumerate(merged_tasks)}




eligible, row_lookup, offsets_by_run = build_global_example_index(
    runs, summary,
    filter_same_answer_span=filter_same_answer_span,
    filter_same_first_token=filter_same_first_token,
    y_accuracy_using_first_token=y_accuracy_using_first_token,
)
# ---------------------------
# per-example orthogonality
# ---------------------------

xs_by_layer = [[] for _ in range(n_layers)]
y_acc_by_layer = [[] for _ in range(n_layers)]

for ex in tqdm(eligible, desc="Computing per-example orthogonality"):
    run_idx_1 = ex["run_idx_qx_fx"]
    run_idx_2 = ex["run_idx_qfx_gfx"]
    i1 = ex["idx_qx_fx"]
    i2 = ex["idx_qfx_gfx"]
    y_acc = float(ex["qx_gfx_acc"])

    run1 = runs[run_idx_1]
    run2 = runs[run_idx_2]

    for L in range(n_layers):
        v1 = np.asarray(run1["resid_last"][i1, L], dtype=np.float32)
        v2 = np.asarray(run2["resid_last"][i2, L], dtype=np.float32)
        x = orthogonality(v1, v2, mode=ortho_mode)

        xs_by_layer[L].append(float(x))
        y_acc_by_layer[L].append(y_acc)




# ---------------------------
# layer stats
# ---------------------------
layer_stats = []
valid_layers = []

for L in range(n_layers):
    x = np.asarray(xs_by_layer[L], dtype=np.float64)

    stat = {
        "layer": int(L),
        "n": int(len(x)),
        "mean_orthogonality": None if len(x) == 0 else float(np.mean(x)),
        "std_orthogonality": None if len(x) == 0 else float(np.std(x)),
        "min_orthogonality": None if len(x) == 0 else float(np.min(x)),
        "max_orthogonality": None if len(x) == 0 else float(np.max(x)),
        "constant_orthogonality": bool(len(x) > 0 and np.allclose(x, x[0])),
    }
    layer_stats.append(stat)

    if len(x) >= min_examples_per_layer and not stat["constant_orthogonality"]:
        valid_layers.append(L)


print(f"Valid layers with orthogonality variation: {valid_layers}")
skipped_layers = [s["layer"] for s in layer_stats if s["constant_orthogonality"]]
print(f"Skipped constant-orthogonality layers: {skipped_layers}")

# ---------------------------
# correlations
# ---------------------------
real_corr = compute_layerwise_correlations(
    xs_by_layer,
    y_acc_by_layer,
    min_examples_per_layer,
)

shuffle_corr = None
if random:
    shuffle_corr = compute_shuffle_correlation_baseline(
        xs_by_layer,
        y_acc_by_layer,
        min_examples_per_layer,
        n_trials=n_random_trials,
        random_seed=random_seed,
    )

corr_rows = []
for L in range(n_layers):
    corr_rows.append({
        "layer": int(L),
        "n": int(real_corr["n_per_layer"][L]),
        "pearson_acc": None if np.isnan(real_corr["pearson_acc"][L]) else float(real_corr["pearson_acc"][L]),
        "spearman_acc": None if np.isnan(real_corr["spearman_acc"][L]) else float(real_corr["spearman_acc"][L]),
        "shuffle_pearson_acc": None if shuffle_corr is None or np.isnan(shuffle_corr["pearson_acc"][L]) else float(shuffle_corr["pearson_acc"][L]),
        "shuffle_spearman_acc": None if shuffle_corr is None or np.isnan(shuffle_corr["spearman_acc"][L]) else float(shuffle_corr["spearman_acc"][L]),
    })





ALL_TASKS_QxFx_QFxGFx = {
    'antonym-french': ('antonym', 'french'),
    'antonym-german': ('antonym', 'german'),
    'antonym-spanish': ('antonym', 'spanish'),
    'landmark-country-capital': ('landmark-country', 'country-capital'),
    'mod-twenty-times-two': ('mod-twenty', 'times-two'),
    'park-country-capital': ('park-country', 'country-capital'),
    'person-university-founder': ('person-university', 'university-founder'),
    'plus-hundred-times-two': ('plus-hundred', 'times-two'),
    'plus-ten-times-two': ('plus-ten', 'times-two'),
    'product-company-ceo': ('product-company', 'company-ceo'),
    'product-company-hq': ('product-company', 'company-hq'),
    'rgb-rot120-name': ('rgb-rot120', 'rgb-name'),
    'word-int-times-two': ('word-int', 'times-two'),
    'word-substring-reverse': ('word-substring', 'reverse'),
    'person-university-year': ('person-university', 'university-year'),
    'movie-director-birthyear': ('movie-director', 'director-birthyear'),
    'book-author-birthyear': ('book-author', 'author-birthyear'),
    'song-artist-birthyear': ('song-artist', 'artist-birthyear'),
    #"director-birthyear-str": ('director-birthyear', 'birthyear-str'),
    "director-birthyear-times-two": ('director-birthyear', 'birthyear-times-two'),
    #"artist-birthyear-str": ('artist-birthyear', 'birthyear-str'),
    "artist-birthyear-times-two": ('artist-birthyear', 'birthyear-times-two'),
    #"author-birthyear-str": ('author-birthyear', 'birthyear-str'),
    "author-birthyear-times-two": ('author-birthyear', 'birthyear-times-two'),
    "num-mod7-weekday": ('num-mod7', 'int-weekday'),
    "num-mod12-month": ('num-mod12', 'int-month'),
    "int-plus5-parity":('int-plus5','int-parity'),
    "int-plus5-str":('int-plus5','int-str'),
    "int-mod4-season":('int-mod4','int-season'),
    "int-plus2-str":('int-plus2','int-str'),
    "int-plus8-str":('int-plus8','int-str'),
    "int-mod9-str":('int-mod9','int-str'),
    "int-plus2-parity":('int-plus2','int-parity'),
    "int-plus8-parity":('int-plus8','int-parity'),
}






import os
import numpy as np
from tqdm import tqdm

from multihop_experiment import (
    run_auto_clustering, load_auto_clusters, resolve_cluster_label, print_cluster_report,
)

# ---------------------------------------------------------
# Automatic per-layer clustering (replaces hand-picked A/B/C dataset groups).
# Cluster membership is now allowed to differ by layer, so task_to_cluster is
# {layer: {task: cluster_label}} rather than a flat, layer-constant dict.
# See multihop_experiment.py: KMeans over balanced per-task centroids, k
# chosen per layer by silhouette score, fit on a train-only subset by default
# (auto_cluster_using='train') to avoid leaking test examples into the
# cluster centers used to center them.
# ---------------------------------------------------------
d_model = runs[0]["d_model"]

# ---------------------------------------------------------
# Single source of truth for whether representations are cluster-mean-centered
# at all. If False, run_auto_clustering is skipped entirely (no point paying
# for per-layer KMeans/silhouette/PCA over dataset centroids) and every
# consumer of get_hop_vector() below gets the raw, uncentered residual-stream
# vector instead.
# ---------------------------------------------------------
CLUSTER_MEAN_CENTER = True

# If True, skip run_auto_clustering entirely and instead build task_to_cluster /
# exact_mu directly from the hand-picked A/B/C dataset groups below (same logic
# as the manual clustering block in multihop_experiments.py). Set via
# --manual_cluster/--no-manual_cluster (default: True).
MANUAL_CLUSTER = _args.manual_cluster

AUTO_CLUSTER_USING = "train"   # "train" (no test leakage, recommended) or "all"
AUTO_CLUSTER_DEBUG_MODE = True   # if True, print per-cluster datasets + compare against manual A/B/C
AUTO_CLUSTER_PCA_BEFORE_CLUSTERING = True   # if True, cluster on PCA-reduced centroids instead of raw ones (PCA is always computed for the debug plot regardless)
AUTO_CLUSTER_SAVE_PATH = os.path.join(
    EXPERIMENTS_DIR, "auto_clusters",
    f"auto_clusters_{AUTO_CLUSTER_USING}.json",
)
# Interactive per-layer cluster-visualization HTML plots, only written when
# AUTO_CLUSTER_DEBUG_MODE is True (see run_auto_clustering's debug_plot_dir).
AUTO_CLUSTER_DEBUG_PLOT_DIR = os.path.join(
    EXPERIMENTS_DIR, "auto_clusters",
    f"auto_cluster_plots_{AUTO_CLUSTER_USING}",
)

# Manual ground truth kept only for the AUTO_CLUSTER_DEBUG_MODE comparison
# below — it no longer drives task_to_cluster/exact_mu (see previous manual
# A/B/C block, now replaced by run_auto_clustering).
_MANUAL_C_DATASETS = ['antonym','landmark-country','french','german','spanish','country-capital','park-country','person-university','product-company',
'company-ceo','word-substring','person-university' ,'movie-director','book-author','song-artist',
            'int-weekday','int-month','int-season','int-parity','int-str',
           'num-mod7','num-mod12','int-plus5','int-mod4','int-plus2','int-plus8','int-mod9'
           'rgb-name','university-founder','company-hq',  'reverse']
_MANUAL_A_DATASETS = ['mod-twenty', 'times-two','plus-hundred','plus-ten','word-int','rgb-rot120', 'birthyear-times-two']
_MANUAL_B_DATASETS = ['university-year', 'director-birthyear','author-birthyear','artist-birthyear']

_MANUAL_B_DATASETS=['university-year', 'director-birthyear','author-birthyear','artist-birthyear']
_MANUAL_C_DATASETS=['antonym','landmark-country','french','german','spanish','country-capital','park-country','person-university','product-company',
'company-ceo','word-substring','person-university' ,'movie-director','book-author','song-artist',
            'int-parity','int-str','rgb-name','university-founder','company-hq',  'reverse',
            'int-plus5','int-mod4','int-plus2','int-plus8','int-mod9']
_MANUAL_A_DATASETS = ['mod-twenty', 'times-two','plus-hundred','plus-ten','word-int','rgb-rot120', 'birthyear-times-two']


_MANUAL_CLUSTERS = {'A': _MANUAL_A_DATASETS, 'B': _MANUAL_B_DATASETS, 'C': _MANUAL_C_DATASETS}

if not CLUSTER_MEAN_CENTER:
    print("CLUSTER_MEAN_CENTER=False: skipping auto-clustering entirely, using raw representations.")
    task_to_cluster = None
    exact_mu = None
elif MANUAL_CLUSTER:
    print("MANUAL_CLUSTER=True: using manual A/B/C dataset groups instead of run_auto_clustering.")

    task_to_cluster = {}
    for task_name, (qx, gfx) in ALL_TASKS_QxFx_QFxGFx.items():
        if qx in _MANUAL_A_DATASETS or gfx in _MANUAL_A_DATASETS:
            task_to_cluster[task_name] = 'A'
        elif qx in _MANUAL_B_DATASETS or gfx in _MANUAL_B_DATASETS:
            task_to_cluster[task_name] = 'B'
        else:
            task_to_cluster[task_name] = 'C'

    sum_raw = {'A': [np.zeros(d_model, dtype=np.float32) for _ in range(n_layers)],
               'B': [np.zeros(d_model, dtype=np.float32) for _ in range(n_layers)],
               'C': [np.zeros(d_model, dtype=np.float32) for _ in range(n_layers)]}
    counts = {'A': 0, 'B': 0, 'C': 0}

    print("Calculating exact centers of mass...")
    for ex in eligible:
        task_name = ex["task"]
        cluster_label = task_to_cluster.get(task_name, 'C')

        run_idx_1 = ex["run_idx_qx_fx"]
        run_idx_2 = ex["run_idx_qfx_gfx"]
        i1 = ex["idx_qx_fx"]
        i2 = ex["idx_qfx_gfx"]

        for L in range(n_layers):
            v1 = np.asarray(runs[run_idx_1]["resid_last"][i1, L], dtype=np.float32)
            v2 = np.asarray(runs[run_idx_2]["resid_last"][i2, L], dtype=np.float32)

            sum_raw[cluster_label][L] += v1
            sum_raw[cluster_label][L] += v2

        counts[cluster_label] += 2

    # Divide by counts to get the true mathematical mean
    exact_mu = {'A': [], 'B': [], 'C': []}
    for cluster in ['A', 'B', 'C']:
        for L in range(n_layers):
            if counts[cluster] > 0:
                exact_mu[cluster].append(sum_raw[cluster][L] / counts[cluster])
            else:
                exact_mu[cluster].append(np.zeros(d_model, dtype=np.float32))
else:
    #if os.path.exists(AUTO_CLUSTER_SAVE_PATH):
    #    print(f"Loading saved auto clusters from: {AUTO_CLUSTER_SAVE_PATH}")
    #    _auto_cluster_result = load_auto_clusters(AUTO_CLUSTER_SAVE_PATH)
    #else:
    #    print(f"No saved auto clusters at {AUTO_CLUSTER_SAVE_PATH}, computing...")
    _auto_cluster_result = run_auto_clustering(
        eligible=eligible,
        runs=runs,
        task_names=sorted({d for pair in ALL_TASKS_QxFx_QFxGFx.values() for d in pair}),
        n_layers=n_layers,
        d_model=d_model,
        auto_cluster_using=AUTO_CLUSTER_USING,
        train_ratio=0.1,
        k_range=range(2,10),#range(2,5),
        balance_n=30,
        seed=0,
        #min_silhouette_improvement=0.00,#AUTO_CLUSTER_MIN_SILHOUETTE_IMPROVEMENT,
        pca_before_clustering=AUTO_CLUSTER_PCA_BEFORE_CLUSTERING,
        pca_n_components=3,
        debug_mode=AUTO_CLUSTER_DEBUG_MODE,
        manual_clusters=_MANUAL_CLUSTERS,
        component_task_map=ALL_TASKS_QxFx_QFxGFx,
        save_path=AUTO_CLUSTER_SAVE_PATH,
        debug_plot_dir=AUTO_CLUSTER_DEBUG_PLOT_DIR,
    )

    task_to_cluster = _auto_cluster_result["task_to_cluster"]   # {layer: {single_hop_dataset_name: cluster_label}}
    exact_mu = _auto_cluster_result["exact_mu"]                 # {cluster_label: [vec_per_layer]}
    print("Auto-selected k per layer:", _auto_cluster_result["k_by_layer"])

    if AUTO_CLUSTER_DEBUG_MODE:
        print_cluster_report(
            task_to_cluster,
            manual_clusters=_MANUAL_CLUSTERS,
            component_task_map=ALL_TASKS_QxFx_QFxGFx,
            debug=_auto_cluster_result.get("debug"),
        )

# ---------------------------------------------------------
# Every consumer of an example's q1 (qx) / q2 (gfx) vector — the example-example
# experiment, build_example_subspace_scores, and compute_tasklevel_geometry_from_runs
# (subspace-subspace) — goes through get_hop_vector() instead of indexing
# runs[...]["resid_last"] directly and deciding again whether/how to center.
# That way there is exactly one place that can make the centered/uncentered
# decision, so it can't drift out of sync between experiments. See
# CLUSTER_MEAN_CENTER above.
# ---------------------------------------------------------

def get_hop_vector(ex, layer, hop):
    """
    The single accessor for an example's q1 (qx / QxFx) or q2 (gfx / QFxGFx)
    representation vector at `layer`, with cluster mean-centering applied
    according to the single CLUSTER_MEAN_CENTER flag above (using the single
    task_to_cluster/exact_mu/ALL_TASKS_QxFx_QFxGFx computed once above).
    """
    if hop == "q1":
        run_idx, i = ex["run_idx_qx_fx"], ex["idx_qx_fx"]
        dataset_name = ALL_TASKS_QxFx_QFxGFx.get(ex["task"], (None, None))[0]
    elif hop == "q2":
        run_idx, i = ex["run_idx_qfx_gfx"], ex["idx_qfx_gfx"]
        dataset_name = ALL_TASKS_QxFx_QFxGFx.get(ex["task"], (None, None))[1]
    else:
        raise ValueError(f"hop must be 'q1' or 'q2', got {hop!r}")

    v = np.asarray(runs[run_idx]["resid_last"][i, layer], dtype=np.float32)

    if not CLUSTER_MEAN_CENTER:
        return v

    if MANUAL_CLUSTER:
        # Manual task_to_cluster is keyed by the composite two-hop task name
        # (matching multihop_experiments.py) and applies one shared cluster
        # mean to both hops of an example -- not by single-hop dataset name.
        cluster_label = task_to_cluster.get(ex["task"], 'C')
    else:
        cluster_label = resolve_cluster_label(dataset_name, layer, task_to_cluster, 'C')
    if cluster_label is None:
        return v
    return v - np.asarray(exact_mu[cluster_label][layer], dtype=np.float32)


# ---------------------------------------------------------
# Step 2: Compute Cluster-Centered Orthogonality
# ---------------------------------------------------------




def compute_ci(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return 1.0 - abs(cos_sim)



xs_by_layer = [[] for _ in range(n_layers)]
y_acc_by_layer = [[] for _ in range(n_layers)]


for ex in tqdm(eligible, desc="Computing cluster-centered ci"):
    y_acc = float(ex["qx_gfx_acc"])

    for L in range(n_layers):
        # get_hop_vector applies (or skips) centering per the single
        # CLUSTER_MEAN_CENTER flag defined above — this is "the" example-example
        # representation, consistent with what build_example_subspace_scores /
        # compute_tasklevel_geometry_from_runs use for the same examples.
        v1_centered = get_hop_vector(ex, L, "q1")
        v2_centered = get_hop_vector(ex, L, "q2")

        x = compute_ci(v1_centered, v2_centered)

        xs_by_layer[L].append(float(x))
        y_acc_by_layer[L].append(y_acc)

import numpy as np

# This sanity check only makes sense when clustering/centering is actually
# happening (it reaches into task_to_cluster directly, not via
# get_hop_vector) — skip it entirely under CLUSTER_MEAN_CENTER=False.
if CLUSTER_MEAN_CENTER:
    # Select a layer to test
    target_layer =9

    # ---------------------------------------------------------
    # Test Setup: Accumulate vectors for each cluster
    # ---------------------------------------------------------
    from collections import defaultdict
    raw_vectors = defaultdict(list)

    print(f"Testing mathematical properties of Mean Centering at Layer {target_layer}...\n")

    for ex in eligible:
        task_name = ex["task"]
        if MANUAL_CLUSTER:
            cluster_label_1 = task_to_cluster.get(task_name, 'C')
            cluster_label_2 = cluster_label_1
        else:
            qx, gfx = ALL_TASKS_QxFx_QFxGFx.get(task_name, (None, None))
            cluster_label_1 = resolve_cluster_label(qx, target_layer, task_to_cluster, 'C')
            cluster_label_2 = resolve_cluster_label(gfx, target_layer, task_to_cluster, 'C')

        run_idx_1 = ex["run_idx_qx_fx"]
        run_idx_2 = ex["run_idx_qfx_gfx"]
        i1 = ex["idx_qx_fx"]
        i2 = ex["idx_qfx_gfx"]

        if cluster_label_1 is not None:
            v1 = np.asarray(runs[run_idx_1]["resid_last"][i1, target_layer], dtype=np.float32)
            raw_vectors[cluster_label_1].append(v1)
        if cluster_label_2 is not None:
            v2 = np.asarray(runs[run_idx_2]["resid_last"][i2, target_layer], dtype=np.float32)
            raw_vectors[cluster_label_2].append(v2)

    # ---------------------------------------------------------
    # Perform the Test for each Cluster
    # ---------------------------------------------------------
    for cluster in sorted(raw_vectors.keys()):
        if len(raw_vectors[cluster]) == 0:
            continue

        # Convert list of vectors into a 2D numpy matrix: Shape = (N_samples, d_model)
        cluster_matrix = np.stack(raw_vectors[cluster])

        # 1. Calculate the BEFORE mean (The Anisotropic Bias / Center of Mass)
        # We take the mean across the rows (axis=0) to get a single d_model vector
        mu_before = np.mean(cluster_matrix, axis=0)
        magnitude_before = np.linalg.norm(mu_before)

        # 2. Perform the Mean Centering operation
        # Numpy broadcasting automatically subtracts the 1D mean vector from every row
        centered_matrix = cluster_matrix - mu_before

        # 3. Calculate the AFTER mean (Should be exactly a zero vector)
        mu_after = np.mean(centered_matrix, axis=0)
        magnitude_after = np.linalg.norm(mu_after)

        print(f"--- Cluster {cluster} (n={len(raw_vectors[cluster])} vectors) ---")
        print(f"Magnitude of Mean BEFORE centering: {magnitude_before:.6f}")

        if magnitude_after < 1e-5: # Floating point threshold
            print(f"Magnitude of Mean AFTER centering:  {magnitude_after:.8f}  --> [SUCCESS! Mean is 0]")
        else:
            print(f"Magnitude of Mean AFTER centering:  {magnitude_after:.8f}  --> [WARNING! Mean is NOT 0]")

        # Extra check: ensure no single dimension is hiding a non-zero mean
        max_absolute_mean_dim = np.max(np.abs(mu_after))
        print(f"Largest single dimension in after-mean: {max_absolute_mean_dim:.8f}\n")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#xs_by_layer
# Helper function to compute raw orthogonality (no centering)
def compute_raw_orthogonality(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return 1.0 - abs(cos_sim)

# Choose a specific layer to plot (e.g., a middle or late layer where anisotropy is strong)
target_layer = 13 # Change this to visualize different layers

raw_ortho_values = []
corrected_ortho_values = xs_by_layer[target_layer] # Already computed in previous block

# Calculate raw orthogonality for the exact same examples
for ex in eligible:
    run_idx_1 = ex["run_idx_qx_fx"]
    run_idx_2 = ex["run_idx_qfx_gfx"]
    i1 = ex["idx_qx_fx"]
    i2 = ex["idx_qfx_gfx"]

    v1 = np.asarray(runs[run_idx_1]["resid_last"][i1, target_layer], dtype=np.float32)
    v2 = np.asarray(runs[run_idx_2]["resid_last"][i2, target_layer], dtype=np.float32)
    
    raw_ortho = compute_raw_orthogonality(v1, v2)
    raw_ortho_values.append(raw_ortho)

# ---------------------------------------------------------
# Plotting the Distributions
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot the raw (uncentered) orthogonality
sns.kdeplot(raw_ortho_values, 
            fill=True, 
            color="red", 
            alpha=0.4, 
            label="Raw Orthogonality (Uncentered)")

# Plot the global-corrected orthogonality
sns.kdeplot(corrected_ortho_values, 
            fill=True, 
            color="blue", 
            alpha=0.4, 
            label="Global-Corrected Orthogonality (Mean-Centered)")

plt.title(f"Distribution of Task Orthogonality (Layer {target_layer})", fontsize=16)
plt.xlabel("Orthogonality ($1 - |\cos(\\theta)|$)", fontsize=14)
plt.ylabel("Density", fontsize=14)

# Set x-axis limits from 0 (perfect overlap) to 1 (perfect orthogonality)
plt.xlim(0, 1.1)

# Add a vertical dashed line at 1.0 for reference (perfect orthogonality)
plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label="Perfect Orthogonality (1.0)")

plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Print Summary Statistics
# ---------------------------------------------------------
print(f"--- Layer {target_layer} Summary Statistics ---")
print(f"Raw Orthogonality Mean:       {np.mean(raw_ortho_values):.4f}")
print(f"Corrected Orthogonality Mean: {np.mean(corrected_ortho_values):.4f}")
print(f"Mean Shift (Correction Delta): {np.mean(corrected_ortho_values) - np.mean(raw_ortho_values):.4f}")

import html
import numpy as np
import plotly.graph_objects as go


def _html_escape(x):
    return html.escape("" if x is None else str(x))


def _build_dataset_composition_html(
    ex_rows,
    *,
    dataset_field="task",
    max_dataset_rows=20,
):
    if len(ex_rows) == 0:
        return "<b>Empty bin</b>"

    dataset_counts = {}

    for ex in ex_rows:
        ds = ex.get(dataset_field, "UNKNOWN")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1


    total = len(ex_rows)
    dataset_items = sorted(dataset_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dataset_items_show = dataset_items[:max_dataset_rows]

    html_parts = [
        "<b style='font-size:13px'>Dataset composition</b><br>",
        f"<span>Total examples in bin: {total}</span><br><br>",
    ]

    for ds, c in dataset_items_show:
        pct = 100.0 * c / total
        html_parts.append(
            f"• <b>{_html_escape(ds)}</b>: {c} ({pct:.1f}%)<br>"
        )

    if len(dataset_items) > max_dataset_rows:
        html_parts.append(
            f"<br><i>... {len(dataset_items) - max_dataset_rows} more datasets</i>"
        )

    return "".join(html_parts)


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def auc_of_curve(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    return float(np.trapz(ys, xs))


from sklearn.metrics import average_precision_score as _avg_prec_score
from sklearn.metrics import precision_recall_curve as _prec_recall_curve


def compute_failure_pr_auc(x, y_correct, n_perm=200, n_boot=200, ci=0.95, seed=0):
    """
    PR-AUC for failure prediction (positive class = failure = y_correct==0).
    failure_score = 1 - x, so lower x (more orthogonal) predicts failure.

    Statistical significance (mirrors he_probe/analysis_atomic_avg.py's
    compute_failure_pr_auc_from_rows — same two tests, same rationale):
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

    Returns pr_auc, normalized_pr_auc, failure_rate, precision/recall/thresholds,
    perm_p, pr_auc_improvement_ci_lo/hi, normalized_pr_auc_ci_lo/hi.
    """
    x = np.asarray(x, dtype=np.float64)
    y_correct = np.asarray(y_correct, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y_correct)
    x, y_correct = x[mask], y_correct[mask]

    nan = float("nan")
    empty_stats = {
        "precision": np.array([]), "recall": np.array([]), "thresholds": np.array([]),
        "perm_p": nan, "pr_auc_improvement_ci_lo": nan, "pr_auc_improvement_ci_hi": nan,
        "normalized_pr_auc_ci_lo": nan, "normalized_pr_auc_ci_hi": nan, "n_boot": n_boot, "ci": ci,
    }

    if len(x) == 0:
        return {"pr_auc": nan, "normalized_pr_auc": nan, "failure_rate": nan, **empty_stats}

    y_failure = (1 - y_correct).astype(int)
    failure_rate = float(np.mean(y_failure))

    if len(np.unique(y_failure)) < 2:
        return {"pr_auc": nan, "normalized_pr_auc": nan, "failure_rate": failure_rate, **empty_stats}

    failure_score = 1.0 - x
    n = len(y_failure)
    pr_auc = float(_avg_prec_score(y_failure, failure_score))
    precision, recall, thresholds = _prec_recall_curve(y_failure, failure_score)
    normalized_pr_auc = (
        float((pr_auc - failure_rate) / (1.0 - failure_rate)) if failure_rate < 1.0 else nan
    )

    rng = np.random.default_rng(seed)
    perm_aucs = np.array([
        float(_avg_prec_score(y_failure, rng.permutation(failure_score)))
        for _ in range(n_perm)
    ])
    perm_p = float(np.mean(perm_aucs >= pr_auc)) if n_perm > 0 else nan

    boot_rng = np.random.default_rng(seed + 1)
    boot_improvement = np.full(n_boot, nan)
    boot_normalized = np.full(n_boot, nan)
    for i in range(n_boot):
        idx = boot_rng.integers(0, n, size=n)
        yb, sb = y_failure[idx], failure_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        fr_b = float(np.mean(yb))
        pr_b = float(_avg_prec_score(yb, sb))
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
        "pr_auc": pr_auc, "normalized_pr_auc": normalized_pr_auc, "failure_rate": failure_rate,
        "precision": precision, "recall": recall, "thresholds": thresholds, "perm_p": perm_p,
        "pr_auc_improvement_ci_lo": imp_lo, "pr_auc_improvement_ci_hi": imp_hi,
        "normalized_pr_auc_ci_lo": norm_lo, "normalized_pr_auc_ci_hi": norm_hi,
        "n_boot": n_boot, "ci": ci,
    }


def plot_failure_pr_curve(pr_stats, save_path=None, show=True, title=None):
    precision = pr_stats["precision"]
    recall = pr_stats["recall"]
    pr_auc = pr_stats.get("pr_auc", np.nan)
    norm_pr_auc = pr_stats.get("normalized_pr_auc", np.nan)
    failure_rate = pr_stats.get("failure_rate", np.nan)
    perm_p = pr_stats.get("perm_p", np.nan)

    if len(precision) == 0 or len(recall) == 0:
        print("No valid PR curve to plot.")
        return None

    p_str = f", p(perm)={perm_p:.3f}" if np.isfinite(perm_p) else ""
    ci_lo = pr_stats.get("normalized_pr_auc_ci_lo", np.nan)
    ci_hi = pr_stats.get("normalized_pr_auc_ci_hi", np.nan)
    ci_str = f"\nnorm 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else ""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, linewidth=2.5,
            label=f"Geometry ranking\nPR-AUC={pr_auc:.3f}, norm={norm_pr_auc:.3f}{p_str}{ci_str}")
    ax.axhline(failure_rate, linestyle="--", linewidth=1.8, color="black",
               label=f"Random baseline={failure_rate:.3f}")
    ax.set_xlabel("Recall: fraction of all failures found", fontsize=13)
    ax.set_ylabel("Precision: fraction of predicted failures that fail", fontsize=13)
    ax.set_title(title or "Failure Precision--Recall Curve", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        _json_path = os.path.splitext(save_path)[0] + ".json"
        with open(_json_path, "w") as _jf:
            json.dump({
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": np.asarray(pr_stats.get("thresholds", [])).tolist(),
                "pr_auc": float(pr_auc) if np.isfinite(pr_auc) else None,
                "normalized_pr_auc": float(norm_pr_auc) if np.isfinite(norm_pr_auc) else None,
                "failure_rate": float(failure_rate) if np.isfinite(failure_rate) else None,
                "perm_p": float(perm_p) if np.isfinite(perm_p) else None,
                "pr_auc_improvement_ci_lo": (
                    float(pr_stats["pr_auc_improvement_ci_lo"])
                    if np.isfinite(pr_stats.get("pr_auc_improvement_ci_lo", np.nan)) else None
                ),
                "pr_auc_improvement_ci_hi": (
                    float(pr_stats["pr_auc_improvement_ci_hi"])
                    if np.isfinite(pr_stats.get("pr_auc_improvement_ci_hi", np.nan)) else None
                ),
                "normalized_pr_auc_ci_lo": float(ci_lo) if np.isfinite(ci_lo) else None,
                "normalized_pr_auc_ci_hi": float(ci_hi) if np.isfinite(ci_hi) else None,
            }, _jf, indent=2)
        print("Saved:", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def cumulative_mean_from_scores(x, y, quantile_step=0.05):
    """
    Real cumulative curve:
    sort by x ascending, then cumulative mean of y over lowest-orthogonality prefixes.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    y_sorted = y[order]
    n = len(y_sorted)

    step_n = max(1, int(np.ceil(quantile_step * n)))

    xs01 = []
    cum_means = []
    counts = []

    end = step_n
    while end <= n:
        xs01.append(end / n)
        cum_means.append(float(np.mean(y_sorted[:end])))
        counts.append(end)
        end += step_n

    if len(counts) == 0 or counts[-1] != n:
        xs01.append(1.0)
        cum_means.append(float(np.mean(y_sorted)))
        counts.append(n)

    return (
        np.asarray(xs01, dtype=np.float64),
        np.asarray(cum_means, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def bin_means_from_scores(x, y, quantile_step=0.05):
    """
    Real non-cumulative bins:
      0-10, 10-20, ...
    after sorting by x ascending.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), []

    order = np.argsort(x)
    y_sorted = y[order]
    n = len(y_sorted)

    step_n = max(1, int(np.ceil(quantile_step * n)))

    means = []
    labels = []

    start = 0
    while start < n:
        end = min(start + step_n, n)
        means.append(float(np.mean(y_sorted[start:end])))
        lo_pct = 100.0 * start / n
        hi_pct = 100.0 * end / n
        labels.append(f"{lo_pct:.0f}-{hi_pct:.0f}%")
        start = end

    return np.asarray(means, dtype=np.float64), labels


def mean_random_cumulative_from_shuffle(y, quantile_step=0.05, n_trials=20, random_seed=0):
    """
    Random baseline for cumulative plot:
    shuffle y, ignore x ordering entirely, then compute the same cumulative prefixes.
    Returns mean curve and mean AUC across trials.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.array([]), np.array([]), np.nan, np.array([])

    n = len(y)
    step_n = max(1, int(np.ceil(quantile_step * n)))

    xs01 = []
    end = step_n
    while end <= n:
        xs01.append(end / n)
        end += step_n
    if len(xs01) == 0 or xs01[-1] != 1.0:
        xs01.append(1.0)
    xs01 = np.asarray(xs01, dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    all_curves = []
    all_aucs = []

    for _ in range(n_trials):
        y_perm = rng.permutation(y)
        vals = []
        for frac in xs01:
            end = int(np.ceil(frac * n))
            vals.append(float(np.mean(y_perm[:end])))
        vals = np.asarray(vals, dtype=np.float64)
        all_curves.append(vals)
        all_aucs.append(auc_of_curve(xs01, vals))

    all_curves = np.asarray(all_curves, dtype=np.float64)
    all_aucs = np.asarray(all_aucs, dtype=np.float64)

    return xs01, np.mean(all_curves, axis=0), float(np.mean(all_aucs)), all_aucs


def mean_random_bins_from_shuffle(y, quantile_step=0.05, n_trials=20, random_seed=0):
    """
    Random baseline for non-cumulative plot:
    shuffle y, then cut into the same 0-10, 10-20, ... bins.
    Returns mean bin curve and mean AUC over equally spaced bin positions.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.array([]), [], np.nan, np.array([])

    n = len(y)
    step_n = max(1, int(np.ceil(quantile_step * n)))

    labels = []
    starts_ends = []
    start = 0
    while start < n:
        end = min(start + step_n, n)
        starts_ends.append((start, end))
        lo_pct = 100.0 * start / n
        hi_pct = 100.0 * end / n
        labels.append(f"{lo_pct:.0f}-{hi_pct:.0f}%")
        start = end

    xs_plot = np.arange(len(starts_ends), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    all_curves = []
    all_aucs = []

    for _ in range(n_trials):
        y_perm = rng.permutation(y)
        vals = []
        for start, end in starts_ends:
            vals.append(float(np.mean(y_perm[start:end])))
        vals = np.asarray(vals, dtype=np.float64)
        all_curves.append(vals)
        all_aucs.append(auc_of_curve(xs_plot, vals) if len(xs_plot) > 1 else np.nan)

    all_curves = np.asarray(all_curves, dtype=np.float64)
    all_aucs = np.asarray(all_aucs, dtype=np.float64)

    return np.mean(all_curves, axis=0), labels, float(np.nanmean(all_aucs)), all_aucs


def _get_selected_indices(eligible, dataset_names, dataset_field="task"):
    dataset_names = set(dataset_names)
    return [
        i for i, ex in enumerate(eligible)
        if ex.get(dataset_field) in dataset_names
    ]


def _split_train_test(indices, train_ratio=0.1, random_seed=0):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("No indices to split.")

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(indices)

    n_train = max(1, int(round(train_ratio * len(perm))))
    if len(perm) > 1:
        n_train = min(n_train, len(perm) - 1)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    if len(test_idx) == 0:
        raise ValueError("Empty test split.")


    return train_idx.tolist(), test_idx.tolist()


def _get_xy_for_layer_and_indices(layer, indices, xs_by_layer, y_acc_by_layer):
    x = np.asarray([xs_by_layer[layer][i] for i in indices], dtype=np.float64)
    y = np.asarray([y_acc_by_layer[layer][i] for i in indices], dtype=np.float64)
    return x, y


import numpy as np
import matplotlib.pyplot as plt


def _orthogonality_bin_labels_from_x(x, quantile_step=0.05):
    """
    Returns labels like:
      ['0.12-0.18', '0.18-0.23', ...]
    for the quantile bins induced by x.
    """
    x = np.asarray(x, dtype=float)
    edges = np.arange(0.0, 1.0 + 1e-9, quantile_step)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)

    qvals = np.quantile(x, edges)
    labels = [
        f"{qvals[i]:.3f}-{qvals[i+1]:.3f}"
        for i in range(len(qvals) - 1)
    ]
    return labels, qvals


def _orthogonality_cumulative_labels_from_x(x, quantile_step=0.05):
    """
    Returns labels for cumulative buckets:
      ['<=0.18', '<=0.23', ...]
    corresponding to the cumulative quantile cutoffs.
    """
    x = np.asarray(x, dtype=float)
    xs01 = np.arange(quantile_step, 1.0 + 1e-9, quantile_step)
    if len(xs01) == 0 or xs01[-1] < 1.0:
        xs01 = np.append(xs01, 1.0)

    qvals = np.quantile(x, xs01)
    labels = [f"\u2264{q:.3f}" for q in qvals]  # ≤
    return labels, qvals


def plot_cumulative_curve_for_indices(
    layer,
    indices,
    xs_by_layer,
    y_acc_by_layer,
    *,
    quantile_step=0.05,
    n_random_trials=20,
    random_seed=0,
    title=None,
    save_path=None,
    show=True,
    show_ortho_range=False
):
    """
    Paper-style cumulative curve.

    The score x is orthogonality = 1 - |cos|, so sorting x ascending is
    equivalent to sweeping CI from high to low. The plot uses the same visual
    style as the provided MultihopQA example:
      - purple line + small circular markers
      - light purple cumulative fill/vertical bands
      - gray dashed mean-accuracy baseline
      - exactly 3 x ticks: 0.10, 0.50, 1.00
      - exactly 3 y ticks
      - x label: Minimum CI cutoff percentile (high to low)
      - y label: Mean Accuracy
      - PR-AUC / baseline annotation inside the plot
    """
    import matplotlib.ticker as mticker

    x, y = _get_xy_for_layer_and_indices(layer, indices, xs_by_layer, y_acc_by_layer)
    xs01, cum_y, counts = cumulative_mean_from_scores(x, y, quantile_step=quantile_step)
    auc = auc_of_curve(xs01, cum_y)

    rpb, rpb_p = float("nan"), float("nan")
    pr_auc = float("nan")
    pr_auc_perm_p, pr_auc_ci_lo, pr_auc_ci_hi = float("nan"), float("nan"), float("nan")
    m_pb = np.isfinite(x) & np.isfinite(y)
    if int(m_pb.sum()) >= 3 and len(np.unique(y[m_pb])) >= 2:
        try:
            rpb, rpb_p = pointbiserialr(y[m_pb], x[m_pb])
            rpb, rpb_p = -float(rpb), float(rpb_p)
        except Exception:
            pass
        try:
            # failure_score = 1 - x: lower orthogonality / higher CI predicts error
            _pr_stats = compute_failure_pr_auc(x[m_pb], y[m_pb], seed=random_seed)
            pr_auc = _pr_stats["pr_auc"]
            pr_auc_perm_p = _pr_stats["perm_p"]
            pr_auc_ci_lo = _pr_stats["pr_auc_improvement_ci_lo"]
            pr_auc_ci_hi = _pr_stats["pr_auc_improvement_ci_hi"]
        except Exception:
            pass

    baseline_pr_auc = float(1 - np.mean(y[m_pb])) if int(m_pb.sum()) >= 1 else float("nan")
    mean_acc_baseline = float(np.mean(y[m_pb])) if int(m_pb.sum()) >= 1 else float("nan")

    # -----------------------------
    # Styling to match reference plot
    # -----------------------------
    purple = "#9e3fa8"
    fill_purple = "#ead5ea"
    baseline_gray = "#c9cdcd"
    tick_gray = "#6f6f6f"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # Use a y floor below the data so the shaded area looks like the reference.
    finite_y = cum_y[np.isfinite(cum_y)]
    if len(finite_y) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        data_min = min(float(np.min(finite_y)), mean_acc_baseline if np.isfinite(mean_acc_baseline) else float(np.min(finite_y)))
        data_max = max(float(np.max(finite_y)), mean_acc_baseline if np.isfinite(mean_acc_baseline) else float(np.max(finite_y)))
        pad = max(0.025, 0.18 * (data_max - data_min + 1e-12))
        y_min = max(0.0, data_min - pad)
        y_max = min(1.0, data_max + pad)
        if y_max - y_min < 0.08:
            mid = 0.5 * (y_min + y_max)
            y_min = max(0.0, mid - 0.04)
            y_max = min(1.0, mid + 0.04)

    # Smooth purple gradient fill under the curve.
    # Make the left side noticeably darker and let the right side fade out
    # much more strongly for higher contrast.
    if len(xs01) > 0:
        # base wash
        ax.fill_between(
            xs01,
            cum_y,
            y2=y_min,
            color=fill_purple,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )

        # darker-left / lighter-right overlay
        n_seg = max(1, len(xs01) - 1)
        for k in range(n_seg):
            x0 = float(xs01[k])
            x1 = float(xs01[k + 1]) if len(xs01) > 1 else float(xs01[k])
            y0 = float(cum_y[k])
            y1 = float(cum_y[k + 1]) if len(cum_y) > 1 else float(cum_y[k])

            xseg = np.linspace(x0, x1, 96)
            yseg = np.linspace(y0, y1, 96)

            frac = k / max(1, n_seg - 1)
            alpha = 1.00 - 0.98 * frac   # ultra contrast: near-opaque left, nearly invisible right

            ax.fill_between(
                xseg,
                yseg,
                y2=y_min,
                color=fill_purple,
                alpha=max(0.01, alpha),
                linewidth=0,
                zorder=1,
            )

    ax.plot(
        xs01,
        cum_y,
        color=purple,
        marker="o",
        markersize=3.5,
        linewidth=2.2,
        markerfacecolor=purple,
        markeredgecolor=purple,
        zorder=3,
    )

    if np.isfinite(mean_acc_baseline):
        ax.axhline(mean_acc_baseline, color=baseline_gray, linestyle="--", linewidth=2.0, zorder=1)

    ax.set_xlim(0.10, 1.0)
    ax.set_ylim(y_min, y_max)

    # Exactly the 3 ticks from the reference.
    ax.set_xticks([0.10, 0.50, 1.00])
    ax.set_xticklabels(["0.10", "0.50", "1.00"])
    ax.yaxis.set_major_locator(mticker.LinearLocator(3))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.set_xlabel("Minimum CI cutoff percentile (high to low)")
    ax.set_ylabel("Mean Accuracy")
    rpb_title = f"$r_{{pb}}$={rpb:.3f}" if np.isfinite(rpb) else "$r_{pb}$=nan"
    ax.set_title(f"{title or 'MultihopQA'}\n{rpb_title}", pad=16)

    ax.grid(True, axis="both", color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color(tick_gray)
    ax.tick_params(axis="both", width=1.1, length=5, colors="black", pad=2)

    # Text annotation in the lower-right, matching the reference layout.
    pr_sig_suffix = f", p={pr_auc_perm_p:.3f}" if np.isfinite(pr_auc_perm_p) else ""
    pr_text = f"PR-AUC: {pr_auc:.3f}{pr_sig_suffix}" if np.isfinite(pr_auc) else "PR-AUC: nan"
    base_text = f"Baseline: {baseline_pr_auc:.3f}" if np.isfinite(baseline_pr_auc) else "Baseline: nan"
    ax.text(
        0.98, 0.21,
        pr_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.98, 0.12,
        base_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        color="#7f7f7f",
    )

    fig.tight_layout(pad=0.8)

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "xs01": xs01,
        "cum_y": cum_y,
        "counts": counts,
        "auc": auc,
        "pr_auc": pr_auc,
        "pr_auc_perm_p": pr_auc_perm_p,
        "pr_auc_ci_lo": pr_auc_ci_lo,
        "pr_auc_ci_hi": pr_auc_ci_hi,
        "n": len(indices),
        "rpb": rpb,
        "rpb_p": rpb_p,
        "rand_xs01": None,
        "rand_cum_y": None,
        "rand_auc": None,
        "rand_trial_aucs": None,
    }


def _error_recall_auc_perm_p(correct_sorted, observed_auc_above, n_perm=200, seed=0):
    """
    Permutation test for error_recall_auc_above_diagonal: shuffles which examples
    are correct/incorrect (independent of score-based ordering) n_perm times,
    recomputing the AUC-above-diagonal each time as the null distribution;
    perm_p = fraction of null >= observed AUC (higher AUC = errors more
    front-loaded at low scores, so this is a one-sided test). Same pattern as
    compute_failure_pr_auc above.
    """
    n = len(correct_sorted)
    total_errors = int(np.sum(1 - correct_sorted))
    if n_perm <= 0 or total_errors == 0 or total_errors == n:
        return float("nan")
    xs = np.arange(1, n + 1) / n
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(correct_sorted)
        ys = np.cumsum(1 - shuffled) / total_errors
        null_aucs[i] = np.trapz(ys - xs, xs)
    return float(np.mean(null_aucs >= observed_auc_above))


def plot_error_recall_curve_for_indices(
    layer,
    indices,
    xs_by_layer,
    y_acc_by_layer,
    *,
    title=None,
    save_path=None,
    show=True,
    n_perm=200,
    random_seed=0,
):
    """
    Cumulative error recall curve for a given layer and set of example indices.

    Sort examples by orthogonality ascending (lowest score = predicted hardest first).
    X = fraction of examples included (0..1).
    Y = fraction of total errors captured among those examples.

    Random baseline: Y = X (diagonal) — including x% of examples at random captures x% of errors.
    A curve above the diagonal means errors are concentrated at low-score examples,
    i.e. the predictor successfully front-loads failures.

    Uses the same purple/gray color scheme as plot_cumulative_curve_for_indices,
    exactly 3 x/y ticks, and a larger legend font.

    error_recall_auc_above_diagonal is permutation-tested (see
    _error_recall_auc_perm_p): perm_p is the fraction of times a random
    correct/incorrect shuffle achieves an AUC-above-diagonal at least as high
    as the observed one.

    Also writes a "plot_error_recall.json" file next to `save_path` containing the
    (orthogonality, correct) pairs and metrics needed to regenerate this figure.
    """
    import matplotlib.ticker as mticker

    purple = "#9e3fa8"
    baseline_gray = "#c9cdcd"
    tick_gray = "#6f6f6f"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    def _style_axes(ax):
        ax.xaxis.set_major_locator(mticker.LinearLocator(3))
        ax.yaxis.set_major_locator(mticker.LinearLocator(3))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color(tick_gray)
        ax.tick_params(axis="both", width=1.1, length=5, colors="black", pad=2)

    x, y = _get_xy_for_layer_and_indices(layer, indices, xs_by_layer, y_acc_by_layer)

    parent = os.path.dirname(save_path) if save_path is not None else None
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_json = os.path.join(parent, "plot_error_recall.json") if parent else "plot_error_recall.json"

    def _dump(payload):
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    correct_sorted = y[order]
    n = len(x_sorted)

    if n == 0:
        _dump({"title": title, "save_path": save_path, "rows": []})
        return {"error_recall_auc_above_diagonal": float("nan"), "perm_p": float("nan"), "error_rate": float("nan"), "n": 0}

    total_errors = int(np.sum(1 - correct_sorted))

    if total_errors == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color=baseline_gray, linewidth=2.0, label="Random baseline (x=y)")
        ax.set_xlabel("Fraction of examples included", fontsize=20)
        ax.set_ylabel("Fraction of total errors captured", fontsize=20)
        ax.set_title((title or f"Error Recall Curve") + " [no errors in this set]")
        _style_axes(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, format="svg", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        _dump({
            "title": title,
            "save_path": save_path,
            "rows": [{"orthogonality": float(xi), "correct": int(ci)} for xi, ci in zip(x_sorted, correct_sorted)],
            "error_rate": 0.0,
            "n": int(n),
        })
        return {"error_recall_auc_above_diagonal": float("nan"), "perm_p": float("nan"), "error_rate": 0.0, "n": int(n)}

    xs = np.arange(1, n + 1) / n
    ys = np.cumsum(1 - correct_sorted) / total_errors
    error_rate = total_errors / n

    perfect_xs = np.array([0.0, error_rate, 1.0])
    perfect_ys = np.array([0.0, 1.0, 1.0])

    auc_above = float(np.trapz(ys - xs, xs))
    perm_p = _error_recall_auc_perm_p(correct_sorted, auc_above, n_perm=n_perm, seed=random_seed)
    perm_p_str = f", p(perm)={perm_p:.3f}" if np.isfinite(perm_p) else ""

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=baseline_gray, linewidth=2.0, label="Random baseline (x=y)")
    ax.plot(perfect_xs, perfect_ys, linestyle=":", color=purple, alpha=0.45, linewidth=1.8,
            label=f"Perfect predictor (error rate={error_rate:.3f})")
    ax.plot(xs, ys, color=purple, marker="o", markersize=2, linewidth=2.2,
            label=f"Predictor (AUC above diag={auc_above:.3f}{perm_p_str})")
    ax.set_xlabel("Fraction of examples included", fontsize=20)
    ax.set_ylabel("Fraction of errors captured", fontsize=20)
    ax.set_title(title or f"Error Recall Curve")
    _style_axes(ax)
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    _dump({
        "title": title,
        "save_path": save_path,
        "rows": [{"orthogonality": float(xi), "correct": int(ci)} for xi, ci in zip(x_sorted, correct_sorted)],
        "error_rate": float(error_rate),
        "error_recall_auc_above_diagonal": auc_above,
        "perm_p": perm_p,
        "n": int(n),
        "total_errors": total_errors,
    })

    return {
        "error_recall_auc_above_diagonal": auc_above,
        "perm_p": perm_p,
        "error_rate": float(error_rate),
        "n": int(n),
        "total_errors": total_errors,
    }


def plot_mean_ortho_against_mean_acc_for_indices(
    layer,
    indices,
    xs_by_layer,
    y_acc_by_layer,
    eligible,
    *,
    dataset_names,
    dataset_field="task",
    annotate=True,
    title=None,
    save_path=None,
    show=True,
):
    rows = []

    for ds in dataset_names:
        ds_idxs = [i for i in indices if eligible[i].get(dataset_field) == ds]
        if len(ds_idxs) == 0:
            continue

        x = np.asarray([xs_by_layer[layer][i] for i in ds_idxs], dtype=np.float64)
        y = np.asarray([y_acc_by_layer[layer][i] for i in ds_idxs], dtype=np.float64)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        rows.append({
            "dataset": ds,
            "mean_orthogonality": float(np.mean(x)),
            "mean_accuracy": float(np.mean(y)),
            "n": int(len(x)),
            "std_orthogonality": float(np.std(x)),
            "std_accuracy": float(np.std(y)),
        })

    if len(rows) == 0:
        raise ValueError("No usable dataset-level points for the scatter plot.")

    xs = np.asarray([r["mean_orthogonality"] for r in rows], dtype=np.float64)
    ys = np.asarray([r["mean_accuracy"] for r in rows], dtype=np.float64)

    if len(rows) >= 2 and not np.allclose(xs, xs[0]) and not np.allclose(ys, ys[0]):
        pearson_r, pearson_p = pearsonr(xs, ys)
        spearman_rho, spearman_p = spearmanr(xs, ys)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_rho, spearman_p = np.nan, np.nan

    if title is None:
        title = (
            f"Layer {layer}: mean orthogonality vs mean accuracy\n"
            f"Pearson r={pearson_r:.3f} (p={pearson_p:.3g}), "
            f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g})"
        )

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys)

    if annotate:
        for r in rows:
            plt.text(
                r["mean_orthogonality"],
                r["mean_accuracy"],
                r["dataset"],
                fontsize=8,
            )

    plt.xlabel("Mean orthogonality")
    plt.ylabel("Mean accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        _json_path = os.path.splitext(save_path)[0] + ".json"
        with open(_json_path, "w") as _jf:
            json.dump({
                "rows": rows,
                "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else None,
                "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else None,
                "spearman_rho": float(spearman_rho) if np.isfinite(spearman_rho) else None,
                "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else None,
            }, _jf, indent=2)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Layer {layer}")
    print(f"Datasets used: {len(rows)}")
    print(f"Pearson r = {pearson_r:.6f}, p = {pearson_p:.6g}")
    print(f"Spearman rho = {spearman_rho:.6f}, p = {spearman_p:.6g}")

    return {
        "rows": rows,
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
        "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
        "spearman_rho": float(spearman_rho) if np.isfinite(spearman_rho) else np.nan,
        "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
    }

def plot_orthogonality_distribution_for_indices(
    layer,
    indices,
    xs_by_layer,
    *,
    bins=40,
    title=None,
    save_path=None,
    show=True,
    zoom_quantiles=None,   # e.g. (1, 99) or None
):
    """
    Plot orthogonality distribution for a subset of examples.

    Args:
        layer: int
        indices: list of example indices
        xs_by_layer: list-like where xs_by_layer[layer][i] is orthogonality
        bins: number of histogram bins
        zoom_quantiles: optional tuple (qlo, qhi) to zoom x-range to percentiles
    """
    x = np.asarray([xs_by_layer[layer][i] for i in indices], dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("No finite orthogonality values for the selected indices.")

    if zoom_quantiles is not None:
        qlo, qhi = zoom_quantiles
        lo = float(np.percentile(x, qlo))
        hi = float(np.percentile(x, qhi))
        x = x[(x >= lo) & (x <= hi)]
        if len(x) == 0:
            raise ValueError("No values remain after zoom_quantiles filtering.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))

    if np.isclose(x_min, x_max):
        eps = max(1e-6, abs(x_min) * 1e-4 + 1e-6)
        edges = np.linspace(x_min - eps, x_max + eps, bins + 1)
    else:
        edges = np.linspace(x_min, x_max, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=edges, edgecolor="black", linewidth=0.4)
    plt.xlabel("Orthogonality")
    plt.ylabel("Number of datapoints")

    if title is None:
        title = (
            f"Layer {layer} orthogonality distribution\n"
            f"n={len(x)}, mean={x_mean:.4f}, std={x_std:.4f}, "
            f"min={x_min:.4f}, max={x_max:.4f}"
        )
    plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        _json_path = os.path.splitext(save_path)[0] + ".json"
        with open(_json_path, "w") as _jf:
            json.dump({
                "n": int(len(x)),
                "mean": float(x_mean),
                "std": float(x_std),
                "min": float(x_min),
                "max": float(x_max),
                "bin_edges": edges.tolist(),
            }, _jf, indent=2)
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "x": x,
        "n": len(x),
        "mean": x_mean,
        "std": x_std,
        "min": x_min,
        "max": x_max,
        "bin_edges": edges,
    }

import numpy as np
import matplotlib.pyplot as plt


def _hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color}")
    return np.array(
        [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=np.float64,
    )


def _rgb01_to_hex(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _padded_limits(vals, pad_frac=0.08):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None, None

    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin

    if span <= 1e-12:
        pad = 0.05 if vmax == 0 else 0.08 * abs(vmax)
    else:
        pad = pad_frac * span

    return vmin - pad, vmax + pad


def _apply_three_ticks_padded_axes(ax, xs, ys, *, tick_labelsize=24):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    import numpy as np

    x_lo, x_hi = _padded_limits(xs, pad_frac=0.10)
    _, y_hi = _padded_limits(ys, pad_frac=0.10)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, y_hi)

    ax.xaxis.set_major_locator(FixedLocator(np.linspace(x_lo, x_hi, 3)))
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0.0, y_hi, 3)))

    fmt = FuncFormatter(lambda x, pos: f"{x:.1f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.tick_params(axis="both", labelsize=tick_labelsize, width=2.0, length=6)


def plot_tasklevel_feature_vs_accuracy(
    rows,
    *,
    x_keys,
    y_key="task_mean_acc",
    save_dir=None,
    show=True,
    use_markers=True,
    figsize=(8, 6),
    point_size=180,
    fit_lw=2.5,
    title_suffix="",
    file_suffix="",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import matplotlib.cm as cm

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for x_key in x_keys:
        print(x_key)
        plot_rows = [
            r for r in rows
            if x_key in r
            and y_key in r
            and np.isfinite(r[x_key])
            and np.isfinite(r[y_key])
        ]

        if len(plot_rows) == 0:
            continue

        xs = np.array([r[x_key] for r in plot_rows], dtype=float)
        xs = 1.0 - xs  # convert from orthogonality (1-CI) to CI
        ys = np.array([r[y_key] for r in plot_rows], dtype=float)
        tasks = [r["task"] for r in plot_rows]

        if len(xs) >= 2 and np.std(xs) > 1e-12 and np.std(ys) > 1e-12:
            r_val, p_val = pearsonr(xs, ys)
        else:
            r_val, p_val = np.nan, np.nan

        unique_tasks = sorted(set(tasks))
        cmap = cm.get_cmap("tab20", len(unique_tasks))

        task_to_color = {t: cmap(i) for i, t in enumerate(unique_tasks)}

        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
        task_to_marker = {
            t: markers[i % len(markers)] for i, t in enumerate(unique_tasks)
        }

        fig, ax = plt.subplots(figsize=figsize)

        for t in unique_tasks:
            idx = [i for i, tt in enumerate(tasks) if tt == t]

            ax.scatter(
                xs[idx],
                ys[idx],
                s=point_size,
                color=task_to_color[t],
                marker=task_to_marker[t] if use_markers else "o",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                label=t,
            )

        if len(xs) > 1 and np.std(xs) > 1e-12:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(
                x_line,
                m * x_line + b,
                color="black",
                linestyle="--",
                linewidth=fit_lw,
                alpha=0.8,
                label="fit",
            )

        ax.set_xlabel(x_key.replace("_ortho", "_ci").replace("angle_by_cumulative_coherence", "ci_by_cumulative_coherence"))
        ax.set_ylabel("Mean Accuracy")

        title_main = (
            "Topic Mean Vector Correlation"
            if "mean" in x_key
            else "Topic Subspace Correlation"
        )

        ps = "p<1e-3" if np.isfinite(p_val) and p_val < 1e-3 else f"p={p_val:.2g}"
        _title = f"{title_main}\nr={r_val:.3f}, {ps}{title_suffix}"
        ax.set_title(_title)

        ax.grid(alpha=0.25, linestyle="--")

        _apply_three_ticks_padded_axes(ax, xs, ys)

        ax.legend(
            title="Dataset",
            fontsize=7,
            title_fontsize=9,
            frameon=True,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

        fig.tight_layout()

        if save_dir:
            path = os.path.join(
                save_dir,
                f"{x_key}_vs_{y_key}_dataset_legend{file_suffix}.svg",
            )
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved: {path}")

            _x_label = x_key.replace("_ortho", "_ci").replace("angle_by_cumulative_coherence", "ci_by_cumulative_coherence")
            _json_data = {
                "x_key": x_key,
                "y_key": y_key,
                "x_label": _x_label,
                "y_label": "Mean Accuracy",
                "title": _title,
                "r_val": float(r_val) if np.isfinite(r_val) else None,
                "p_val": float(p_val) if np.isfinite(p_val) else None,
                "n_tasks": len(unique_tasks),
                "n_points": int(len(xs)),
                "points": [
                    {"task": tasks[i], "x": float(xs[i]), "y": float(ys[i])}
                    for i in range(len(xs))
                ],
            }
            json_path = os.path.join(
                save_dir,
                f"{x_key}_vs_{y_key}_dataset_legend{file_suffix}.json",
            )
            with open(json_path, "w") as _jf:
                json.dump(_json_data, _jf, indent=2)
            print(f"Saved: {json_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
import os
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def build_mean_and_subspace(vecs, var_prop=0.90, center=True):
    """
    vecs: list or array of shape (n, d)

    Returns:
      {
        "mean": (d,),
        "basis": (d, k),
        "rank": k,
        "n_vecs": n,
      }
    """
    X = np.asarray(vecs, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={X.shape}")

    n, d = X.shape
    mu = X.mean(axis=0)

    if n == 0:
        return {
            "mean": np.zeros(d, dtype=np.float32),
            "basis": np.zeros((d, 0), dtype=np.float32),
            "rank": 0,
            "n_vecs": 0,
            "singular_values": np.zeros(0, dtype=np.float32),
        }

    Xc = X - mu if center else X

    if n == 1 or np.allclose(Xc, 0.0):
        return {
            "mean": mu.astype(np.float32),
            "basis": np.zeros((d, 0), dtype=np.float32),
            "rank": 0,
            "n_vecs": int(n),
            "singular_values": np.zeros(0, dtype=np.float32),
        }

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    total = float(np.sum(var))

    if total <= 0:
        k = 0
    else:
        cum = np.cumsum(var) / total
        k = int(np.searchsorted(cum, var_prop) + 1)

    basis = Vt[:k].T.astype(np.float32) if k > 0 else np.zeros((d, 0), dtype=np.float32)
    singular_values = S[:k].astype(np.float32) if k > 0 else np.zeros(0, dtype=np.float32)

    return {
        "mean": mu.astype(np.float32),
        "basis": basis,
        "rank": int(k),
        "n_vecs": int(n),
        "singular_values": singular_values,
    }



import os
import json
import numpy as np
from tqdm import tqdm

from multihop_experiment import resolve_cluster_label as _resolve_cluster_label


def _get_cluster_center_for_task(task, layer, task_to_cluster, exact_mu, cluster_default="C"):
    """
    Return the precomputed cluster center mu for this task/layer.
    Assumes:
        cluster_label = _resolve_cluster_label(task, layer, task_to_cluster, cluster_default)
        exact_mu[cluster_label][layer] is a vector of shape [d_model]
    """
    if task_to_cluster is None or exact_mu is None:
        return None

    cluster_label = _resolve_cluster_label(task, layer, task_to_cluster, cluster_default)
    if cluster_label is None:
        return None
    mu = np.asarray(exact_mu[cluster_label][layer], dtype=np.float32)
    return mu


def _maybe_cluster_center_vecs(vecs, task, layer, task_to_cluster=None, exact_mu=None, cluster_default="C"):
    """
    vecs: np.ndarray of shape [n, d] or list of [d] arrays
    Returns vecs with cluster mean subtracted if requested inputs are provided.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    mu = _get_cluster_center_for_task(task, layer, task_to_cluster, exact_mu, cluster_default)
    if mu is None:
        return vecs
    return vecs - mu[None, :]


def ortho_vec_to_affine_subspace(vec, rep):
    v = np.asarray(vec, dtype=np.float32)
    mu = np.asarray(rep["mean"], dtype=np.float32)
    W = np.asarray(rep["basis"], dtype=np.float32)

    delta = v - mu
    denom = np.linalg.norm(delta)
    if denom < 1e-12:
        return 0.0

    if W.size == 0 or W.shape[1] == 0:
        return 1.0

    proj = W @ (W.T @ delta)
    resid = delta - proj
    return float(np.linalg.norm(resid) / denom)

def ortho_vec_to_linear_subspace(vec, rep):
    v = np.asarray(vec, dtype=np.float32)
    W = np.asarray(rep["basis"], dtype=np.float32)

    denom = np.linalg.norm(v)
    if denom < 1e-12:
        return 0.0

    if W.size == 0 or W.shape[1] == 0:
        return 1.0

    proj = W @ (W.T @ v)
    resid = v - proj
    return float(np.linalg.norm(resid) / denom)
def ortho_vec_to_mean(vec, rep, use_abs=True):
    v = np.asarray(vec, dtype=np.float32)
    mu = np.asarray(rep["mean"], dtype=np.float32)

    nv = np.linalg.norm(v)
    nm = np.linalg.norm(mu)

    if nv < 1e-12 or nm < 1e-12:
        return 1.0

    cos = float(np.dot(v, mu) / (nv * nm))
    if use_abs:
        cos = abs(cos)
    return float(1.0 - cos)

def compute_example_to_rep_ortho(vec, rep, mode="affine"):
    if mode == "affine":
        return ortho_vec_to_affine_subspace(vec, rep)
    elif mode == "linear_subspace":
        return ortho_vec_to_linear_subspace(vec, rep)
    elif mode == "mean":
        return ortho_vec_to_mean(vec, rep)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _get_rep_mean(rep):
    if isinstance(rep, dict):
        for k in ["mean", "mu", "mean_vec", "mean_vector"]:
            if k in rep:
                return np.asarray(rep[k], dtype=np.float32)
    raise KeyError(f"Could not find mean vector in rep keys: {rep.keys()}")


def _get_rep_basis(rep):
    if isinstance(rep, dict):
        for k in ["W", "basis", "U", "components", "subspace"]:
            if k in rep:
                return np.asarray(rep[k], dtype=np.float32)
    raise KeyError(f"Could not find subspace basis in rep keys: {rep.keys()}")


def _get_rep_sv(rep):
    if isinstance(rep, dict):
        for k in ["singular_values", "sv", "singular_vals", "sigmas"]:
            if k in rep:
                return np.asarray(rep[k], dtype=np.float32)
    return None


def build_example_subspace_subspace_scores(
    eligible_examples,
    *,
    task_reps,
    task_field="task",
    x_keys=None,
):
    """
    Each example gets task-level q1/q2 subspace-subspace metrics.
    Examples from the same task therefore share the same x-value.

    Operates only on already-built task_reps (task-level means/subspaces
    from compute_tasklevel_geometry_from_runs), never on raw per-example
    vectors, so there is no separate centering decision to make here — it
    inherits whatever centering task_reps was built with.
    """
    if x_keys is None:
        x_keys = [
            "mean_mean_ortho",
            "sub_sub_min_angle_ortho",
            "sub_sub_max_angle_ortho",
            "sub_sub_mean_angle_ortho",
            "sub_sub_principal_min_angle_ortho",
            "sub_sub_principal_max_angle_ortho",
            "sub_sub_principal_mean_angle_ortho",
            "sub_sub_fro_ortho",
            "sub_sub_mean_weighted_angle_ortho",
            "sub_sub_spectral_norm",
            "sub_sub_one_minus_spectral_norm",
            "sub_sub_one_minus_spectral_norm_unnormalized",
            "subspace_subspace_angle_by_cumulative_coherence",
            "subspace_subspace_angle_by_cumulative_coherence_weighted_by_singular_values",
            "subspace_subspace_angle_by_cumulative_coherence_L2",
            "subspace_subspace_angle_by_cumulative_coherence_sum",
            "subspace_subspace_angle_by_cumulative_coherence_full_set_mean",
        ]

    out = {k: [] for k in x_keys}
    out.update({
        "y_acc": [],
        "examples": [],
        "task": [],
    })

    task_to_scores = {}

    for ex in eligible_examples:
        task = ex[task_field]
        if task not in task_reps:
            continue

        if task not in task_to_scores:
            rep_q1 = task_reps[task]["q1"]
            rep_q2 = task_reps[task]["q2"]

            scores = {}

            if "mean_mean_ortho" in x_keys or "mean_mean_cosine" in x_keys:
                mu1 = _get_rep_mean(rep_q1)
                mu2 = _get_rep_mean(rep_q2)
                scores.update(compute_mean_geometry_features(mu1, mu2))

            subspace_keys = {
                "sub_sub_min_angle_ortho",
                "sub_sub_max_angle_ortho",
                "sub_sub_mean_angle_ortho",
                "sub_sub_principal_min_angle_ortho",
                "sub_sub_principal_max_angle_ortho",
                "sub_sub_principal_mean_angle_ortho",
                "sub_sub_fro_ortho",
                "sub_sub_mean_weighted_angle_ortho",
                "sub_sub_spectral_norm",
                "sub_sub_one_minus_spectral_norm",
                "sub_sub_one_minus_spectral_norm_unnormalized",
            }

            if any(k in x_keys for k in subspace_keys):
                W1 = _get_rep_basis(rep_q1)
                W2 = _get_rep_basis(rep_q2)
                sv1 = _get_rep_sv(rep_q1)
                sv2 = _get_rep_sv(rep_q2)
                scores.update(compute_subspace_geometry_features(W1, W2, svd_values1=sv1, svd_values2=sv2))

            if "subspace_subspace_angle_by_cumulative_coherence" in x_keys:
                scores["subspace_subspace_angle_by_cumulative_coherence"] = task_reps[task].get(
                    "subspace_subspace_angle_by_cumulative_coherence", np.nan
                )
            if "subspace_subspace_angle_by_cumulative_coherence_weighted_by_singular_values" in x_keys:
                scores["subspace_subspace_angle_by_cumulative_coherence_weighted_by_singular_values"] = task_reps[task].get(
                    "subspace_subspace_angle_by_cumulative_coherence_weighted_by_singular_values", np.nan
                )
            if "subspace_subspace_angle_by_cumulative_coherence_L2" in x_keys:
                scores["subspace_subspace_angle_by_cumulative_coherence_L2"] = task_reps[task].get(
                    "subspace_subspace_angle_by_cumulative_coherence_L2", np.nan
                )
            if "subspace_subspace_angle_by_cumulative_coherence_sum" in x_keys:
                scores["subspace_subspace_angle_by_cumulative_coherence_sum"] = task_reps[task].get(
                    "subspace_subspace_angle_by_cumulative_coherence_sum", np.nan
                )
            if "subspace_subspace_angle_by_cumulative_coherence_full_set_mean" in x_keys:
                scores["subspace_subspace_angle_by_cumulative_coherence_full_set_mean"] = task_reps[task].get(
                    "subspace_subspace_angle_by_cumulative_coherence_full_set_mean", np.nan
                )

            task_to_scores[task] = scores

        scores = task_to_scores[task]

        if any(k not in scores or not np.isfinite(scores[k]) for k in x_keys):
            continue

        for k in x_keys:
            out[k].append(float(scores[k]))

        out["y_acc"].append(float(ex["qx_gfx_acc"]))
        out["examples"].append(ex)
        out["task"].append(task)

    for k in x_keys + ["y_acc"]:
        out[k] = np.asarray(out[k], dtype=np.float32)

    return out
def build_example_subspace_scores(
    eligible_examples,
    runs,   # unused: get_hop_vector reads the global `runs` directly; kept for call-site compatibility
    layer,
    task_reps,
    *,
    task_field="task",
):
    """
    q1_vec/q2_vec come from get_hop_vector(), the single shared accessor that
    applies (or skips) cluster mean-centering per the module-level
    CLUSTER_MEAN_CENTER flag — the same one used to build task_reps (via
    compute_tasklevel_geometry_from_runs) and the example-example experiment,
    so example vectors and task subspaces are guaranteed to be centered
    consistently with each other.
    """
    out = {
        "q1vec_q2_affine": [],
        "q1vec_q2_linear_subspace": [],
        "q1vec_q2_mean": [],
        "q2vec_q1_affine": [],
        "q2vec_q1_linear_subspace": [],
        "q2vec_q1_mean": [],
        "y_acc": [],
        "examples": [],
    }

    L = layer

    for ex in eligible_examples:
        task = ex[task_field]
        if task not in task_reps:
            continue

        rep_q1 = task_reps[task]["q1"]
        rep_q2 = task_reps[task]["q2"]

        q1_vec = get_hop_vector(ex, L, "q1")
        q2_vec = get_hop_vector(ex, L, "q2")

        out["q1vec_q2_affine"].append(
            compute_example_to_rep_ortho(q1_vec, rep_q2, mode="affine")
        )
        out["q1vec_q2_linear_subspace"].append(
            compute_example_to_rep_ortho(q1_vec, rep_q2, mode="linear_subspace")
        )
        out["q1vec_q2_mean"].append(
            compute_example_to_rep_ortho(q1_vec, rep_q2, mode="mean")
        )

        out["q2vec_q1_affine"].append(
            compute_example_to_rep_ortho(q2_vec, rep_q1, mode="affine")
        )
        out["q2vec_q1_linear_subspace"].append(
            compute_example_to_rep_ortho(q2_vec, rep_q1, mode="linear_subspace")
        )
        out["q2vec_q1_mean"].append(
            compute_example_to_rep_ortho(q2_vec, rep_q1, mode="mean")
        )

        out["y_acc"].append(float(ex["qx_gfx_acc"]))
        out["examples"].append(ex)

    for k in out:
        if k != "examples":
            out[k] = np.asarray(out[k], dtype=np.float32)

    return out


def noncumulative_mean_from_scores(x, y, *, quantile_step=0.05):
    """
    Split x into quantile bins and compute mean(y) in each bin.

    Returns:
        xs01: bin positions (same style as cumulative)
        bin_means: mean y in each bin
        counts: number of points per bin
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    # sort by x (low → high orthogonality)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    n = len(x_sorted)

    # determine bin edges
    step = max(1, int(np.ceil(n * quantile_step)))
    edges = list(range(0, n, step))
    if edges[-1] != n:
        edges.append(n)

    bin_means = []
    counts = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue

        y_bin = y_sorted[lo:hi]
        bin_means.append(float(np.mean(y_bin)))
        counts.append(hi - lo)

    bin_means = np.array(bin_means, dtype=np.float32)
    counts = np.array(counts, dtype=np.int32)

    # x-axis (same convention as cumulative)
    xs01 = np.linspace(quantile_step, 1.0, len(bin_means))

    return xs01, bin_means, counts

from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def _fmt_pval(p):
    if not np.isfinite(p):
        return "nan"
    if p < 1e-3:
        return "<1e-3"
    return f"{p:.2g}"


def plot_noncumulative_curve_for_indices(
    layer,
    indices,
    xs_by_layer,
    y_acc_by_layer,
    *,
    quantile_step=0.05,
    n_random_trials=20,
    random_seed=0,
    title=None,
    save_path=None,
    show=True,
    show_ortho_range=False
):
    """
    If show_ortho_range=True, x tick labels show actual orthogonality ranges
    instead of percentile-bin names.

    Legend reports:
      - raw Pearson on example-level (x, y)
      - binned Pearson on (mean_x_per_bin, mean_y_per_bin)
    """
    x, y = _get_xy_for_layer_and_indices(layer, indices, xs_by_layer, y_acc_by_layer)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    mean_xs, bin_means, labels = bin_xy_means_from_scores(x, y, quantile_step=quantile_step)
    xs_plot = np.arange(len(bin_means))

    ylabel = "Mean Accuracy"

    rand_bin_means, rand_labels, rand_metric, rand_trial_metrics = (None, None, None, None)

    # ----- raw Pearson on example-level x, y -----
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        raw_r, raw_p = pearsonr(x, y)
        raw_r = float(raw_r)
        raw_p = float(raw_p)
        raw_r2 = float(raw_r ** 2)
    else:
        raw_r, raw_p, raw_r2 = np.nan, np.nan, np.nan

    # ----- binned Pearson on mean_xs, bin_means -----
    if len(mean_xs) >= 2 and np.std(mean_xs) > 0 and np.std(bin_means) > 0:
        bin_r, bin_p = pearsonr(mean_xs, bin_means)
        bin_r = float(bin_r)
        bin_p = float(bin_p)
        bin_r2 = float(bin_r ** 2)
    else:
        bin_r, bin_p, bin_r2 = np.nan, np.nan, np.nan

    label = (
        #f"raw Pearson: r²={raw_r2:.3f}, p={_fmt_pval(raw_p)} | "
        f"binned Pearson: r={bin_r:.3f}, p={_fmt_pval(bin_p)}"
    )

    _fs = 16
    _purple = "#9e3fa8"
    plt.figure(figsize=(8, 5))
    plt.plot(xs_plot, bin_means, color=_purple, marker="o", markersize=5,
              linewidth=2.2, markerfacecolor=_purple, markeredgecolor=_purple, label=label)

    # keep only 3 evenly-spaced ticks
    _n = len(xs_plot)
    _sel = np.round(np.linspace(0, _n - 1, min(3, _n))).astype(int)
    if show_ortho_range:
        ortho_labels, _ = _orthogonality_bin_labels_from_x(x, quantile_step=quantile_step)
        plt.xticks(xs_plot[_sel], [ortho_labels[i] for i in _sel], rotation=45, ha="right", fontsize=_fs)
        plt.xlabel("Bins Sorted by CI (high to low)", fontsize=_fs)
    else:
        plt.xticks(xs_plot[_sel], [labels[i] for i in _sel], rotation=45, ha="right", fontsize=_fs)
        plt.xlabel("Bins Sorted by CI (high to low)", fontsize=_fs)

    plt.yticks(fontsize=_fs)
    plt.locator_params(axis="y", nbins=3)
    plt.ylabel(ylabel, fontsize=_fs)
    plt.title(title or f"Layer {layer} non-cumulative accuracy\nn={len(x)}", fontsize=_fs)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=_fs)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "xs_plot": xs_plot,
        "bin_means": bin_means,
        "bin_mean_x": mean_xs,
        "labels": labels,
        "n": len(x),
        "raw_r": raw_r,
        "raw_r2": raw_r2,
        "raw_p": raw_p,
        "bin_r": bin_r,
        "bin_r2": bin_r2,
        "bin_p": bin_p,
        "rand_bin_means": rand_bin_means,
        "rand_metric": rand_metric,
        "rand_trial_metrics": rand_trial_metrics,
    }


def bin_xy_means_from_scores(x, y, quantile_step=0.05):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) == 0:
        return np.array([]), np.array([]), []

    n_bins = int(round(1.0 / quantile_step))
    qs = np.linspace(0.0, 1.0, n_bins + 1)

    edges = np.quantile(x, qs)

    mean_xs, mean_ys, labels = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)

        if mask.sum() == 0:
            continue

        mean_xs.append(float(np.mean(x[mask])))
        mean_ys.append(float(np.mean(y[mask])))
        labels.append(f"{qs[i]:.2f}-{qs[i+1]:.2f}")

    return np.array(mean_xs), np.array(mean_ys), labels


import os
import json
import numpy as np
from tqdm import tqdm


# =========================================================
# Basic helpers
# =========================================================

def _cosine_similarity(v1, v2, eps=1e-12):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return np.nan
    return float(np.dot(v1, v2) / (n1 * n2))


def _principal_angle_cosines(W1, W2):
    """
    Return principal-angle cosines between two subspaces.

    W1: [d, k1] orthonormal basis
    W2: [d, k2] orthonormal basis

    Returns:
        s: singular values of W1^T W2, sorted descending
           These are cos(theta_i), always in [0,1].
    """
    W1 = np.asarray(W1, dtype=np.float32)
    W2 = np.asarray(W2, dtype=np.float32)

    if W1.ndim != 2 or W2.ndim != 2:
        raise ValueError("W1 and W2 must be 2D arrays of shape [d, k].")

    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.array([], dtype=np.float32)

    M = W1.T @ W2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s.astype(np.float32)


def _pairwise_coherence(W1, W2):
    """
    Max over per-basis-vector mean abs-cosines between two orthonormal bases.

    For each column v_i in W1: compute mean(|v_i · u_j| for j in W2).
    For each column u_j in W2: compute mean(|u_j · v_i| for i in W1).
    Returns the max across all these means, or nan if either basis is empty.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2) absolute cosines between all basis vector pairs
    row_means = M.mean(axis=1)  # mean cosine of each W1 vec with all W2 vecs
    col_means = M.mean(axis=0)  # mean cosine of each W2 vec with all W1 vecs
    return float(np.max(np.concatenate([row_means, col_means])))


def _pairwise_coherence_sum(W1, W2):
    """
    Max over per-basis-vector summed abs-cosines between two orthonormal bases.

    Like _pairwise_coherence but uses sum instead of mean, so it scales with
    subspace dimensionality in addition to alignment strength.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2) absolute cosines between all basis vector pairs
    row_sums = M.sum(axis=1)  # sum cosine of each W1 vec with all W2 vecs
    col_sums = M.sum(axis=0)  # sum cosine of each W2 vec with all W1 vecs
    return float(np.max(np.concatenate([row_sums, col_sums])))


def _pairwise_coherence_full_set_mean(W1, W2):
    """
    Max over per-basis-vector 'full-set mean' abs-cosines.

    For each v_i in W1: sum(|v_i·u_j| for j in W2) / (k1+k2-1).
    The denominator treats the k1-1 within-subspace pairs as 0 (they are,
    by SVD orthonormality), so this is the true mean over all k1+k2-1 other
    basis vectors in the combined set.  Same denominator for both sides.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    k1, k2 = W1.shape[1], W2.shape[1]
    M = np.abs(W1.T @ W2)  # (k1, k2)
    denom = k1 + k2 - 1
    row_scores = M.sum(axis=1) / denom
    col_scores = M.sum(axis=0) / denom
    return float(np.max(np.concatenate([row_scores, col_scores])))


def _pairwise_coherence_weighted(W1, W2, sv1, sv2, weight_mode="sv"):
    """
    Like _pairwise_coherence but each cosine |v_i · u_j| is weighted by the
    singular value of the target vector (sv2[j] when W1 is source, sv1[i] when W2 is source).
    weight_mode: "sv" normalised by sum(sv); "sv_squared" normalised by sum(sv²);
                 "none": direct sv-weighted sum, no normalisation.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2)
    sv1 = np.asarray(sv1, dtype=np.float64)
    sv2 = np.asarray(sv2, dtype=np.float64)
    if weight_mode == "none":
        row_means = M @ sv2
        col_means = sv1 @ M
    else:
        w1 = sv1 ** 2 if weight_mode == "sv_squared" else sv1
        w2 = sv2 ** 2 if weight_mode == "sv_squared" else sv2
        s1 = w1.sum(); s2 = w2.sum()
        row_means = (M @ w2) / s2 if s2 > 0 else M.mean(axis=1)
        col_means = (w1 @ M) / s1 if s1 > 0 else M.mean(axis=0)
    return float(np.max(np.concatenate([row_means, col_means])))


def _pairwise_coherence_L2(W1, W2):
    """
    Like _pairwise_coherence but aggregate per basis vector with sqrt(mean(cosine^4))
    (i.e. L2 norm of the squared cosines, normalised by count).
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2)
    M4 = M ** 4
    row_means = np.sqrt(M4.mean(axis=1))
    col_means = np.sqrt(M4.mean(axis=0))
    return float(np.max(np.concatenate([row_means, col_means])))


# =========================================================
# Mean-vector geometry
# =========================================================

def compute_mean_geometry_features(mu1, mu2):
    """
    Compute mean-vector angular metrics.

    Returns:
      {
        "mean_mean_cosine": ...,
        "mean_mean_ortho": ...,         # 1 - abs(cos)
        "mean_mean_signed_ortho": ...,  # 1 - cos
      }
    """
    cos = _cosine_similarity(mu1, mu2)

    if np.isnan(cos):
        return {
            "mean_mean_cosine": np.nan,
            "mean_mean_ortho": np.nan,
            "mean_mean_signed_ortho": np.nan,
        }

    return {
        "mean_mean_cosine": float(cos),
        "mean_mean_ortho": float(1.0 - abs(cos)),
        "mean_mean_signed_ortho": float(1.0 - cos),
    }


# =========================================================
# Mean-to-subspace cross geometry
# =========================================================

def compute_mean_subspace_features(mu, W, prefix, eps=1e-12):
    """
    Compute vector-to-subspace interaction features.

    mu: [d]
    W : [d, k] orthonormal basis

    Returns:
      {
        f"{prefix}_proj_norm": ...,
        f"{prefix}_proj_frac": ...,
        f"{prefix}_ortho": ...,
        f"{prefix}_coords_mean": ...,
        f"{prefix}_coords_abs_mean": ...,
        f"{prefix}_coords_max": ...,
        f"{prefix}_coords_min": ...,
      }
    """
    mu = np.asarray(mu, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)

    mu_norm = np.linalg.norm(mu)
    if mu_norm < eps or W.ndim != 2 or W.shape[1] == 0:
        return {
            f"{prefix}_proj_norm": np.nan,
            f"{prefix}_proj_frac": np.nan,
            f"{prefix}_ortho": np.nan,
            f"{prefix}_coords_mean": np.nan,
            f"{prefix}_coords_abs_mean": np.nan,
            f"{prefix}_coords_max": np.nan,
            f"{prefix}_coords_min": np.nan,
        }

    coords = W.T @ mu
    proj_norm = float(np.linalg.norm(coords))
    proj_frac = float(proj_norm / max(mu_norm, eps))

    return {
        f"{prefix}_proj_norm": proj_norm,
        f"{prefix}_proj_frac": proj_frac,
        f"{prefix}_ortho": float(1.0 - proj_frac),
        f"{prefix}_coords_mean": float(np.mean(coords)),
        f"{prefix}_coords_abs_mean": float(np.mean(np.abs(coords))),
        f"{prefix}_coords_max": float(np.max(coords)),
        f"{prefix}_coords_min": float(np.min(coords)),
    }


# =========================================================
# Basis-dependent signed alignment
# =========================================================

def compute_signed_basis_alignment_features(W1, W2):
    """
    Basis-dependent signed alignment summaries.

    IMPORTANT:
    - These are NOT subspace-invariant.
    - They depend on the particular orthonormal bases returned by your SVD/PCA.
    - So treat them as exploratory features, not canonical principal-angle geometry.

    Returns:
      {
        "signed_basis_diag_mean": ...,
        "signed_basis_diag_min": ...,
        "signed_basis_diag_max": ...,
        "signed_basis_diag_abs_mean": ...,
      }
    """
    W1 = np.asarray(W1, dtype=np.float32)
    W2 = np.asarray(W2, dtype=np.float32)

    if (
        W1.ndim != 2 or W2.ndim != 2
        or W1.shape[1] == 0 or W2.shape[1] == 0
    ):
        return {
            "signed_basis_diag_mean": np.nan,
            "signed_basis_diag_min": np.nan,
            "signed_basis_diag_max": np.nan,
            "signed_basis_diag_abs_mean": np.nan,
        }

    k = min(W1.shape[1], W2.shape[1])
    M = W1[:, :k].T @ W2[:, :k]
    diag = np.diag(M)

    return {
        "signed_basis_diag_mean": float(np.mean(diag)),
        "signed_basis_diag_min": float(np.min(diag)),
        "signed_basis_diag_max": float(np.max(diag)),
        "signed_basis_diag_abs_mean": float(np.mean(np.abs(diag))),
    }


# =========================================================
# Subspace-subspace geometry
# =========================================================

import numpy as np


def _safe_normalize_svd_weights(svd_values, target_len):
    """
    Convert SVD singular values into normalized variance weights.

    If X = U S V^T, then variance explained by component i
    is proportional to S_i^2.

    So we use:
        weights_i = S_i^2 / sum_j S_j^2
    """
    if svd_values is None:
        return None

    svd_values = np.asarray(svd_values, dtype=float).reshape(-1)

    if len(svd_values) == 0:
        return None

    svd_values = svd_values[:target_len]

    if len(svd_values) < target_len:
        pad = np.zeros(target_len - len(svd_values), dtype=float)
        svd_values = np.concatenate([svd_values, pad])

    svd_values = np.where(np.isfinite(svd_values), svd_values, 0.0)
    svd_values = np.maximum(svd_values, 0.0)

    weights = svd_values ** 2
    total = weights.sum()

    if total <= 0:
        return None

    return weights / total


def _svd_weighted_basis_to_subspace_ortho(W_src, W_tgt, svd_values_src=None):
    """
    SVD-weighted orthogonality from source basis directions to target subspace.

    For each source SVD direction w_i, compute:

        overlap_i = || W_tgt^T w_i ||

    Then:

        ortho_i = 1 - overlap_i

    Weighted by S_i^2, because S_i^2 is proportional to variance explained.
    """
    if W_src is None or W_tgt is None:
        return np.nan

    W_src = np.asarray(W_src, dtype=float)
    W_tgt = np.asarray(W_tgt, dtype=float)

    if W_src.ndim != 2 or W_tgt.ndim != 2:
        return np.nan

    if W_src.shape[0] != W_tgt.shape[0]:
        raise ValueError(
            f"W_src and W_tgt should both have shape (d, k). "
            f"Got W_src.shape={W_src.shape}, W_tgt.shape={W_tgt.shape}."
        )

    k_src = W_src.shape[1]

    if k_src == 0 or W_tgt.shape[1] == 0:
        return np.nan

    weights = _safe_normalize_svd_weights(svd_values_src, k_src)

    if weights is None:
        weights = np.ones(k_src, dtype=float) / k_src

    overlaps = np.linalg.norm(W_tgt.T @ W_src, axis=0)
    overlaps = np.clip(overlaps, 0.0, 1.0)

    ortho = 1.0 - overlaps

    return float(np.sum(weights * ortho))


def compute_subspace_geometry_features(
    W1,
    W2,
    *,
    svd_values1=None,
    svd_values2=None,
):
    """
    Compute subspace-subspace metrics.

    W1, W2:
        Orthonormal SVD bases with shape (d, k).

    svd_values1, svd_values2:
        Singular values associated with the columns of W1 and W2.

    Main weighted metric:
        sub_sub_mean_weighted_angle_ortho

    This uses S_i^2 weights, because singular value squared corresponds
    to variance explained by each SVD direction.
    """
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)

    base_nan = {
        "sub_sub_min_angle_ortho": np.nan,
        "sub_sub_max_angle_ortho": np.nan,
        "sub_sub_mean_angle_ortho": np.nan,
        "sub_sub_principal_min_angle_ortho": np.nan,
        "sub_sub_principal_max_angle_ortho": np.nan,
        "sub_sub_principal_mean_angle_ortho": np.nan,
        "sub_sub_fro_ortho": np.nan,
        "sub_sub_mean_weighted_angle_ortho": np.nan,
        "sub_sub_spectral_norm": np.nan,
        "sub_sub_one_minus_spectral_norm": np.nan,
        "sub_sub_one_minus_spectral_norm_unnormalized": np.nan,
    }

    if W1.ndim != 2 or W2.ndim != 2:
        return base_nan

    if W1.shape[0] != W2.shape[0]:
        raise ValueError(
            f"W1 and W2 should both have shape (d, k). "
            f"Got W1.shape={W1.shape}, W2.shape={W2.shape}."
        )

    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return base_nan

    s = _principal_angle_cosines(W1, W2)

    if len(s) == 0:
        return base_nan

    s = np.asarray(s, dtype=float)
    s = np.clip(s, 0.0, 1.0)

    max_cos = float(np.max(s))
    min_cos = float(np.min(s))
    fro_sq_mean = float(np.mean(s ** 2))

    # Corrected mean: sum(|W1^T W2|) over all k1*k2 cross pairs,
    # denominator = SUM*(SUM-1) counting within-subspace pairs (similarity=0).
    k1, k2 = W1.shape[1], W2.shape[1]
    SUM = k1 + k2
    M_cross = np.abs(W1.T @ W2)  # (k1, k2)
    mean_sim_corrected = float(M_cross.sum() / (SUM * (SUM - 1))) if SUM >= 2 else np.nan

    weighted_12 = _svd_weighted_basis_to_subspace_ortho(
        W_src=W1,
        W_tgt=W2,
        svd_values_src=svd_values1,
    )

    weighted_21 = _svd_weighted_basis_to_subspace_ortho(
        W_src=W2,
        W_tgt=W1,
        svd_values_src=svd_values2,
    )

    if svd_values1 is not None and svd_values2 is not None:
        weighted_ortho = float(np.nanmean([weighted_12, weighted_21]))
    elif svd_values1 is not None:
        weighted_ortho = weighted_12
    elif svd_values2 is not None:
        weighted_ortho = weighted_21
    else:
        # No SVD values supplied: both directions use uniform weights (already the default
        # in _svd_weighted_basis_to_subspace_ortho), so average the two directions.
        weighted_ortho = float(np.nanmean([weighted_12, weighted_21]))

    # Unnormalized spectral norm: scale orthonormal bases by their singular values,
    # then take spectral norm directly. σ_max(diag(s1) W1^T W2 diag(s2))
    if svd_values1 is not None and svd_values2 is not None:
        s1 = np.asarray(svd_values1, dtype=float)
        s2 = np.asarray(svd_values2, dtype=float)
        s1 = s1[:W1.shape[1]]
        s2 = s2[:W2.shape[1]]
        M_unnorm = (W1 * s1).T @ (W2 * s2)
        sv_unnorm = np.linalg.svd(M_unnorm, compute_uv=False)
        one_minus_spectral_norm_unnorm = float(1.0 - sv_unnorm[0]) if len(sv_unnorm) > 0 else np.nan
    else:
        one_minus_spectral_norm_unnorm = np.nan

    return {
        "sub_sub_min_angle_ortho": float(1.0 - float(M_cross.max())),
        "sub_sub_max_angle_ortho": float(1.0 - float(M_cross.min())),
        "sub_sub_mean_angle_ortho": float(1.0 - mean_sim_corrected),
        "sub_sub_principal_min_angle_ortho": float(1.0 - max_cos),
        "sub_sub_principal_max_angle_ortho": float(1.0 - min_cos),
        "sub_sub_principal_mean_angle_ortho": float(1.0 - float(np.mean(s))),
        "sub_sub_fro_ortho": float(1.0 - fro_sq_mean),
        "sub_sub_mean_weighted_angle_ortho": weighted_ortho,
        "sub_sub_spectral_norm": float(max_cos),
        "sub_sub_one_minus_spectral_norm": float(1.0 - max_cos),
        "sub_sub_one_minus_spectral_norm_unnormalized": one_minus_spectral_norm_unnorm,
    }

# =========================================================
# Task-level geometry from runs
# =========================================================

def compute_tasklevel_geometry_from_runs(
    eligible,
    runs,   # unused: get_hop_vector reads the global `runs` directly; kept for call-site compatibility
    layer,
    *,
    subspace_var_prop=0.90,
    subspace_center=False,
    min_examples_per_task=2,
    task_field="task",
    include_cross_features=True,
    include_signed_basis_features=True,
    sv_weight_mode="sv",
):
    """
    Compute task-level geometry for one layer.

    q1_vecs/q2_vecs come from get_hop_vector(), the single shared accessor
    that applies (or skips) cluster mean-centering per the module-level
    CLUSTER_MEAN_CENTER flag — the same one used by build_example_subspace_scores
    and the example-example experiment, so the resulting task subspaces are
    centered consistently with the example vectors they get compared against.
    q1_vecs (qx) and q2_vecs (gfx) are centered independently and can use
    different clusters, since a composite task's two hops can land in
    different clusters.
    """
    task_to_examples = {}
    for ex in eligible:
        task = ex[task_field]
        task_to_examples.setdefault(task, []).append(ex)

    task_to_examples = {
        task: exs for task, exs in task_to_examples.items()
        if len(exs) >= min_examples_per_task
    }

    rows = []
    task_reps = {}

    for task, exs in tqdm(task_to_examples.items(), desc="Computing geometry"):
        task_mean_acc = float(np.mean([ex["qx_gfx_acc"] for ex in exs]))
        task_n = len(exs)
        L = layer

        q1_vecs = np.stack([get_hop_vector(ex, L, "q1") for ex in exs], axis=0)
        q2_vecs = np.stack([get_hop_vector(ex, L, "q2") for ex in exs], axis=0)

        rep_q1 = build_mean_and_subspace(
            q1_vecs,
            var_prop=subspace_var_prop,
            center=subspace_center,
        )
        rep_q2 = build_mean_and_subspace(
            q2_vecs,
            var_prop=subspace_var_prop,
            center=subspace_center,
        )

        task_reps[task] = {
            "q1": rep_q1,
            "q2": rep_q2,
        }

        rows.append({
            "task": task,
            "layer": int(L),
            "n_examples": int(task_n),
            "task_mean_acc": float(task_mean_acc),
            "q1_rank": int(rep_q1["rank"]),
            "q2_rank": int(rep_q2["rank"]),
            "rank_diff": int(abs(rep_q1["rank"] - rep_q2["rank"])),
            "rank_min": int(min(rep_q1["rank"], rep_q2["rank"])),
            "rank_max": int(max(rep_q1["rank"], rep_q2["rank"])),
            "q1_mean_norm": float(np.linalg.norm(rep_q1["mean"])),
            "q2_mean_norm": float(np.linalg.norm(rep_q2["mean"])),
            "mean_norm_diff": float(abs(np.linalg.norm(rep_q1["mean"]) - np.linalg.norm(rep_q2["mean"]))),
            "mean_norm_ratio": float(
                np.linalg.norm(rep_q1["mean"]) / np.linalg.norm(rep_q2["mean"])
            ) if np.linalg.norm(rep_q2["mean"]) > 0 else np.nan,
        })

    # ── subspace_subspace_angle_by_cumulative_coherence ────────────────────────────────
    # Per-task: coherence between that task's own q1 and q2 subspaces.
    # _pairwise_coherence(q1, q2) already returns the max over W1+W2 per-vector mean cosines.
    for row in rows:
        t = row["task"]
        W_q1 = task_reps[t]["q1"]["basis"]
        W_q2 = task_reps[t]["q2"]["basis"]
        if W_q1.shape[1] > 0 and W_q2.shape[1] > 0:
            coh = _pairwise_coherence(W_q1, W_q2)
        else:
            coh = np.nan
        row["subspace_subspace_angle_by_cumulative_coherence"] = 1.0 - abs(coh) if np.isfinite(coh) else np.nan
        task_reps[t]["subspace_subspace_angle_by_cumulative_coherence"] = row["subspace_subspace_angle_by_cumulative_coherence"]

    return rows, task_reps


import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr



# =========================================================
# Helpers for shuffled-y baselines
# =========================================================

def permute_y_within_groups(y, group_labels, random_seed=0):
    rng = np.random.default_rng(random_seed)
    y = np.asarray(y, dtype=np.float64)
    group_labels = np.asarray(group_labels, dtype=object)

    y_perm = y.copy()
    for g in np.unique(group_labels):
        idx = np.where(group_labels == g)[0]
        vals = y_perm[idx].copy()
        rng.shuffle(vals)
        y_perm[idx] = vals
    return y_perm



# =========================================================
# Random vector / random subspace helpers
# =========================================================

def _one_minus_abs_cos(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    c = float(np.dot(a, b) / (na * nb))
    c = np.clip(c, -1.0, 1.0)
    return float(1.0 - abs(c))


def _random_unit_vector(d, rng):
    v = rng.standard_normal(d)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v[0] = 1.0
        n = 1.0
    return (v / n).astype(np.float32)


def _random_vector_matched_norm(ref_vec, rng):
    ref_vec = np.asarray(ref_vec, dtype=np.float32)
    d = ref_vec.shape[0]
    target_norm = float(np.linalg.norm(ref_vec))
    if target_norm < 1e-12:
        return np.zeros(d, dtype=np.float32)
    return (_random_unit_vector(d, rng) * target_norm).astype(np.float32)


def _random_orthonormal_basis(d, k, rng):
    if k <= 0:
        return np.zeros((d, 0), dtype=np.float32)
    M = rng.standard_normal((d, k))
    Q, R = np.linalg.qr(M)
    return Q[:, :k].astype(np.float32)


def _random_rep_like(rep, rng):
    """
    Random mean + random basis with matched norm/rank/dim.
    """
    mu = np.asarray(rep["mean"], dtype=np.float32)
    W = np.asarray(rep["basis"], dtype=np.float32)
    d = mu.shape[0]
    k = 0 if W.ndim != 2 else W.shape[1]

    return {
        "mean": _random_vector_matched_norm(mu, rng),
        "basis": _random_orthonormal_basis(d, k, rng),
        "rank": int(k),
        "n_vecs": int(rep.get("n_vecs", rep.get("n", 0))),
    }


# =========================================================
# Plotting helpers: allow plotting only random line
# =========================================================

def plot_cumulative_curve_from_scores(
    x,
    y,
    *,
    quantile_step=0.05,
    add_random_baseline=True,
    n_random_trials=20,
    random_seed=0,
    title=None,
    save_path=None,
    show=True,
    show_ortho_range=False,
    plot_real=True,
    external_curve_y=None,
    external_curve_label=None,
    external_curve_auc=None,
):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    xs01_real, cum_y_real, counts = cumulative_mean_from_scores(x, y, quantile_step=quantile_step)
    auc_real = auc_of_curve(xs01_real, cum_y_real)

    ylabel = "Cumulative mean accuracy"

    rand_xs01, rand_cum_y, rand_auc, rand_trial_aucs = (None, None, None, None)
    if add_random_baseline:
        rand_xs01, rand_cum_y, rand_auc, rand_trial_aucs = mean_random_cumulative_from_shuffle(
            y,
            quantile_step=quantile_step,
            n_trials=n_random_trials,
            random_seed=random_seed,
        )

    ext_xs01, ext_cum_y, ext_auc = (None, None, None)
    if external_curve_y is not None:
        external_curve_y = np.asarray(external_curve_y, dtype=np.float32)
        ext_xs01, ext_cum_y, _ = cumulative_mean_from_scores(x, external_curve_y, quantile_step=quantile_step)
        ext_auc = auc_of_curve(ext_xs01, ext_cum_y) if external_curve_auc is None else external_curve_auc

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    if plot_real:
        ax.plot(xs01_real, cum_y_real, marker="o", label=f"real (AUC={auc_real:.4f})")

    if external_curve_y is not None and ext_xs01 is not None and len(ext_xs01) > 0:
        lbl = external_curve_label or f"external (AUC={ext_auc:.4f})"
        ax.plot(ext_xs01, ext_cum_y, marker="o", linestyle="--", label=lbl)

    elif add_random_baseline and rand_xs01 is not None and len(rand_xs01) > 0:
        ax.plot(
            rand_xs01,
            rand_cum_y,
            marker="o",
            linestyle="--",
            label=f"shuffle baseline (AUC={rand_auc:.4f})",
        )

    # ---- FIXED 4 TICKS ----
    xticks = np.linspace(xs01_real.min(), xs01_real.max(), 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2f}" for t in xticks])

    ymin, ymax = ax.get_ylim()
    yticks = np.linspace(ymin, ymax, 4)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks])

    ax.set_xlabel("Lowest-orthogonality fraction included")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    else:
        if plot_real:
            ax.set_title(f"Cumulative {y_mode}\nAUC={auc_real:.4f}, n={len(x)}")
        else:
            ax.set_title(f"Cumulative {y_mode} baseline\nn={len(x)}")

    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        _json_path = os.path.splitext(save_path)[0] + ".json"
        with open(_json_path, "w") as _jf:
            json.dump({
                "xs01_real": xs01_real.tolist(),
                "cum_y_real": cum_y_real.tolist(),
                "auc_real": float(auc_real) if np.isfinite(auc_real) else None,
                "counts": counts.tolist(),
                "n": int(len(x)),
            }, _jf, indent=2)

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "xs01_real": xs01_real,
        "cum_y_real": cum_y_real,
        "auc_real": auc_real,
        "counts": counts,
        "n": len(x),
    }
def plot_noncumulative_curve_from_scores(
    x,
    y,
    *,
    quantile_step=0.05,
    add_random_baseline=True,
    n_random_trials=20,
    random_seed=0,
    title=None,
    save_path=None,
    show=True,
    show_ortho_range=False,
    plot_real=True,
    external_curve_y=None,
    external_curve_label=None,
    external_curve_auc=None,
    y_mode="accuracy",
):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    xs01_real, bin_means_real, counts = noncumulative_mean_from_scores(
        x, y, quantile_step=quantile_step
    )

    ylabel = (
        "Mean accuracy"
        if y_mode == "accuracy"
        else "Mean correct-token probability"
    )

    r_real, p_real = pearsonr(x, y) if len(x) > 1 else (np.nan, np.nan)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    if plot_real:
        label = f"real (r={r_real:.4f}, p={p_real:.2e})" if np.isfinite(r_real) else "real"
        ax.plot(xs01_real, bin_means_real, marker="o", label=label)

    # ---- FIXED 4 TICKS ----
    xticks = np.linspace(xs01_real.min(), xs01_real.max(), 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2f}" for t in xticks])

    ymin, ymax = ax.get_ylim()
    yticks = np.linspace(ymin, ymax, 4)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks])

    ax.set_xlabel("Orthogonality fraction")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Noncumulative {y_mode}\nn={len(x)}")

    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        _json_path = os.path.splitext(save_path)[0] + ".json"
        with open(_json_path, "w") as _jf:
            json.dump({
                "xs01_real": xs01_real.tolist(),
                "bin_means_real": bin_means_real.tolist(),
                "counts": counts.tolist(),
                "pearson_r_real": float(r_real) if np.isfinite(r_real) else None,
                "pearson_p_real": float(p_real) if np.isfinite(p_real) else None,
                "n": int(len(x)),
            }, _jf, indent=2)

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "xs01_real": xs01_real,
        "bin_means_real": bin_means_real,
        "counts": counts,
        "pearson_r_real": r_real,
        "pearson_p_real": p_real,
        "n": len(x),
    }


def select_best_layer_by_train_auc_and_plot_test_for_datasets(
    dataset_names,
    xs_by_layer,
    y_acc_by_layer,
    eligible,
    *,
    out_dir,
    dataset_field="task",
    train_ratio=0.1,
    random_seed=0,
    quantile_step=0.05,
    subspace_center=False,
    sweep_subspace_var_prop=(0.85, 0.9, 0.95, 0.99),
    n_random_trials=20,
    show=True,
    plot_dataset=False,
    dataset_own_best_layer=True,
    fixed_dataset_for_QFxGFx=None,
    show_ortho_range=False,
    fixed_layer_to_plot=None,
    sv_weight_mode="sv",
    y_mode_for_curves="accuracy",
    plot_every_layer_test=False,
):

    ensure_dir(out_dir)

    selected_idx = _get_selected_indices(
        eligible,
        dataset_names=dataset_names,
        dataset_field=dataset_field,
    )
    if len(selected_idx) == 0:
        raise ValueError("No examples found for the selected datasets.")

    train_idx, test_idx = _split_train_test(
        selected_idx,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    n_layers = len(xs_by_layer)
    train_auc_rows = []
    best_layer = None
    best_auc = np.inf
    if fixed_layer_to_plot is None:
        for layer in range(n_layers):
            x_train, y_train = _get_xy_for_layer_and_indices(
                layer, train_idx, xs_by_layer, y_acc_by_layer,
            )
            if np.std(x_train) < 1e-8:
                continue
            
            xs01, cum_y, _ = cumulative_mean_from_scores(
                x_train, y_train, quantile_step=quantile_step
            )
            auc = auc_of_curve(xs01, cum_y)
    
            train_auc_rows.append({
                "layer": layer,
                "n_train": len(train_idx),
                "train_auc": auc,
            })
    
            if np.isfinite(auc) and auc < best_auc:
                best_auc = auc
                best_layer = layer
    
        if best_layer is None:
            raise ValueError("Could not select a best layer from train AUC.")

        print(f"Selected best layer by lowest TRAIN AUC: layer={best_layer}, train_auc={best_auc:.6f}")
        print(f"Restricted datasets: {dataset_names}")
        print(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")
    else:
        best_layer = fixed_layer_to_plot

    prefix = (
        f"datasets_{len(dataset_names)}_"
        f"train{train_ratio}_seed{random_seed}_"
        f"bestlayer{best_layer:02d}"
    )
    all_layers_prefix = (
        f"datasets_{len(dataset_names)}_"
        f"train{train_ratio}_seed{random_seed}_"

    )

    if plot_every_layer_test:
        all_layers_test_dir = os.path.join(out_dir, "ex_ex_all_layers_test")
        ensure_dir(all_layers_test_dir)
        pr_auc_summary_all_layers = {}
        for layer in range(n_layers):
            x_layer, y_layer = _get_xy_for_layer_and_indices(
                layer, test_idx, xs_by_layer, y_acc_by_layer,
            )
            if np.std(x_layer) < 1e-8:
                continue
            layer_cumulative_path = os.path.join(
                all_layers_test_dir,
                f"{all_layers_prefix}layer{layer:02d}_test_cumulative_{y_mode_for_curves}.svg",
            )
            plot_cumulative_curve_for_indices(
                layer=layer,
                indices=test_idx,
                xs_by_layer=xs_by_layer,
                y_acc_by_layer=y_acc_by_layer,
                quantile_step=quantile_step,
                n_random_trials=n_random_trials,
                random_seed=random_seed + 101 * layer,
                title=None,
                save_path=layer_cumulative_path,
                show=show,
                show_ortho_range=show_ortho_range,
            )

            layer_pr_stats = compute_failure_pr_auc(
                x_layer, y_layer, seed=random_seed + 101 * layer,
            )
            layer_pr_curve_path = os.path.join(
                all_layers_test_dir,
                f"{all_layers_prefix}layer{layer:02d}_test_pr_curve.svg",
            )
            plot_failure_pr_curve(
                layer_pr_stats,
                save_path=layer_pr_curve_path,
                show=show,
                title=f"Layer {layer} failure PR curve (test)",
            )
            pr_auc_summary_all_layers[layer] = {
                "pr_auc": layer_pr_stats["pr_auc"],
                "normalized_pr_auc": layer_pr_stats["normalized_pr_auc"],
                "failure_rate": layer_pr_stats["failure_rate"],
                "perm_p": layer_pr_stats["perm_p"],
                "pr_auc_improvement_ci_lo": layer_pr_stats["pr_auc_improvement_ci_lo"],
                "pr_auc_improvement_ci_hi": layer_pr_stats["pr_auc_improvement_ci_hi"],
                "normalized_pr_auc_ci_lo": layer_pr_stats["normalized_pr_auc_ci_lo"],
                "normalized_pr_auc_ci_hi": layer_pr_stats["normalized_pr_auc_ci_hi"],
                "n": int(len(x_layer)),
            }

        pr_auc_summary_path = os.path.join(all_layers_test_dir, f"{all_layers_prefix}pr_auc_summary.json")
        with open(pr_auc_summary_path, "w") as _f:
            json.dump(pr_auc_summary_all_layers, _f, indent=2)

    cumulative_path = os.path.join(out_dir, f"{prefix}_test_cumulative_{y_mode_for_curves}.svg")
    noncumulative_path = os.path.join(out_dir, f"{prefix}_test_noncumulative_{y_mode_for_curves}.svg")
    scatter_path = os.path.join(out_dir, f"{prefix}_test_meanortho_vs_meanacc.svg")
    error_recall_path = os.path.join(out_dir, f"{prefix}_test_error_recall.svg")


    test_cum = plot_cumulative_curve_for_indices(
        layer=best_layer,
        indices=test_idx,
        xs_by_layer=xs_by_layer,
        y_acc_by_layer=y_acc_by_layer,
        quantile_step=quantile_step,

        n_random_trials=n_random_trials,
        random_seed=random_seed + 101 * best_layer,
        title=None,
        save_path=cumulative_path,
        show=show,
        show_ortho_range=show_ortho_range,
    )

    test_error_recall = plot_error_recall_curve_for_indices(
        layer=best_layer,
        indices=test_idx,
        xs_by_layer=xs_by_layer,
        y_acc_by_layer=y_acc_by_layer,
        title=None,
        save_path=error_recall_path,
        show=show,
        random_seed=random_seed + 101 * best_layer,
    )

    test_noncum = plot_noncumulative_curve_for_indices(
        layer=best_layer,
        indices=test_idx,
        xs_by_layer=xs_by_layer,
        y_acc_by_layer=y_acc_by_layer,
        
        quantile_step=quantile_step,

        title=None,
        save_path=noncumulative_path,
        show=show,
        show_ortho_range=show_ortho_range,
    )

    test_scatter = plot_mean_ortho_against_mean_acc_for_indices(
        layer=best_layer,
        indices=test_idx,
        xs_by_layer=xs_by_layer,
        y_acc_by_layer=y_acc_by_layer,
        eligible=eligible,
        dataset_names=dataset_names,
        dataset_field=dataset_field,
        annotate=True,
        title=None,
        save_path=scatter_path,
        show=show,
    )
    dist_path = os.path.join(out_dir, f"{prefix}_test_orthogonality_distribution.svg")

    test_dist = plot_orthogonality_distribution_for_indices(
        layer=best_layer,
        indices=test_idx,
        xs_by_layer=xs_by_layer,
        bins=40,
        title=(
            f"TEST orthogonality distribution | best train layer={best_layer}\n"
            f"train_ratio={train_ratio}, n_test={len(test_idx)}"
        ),
        save_path=dist_path,
        show=show,
    )

    pr_auc_path = os.path.join(out_dir, f"{prefix}_test_pr_curve.svg")
    x_best_test, y_best_test = _get_xy_for_layer_and_indices(
        best_layer, test_idx, xs_by_layer, y_acc_by_layer,
    )
    test_pr_stats = compute_failure_pr_auc(
        x_best_test, y_best_test, seed=random_seed + 101 * best_layer,
    )
    plot_failure_pr_curve(
        test_pr_stats,
        save_path=pr_auc_path,
        show=show,
        title=f"TEST failure PR curve | best train layer={best_layer}",
    )






    eligible_test = [eligible[i] for i in test_idx]
    eligible_dev = [eligible[i] for i in train_idx]

    from collections import Counter
    total_task_counts = Counter(ex.get(dataset_field, "UNKNOWN") for ex in eligible_test + eligible_dev)
    kept_tasks = {task for task, count in total_task_counts.items() if count >= 5}
    eligible_test = [ex for ex in eligible_test if ex.get(dataset_field, "UNKNOWN") in kept_tasks]
    eligible_dev = [ex for ex in eligible_dev if ex.get(dataset_field, "UNKNOWN") in kept_tasks]

    test_task_counts = Counter(ex.get(dataset_field, "UNKNOWN") for ex in eligible_test)
    dev_task_counts = Counter(ex.get(dataset_field, "UNKNOWN") for ex in eligible_dev)
    print(f"Tasks with <5 total examples excluded: {sorted(set(total_task_counts) - kept_tasks)}")
    print("Examples per task (test):")
    for task, count in sorted(test_task_counts.items()):
        print(f"  {task}: {count}")
    print("Examples per task (dev):")
    for task, count in sorted(dev_task_counts.items()):
        print(f"  {task}: {count}")

    # Normalise the var_prop sweep to a list
    if isinstance(sweep_subspace_var_prop, (int, float)):
        sweep_subspace_var_prop = [sweep_subspace_var_prop]
    sweep_subspace_var_prop = list(sweep_subspace_var_prop)

    x_keys = _ss_x_keys = [
        "subspace_subspace_angle_by_cumulative_coherence",
    ]

    # =========================================================
    # Dev sweep: find best (var_prop, layer) per metric on dev
    # =========================================================
    print(f"[SubSub] Sweeping var_props={sweep_subspace_var_prop} × {n_layers} layers on dev...")
    _ss_dev_rows_by_vp_layer = {}  # key: (var_prop, layer) -> rows
    for _vp in sweep_subspace_var_prop:
        for _ss_layer in range(n_layers):
            _ss_dev_rows, _ = compute_tasklevel_geometry_from_runs(
                eligible_dev,
                runs,
                layer=_ss_layer,
                subspace_var_prop=_vp,
                subspace_center=subspace_center,
                min_examples_per_task=2,
                task_field=dataset_field,
                sv_weight_mode=sv_weight_mode,
            )
            _ss_dev_rows_by_vp_layer[(_vp, _ss_layer)] = _ss_dev_rows

    # Select best (var_prop, layer) per metric by highest Pearson r on dev
    _ss_best_vp_layer = {}  # metric -> (var_prop, layer) or None
    for _ss_xk in _ss_x_keys:
        _ss_best_key = None
        _ss_best_score = -np.inf
        for _vp in sweep_subspace_var_prop:
            for _ss_layer in range(n_layers):
                _ss_dev_vals = [
                    (r[_ss_xk], r["task_mean_acc"])
                    for r in _ss_dev_rows_by_vp_layer[(_vp, _ss_layer)]
                    if _ss_xk in r and np.isfinite(r[_ss_xk]) and np.isfinite(r["task_mean_acc"])
                ]
                if len(_ss_dev_vals) < 3:
                    continue
                _ss_xs_d = np.array([v[0] for v in _ss_dev_vals])
                _ss_ys_d = np.array([v[1] for v in _ss_dev_vals])
                if np.std(_ss_xs_d) < 1e-12 or np.std(_ss_ys_d) < 1e-12:
                    continue
                _ss_score, _ = pearsonr(_ss_xs_d, _ss_ys_d)
                if _ss_score > _ss_best_score:
                    _ss_best_score = _ss_score
                    _ss_best_key = (_vp, _ss_layer)
        _ss_best_vp_layer[_ss_xk] = _ss_best_key
        if _ss_best_key is not None:
            _vp_b, _L_b = _ss_best_key
            print(f"[SubSub] {_ss_xk}: dev best var_prop={_vp_b}, layer={_L_b}, score={_ss_best_score:.4f}")
        else:
            print(f"[SubSub] {_ss_xk}: no valid (var_prop, layer) found on dev")

    # Most commonly selected var_prop → used for example-subspace plots and fixed-layer plots
    _vp_votes = Counter(k[0] for k in _ss_best_vp_layer.values() if k is not None)
    _best_vp_overall = _vp_votes.most_common(1)[0][0] if _vp_votes else sweep_subspace_var_prop[0]
    print(f"[SubSub] Overall best var_prop (most common across metrics): {_best_vp_overall}")

    # Build task_reps for example-subspace plots (uses cumulative-AUC best_layer + best var_prop)
    rows, task_reps = compute_tasklevel_geometry_from_runs(
        eligible_test,
        runs,
        layer=best_layer,
        subspace_var_prop=_best_vp_overall,
        subspace_center=subspace_center,
        min_examples_per_task=2,
        task_field="task",
        sv_weight_mode=sv_weight_mode,
    )

    # Compute test rows for each unique (var_prop, layer) combo selected across metrics
    _ss_test_rows_by_vp_layer = {}  # key: (var_prop, layer) -> rows
    for _ss_xk in _ss_x_keys:
        _key = _ss_best_vp_layer.get(_ss_xk)
        if _key is None or _key in _ss_test_rows_by_vp_layer:
            continue
        _vp_sel, _layer_sel = _key
        _ss_test_rows, _ = compute_tasklevel_geometry_from_runs(
            eligible_test,
            runs,
            layer=_layer_sel,
            subspace_var_prop=_vp_sel,
            subspace_center=subspace_center,
            min_examples_per_task=2,
            task_field=dataset_field,
            sv_weight_mode=sv_weight_mode,
        )
        _ss_test_rows_by_vp_layer[_key] = _ss_test_rows

    subsub_save_dir = os.path.join(out_dir, "subspace_subspace_dev_layer_selected")
    os.makedirs(subsub_save_dir, exist_ok=True)
    subsub_dev_save_dir = os.path.join(out_dir, "subspace_subspace_dev_layer_selected_dev")
    os.makedirs(subsub_dev_save_dir, exist_ok=True)
    for _ss_xk in _ss_x_keys:
        _key = _ss_best_vp_layer.get(_ss_xk)
        if _key is None:
            print(f"[SubSub] Skipping {_ss_xk}: no valid dev (var_prop, layer) found")
            continue
        _vp_sel, _layer_sel = _key
        _var_suffix = f"_vp{_vp_sel}_layer{_layer_sel}"
        plot_tasklevel_feature_vs_accuracy(
            _ss_dev_rows_by_vp_layer[_key],
            x_keys=[_ss_xk],
            y_key="task_mean_acc",
            save_dir=subsub_dev_save_dir,
            show=show,
            point_size=180,
            title_suffix=f" | dev-best vp={_vp_sel} L={_layer_sel} (dev)",
            file_suffix=_var_suffix,
        )
        plot_tasklevel_feature_vs_accuracy(
            _ss_test_rows_by_vp_layer[_key],
            x_keys=[_ss_xk],
            y_key="task_mean_acc",
            save_dir=subsub_save_dir,
            show=show,
            point_size=180,
            title_suffix=f" | dev-best vp={_vp_sel} L={_layer_sel}",
            file_suffix=_var_suffix,
        )

    # =========================================================
    # Fixed best layer (enfact) scatter plots — use overall best var_prop
    # =========================================================
    best_layer_fixed = 13
    _fixed_key = (_best_vp_overall, best_layer_fixed)
    if _fixed_key in _ss_test_rows_by_vp_layer:
        _ss_fixed_test_rows = _ss_test_rows_by_vp_layer[_fixed_key]
    else:
        _ss_fixed_test_rows, _ = compute_tasklevel_geometry_from_runs(
            eligible_test,
            runs,
            layer=best_layer_fixed,
            subspace_var_prop=_best_vp_overall,
            subspace_center=subspace_center,
            min_examples_per_task=2,
            task_field=dataset_field,
            sv_weight_mode=sv_weight_mode,
        )

    subsub_fixed_save_dir = os.path.join(out_dir, "subspace_subspace_enfact_best_layer")
    os.makedirs(subsub_fixed_save_dir, exist_ok=True)
    _fixed_var_suffix = f"_vp{_best_vp_overall}"
    for _ss_xk in _ss_x_keys:
        plot_tasklevel_feature_vs_accuracy(
            _ss_fixed_test_rows,
            x_keys=[_ss_xk],
            y_key="task_mean_acc",
            save_dir=subsub_fixed_save_dir,
            show=show,
            point_size=180,
            title_suffix=f" | fixed L={best_layer_fixed} vp={_best_vp_overall}",
            file_suffix=_fixed_var_suffix,
        )

    # =========================================================
    # Example-subspace plots
    # =========================================================
    """
    #TODO: i want to have a q1subspace_q2subspace cumualtive plot, where eeach example just sues its subsapce orthogoanlty,so for two exampels int ehs ame dataset-dataset pair they should have the same orthogoanlti
    #so many values will be constant, 
    #but i still want to do a percentile sorted by these angles and plot the cuulative curve and lok at the auc
    example_subspace_dir = os.path.join(out_dir, "example_subspace")
    os.makedirs(example_subspace_dir, exist_ok=True)
    exsub_subsub = build_example_subspace_subspace_scores(
            eligible_test,
            task_reps=task_reps,
            task_field=dataset_field,
            x_keys=x_keys,
        )
    y_ex_subsub = (
        exsub_subsub["y_acc"]
        if y_mode_for_curves == "accuracy"
        else exsub_subsub["y_acc"]
    )
    for x_key in x_keys:
        x_raw = exsub_subsub.get(x_key, None)
    
        if x_raw is None:
            print(f"[SKIP] missing key: {x_key}")
            continue
    
        x_vals = np.asarray(x_raw, dtype=float)
        y_vals = np.asarray(y_ex_subsub, dtype=float)
    
        print(
            f"[DEBUG] {x_key}: "
            f"x_len={len(x_vals)}, y_len={len(y_vals)}, "
            f"x_finite={np.isfinite(x_vals).sum()}, "
            f"y_finite={np.isfinite(y_vals).sum()}, "
            f"x_nan={np.isnan(x_vals).sum() if len(x_vals) else 0}, "
            f"y_nan={np.isnan(y_vals).sum() if len(y_vals) else 0}"
        )
    
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    
        if valid.sum() == 0:
            print(f"[SKIP] {x_key}: no finite x/y pairs.")
            continue

        q1sub_q2sub_cum_path = os.path.join(
            example_subspace_dir,
            f"{prefix}_q1subspace_q2subspace_test_cumulative_{y_mode_for_curves}_{x_key}.svg",
        )
    
        q1sub_q2sub_noncum_path = os.path.join(
            example_subspace_dir,
            f"{prefix}_q1subspace_q2subspace_test_noncumulative_{y_mode_for_curves}_{x_key}.svg",
        )
    
        test_cum_q1sub_q2sub = plot_cumulative_curve_from_scores(
            exsub_subsub[x_key],
            y_ex_subsub,
            quantile_step=quantile_step,
            n_random_trials=n_random_trials,
            random_seed=random_seed + 707 * best_layer,
            title=f"Q1 subspace vs Q2 subspace: {x_key}",
            save_path=q1sub_q2sub_cum_path,
            show=show,
            show_ortho_range=show_ortho_range,
        )
    
        test_noncum_q1sub_q2sub = plot_noncumulative_curve_from_scores(
            exsub_subsub[x_key],
            y_ex_subsub,
            quantile_step=quantile_step,
            n_random_trials=n_random_trials,
            random_seed=random_seed + 808 * best_layer,
            title=f"Q1 subspace vs Q2 subspace: {x_key}",
            save_path=q1sub_q2sub_noncum_path,
            show=show,
            show_ortho_range=show_ortho_range,
        )
        


    exsub = build_example_subspace_scores(
        eligible_test,
        runs,
        layer=best_layer,
        task_reps=task_reps,
        task_field=dataset_field,   # or "task" if always task
    )

    
    y_ex = exsub["y_acc"]

    q1vec_q2sub_cum_path = os.path.join(
        example_subspace_dir,
        f"{prefix}_q1vec_q2subspace_test_cumulative_{y_mode_for_curves}.svg",
    )
    q1vec_q2sub_noncum_path = os.path.join(
        example_subspace_dir,
        f"{prefix}_q1vec_q2subspace_test_noncumulative_{y_mode_for_curves}.svg",
    )

    q2vec_q1sub_cum_path = os.path.join(
        example_subspace_dir,
        f"{prefix}_q2vec_q1subspace_test_cumulative_{y_mode_for_curves}.svg",
    )
    q2vec_q1sub_noncum_path = os.path.join(
        example_subspace_dir,
        f"{prefix}_q2vec_q1subspace_test_noncumulative_{y_mode_for_curves}.svg",
    )
    

    

    test_cum_q1vec_q2sub = plot_cumulative_curve_from_scores(
        exsub["q1vec_q2_linear_subspace"],
        y_ex,
        quantile_step=quantile_step,
       
        n_random_trials=n_random_trials,
        random_seed=random_seed + 303 * best_layer,
        title="Q1 vec vs Q2 affine subspace",
        save_path=q1vec_q2sub_cum_path,
        show=show,
        show_ortho_range=show_ortho_range,
        )

    test_noncum_q1vec_q2sub = plot_noncumulative_curve_from_scores(
        exsub["q1vec_q2_linear_subspace"],
        y_ex,
        quantile_step=quantile_step,
       
        n_random_trials=n_random_trials,
        random_seed=random_seed + 404 * best_layer,
        title="Q1 vec vs Q2 affine subspace",
        save_path=q1vec_q2sub_noncum_path,
        show=show,
        show_ortho_range=show_ortho_range,
    )
    
    test_cum_q2vec_q1sub = plot_cumulative_curve_from_scores(
        exsub["q2vec_q1_linear_subspace"],
        y_ex,
        quantile_step=quantile_step,
       
        n_random_trials=n_random_trials,
        random_seed=random_seed + 505 * best_layer,
        title="Q2 vec vs Q1 affine subspace",
        save_path=q2vec_q1sub_cum_path,
        show=show,
        show_ortho_range=show_ortho_range,
    )
    
    test_noncum_q2vec_q1sub = plot_noncumulative_curve_from_scores(
        exsub["q2vec_q1_linear_subspace"],
        y_ex,
        quantile_step=quantile_step,
       
        n_random_trials=n_random_trials,
        random_seed=random_seed + 606 * best_layer,
        title="Q2 vec vs Q1 affine subspace",
        save_path=q2vec_q1sub_noncum_path,
        show=show,
        show_ortho_range=show_ortho_range,
    )
    """


    if plot_dataset:
        print("\n[Plotting per-dataset results]")

        for dataset_name in dataset_names:
            print(f"\n--- Dataset: {dataset_name} ---")

            # 1. get indices for THIS dataset only
            ds_idx = _get_selected_indices(
                eligible,
                dataset_names=[dataset_name],
                dataset_field=dataset_field,
            )
            if len(ds_idx) == 0:
                print(f"Skipping {dataset_name} (no data)")
                continue

            # 2. split train/test within dataset
            try:
                ds_train_idx, ds_test_idx = _split_train_test(
                    ds_idx,
                    train_ratio=train_ratio,
                    random_seed=random_seed,
                )
            except ValueError as e:
                continue

            # 3. determine best layer
            if dataset_own_best_layer:
                ds_best_layer = None
                ds_best_auc = np.inf

                for layer in range(n_layers):
                    x_train, y_train = _get_xy_for_layer_and_indices(
                        layer,
                        ds_train_idx,
                        xs_by_layer,
                        y_acc_by_layer,
                                y_mode=y_mode_for_selection,
                    )
                    if np.std(x_train) < 1e-8:
                        continue

                    xs01, cum_y, _ = cumulative_mean_from_scores(
                        x_train, y_train, quantile_step=quantile_step
                    )
                    auc = auc_of_curve(xs01, cum_y)

                    if np.isfinite(auc) and auc < ds_best_auc:
                        ds_best_auc = auc
                        ds_best_layer = layer

                if ds_best_layer is None:
                    print(f"Skipping {dataset_name} (no valid layer)")
                    continue

                print(f"[{dataset_name}] best layer (train): {ds_best_layer}, auc={ds_best_auc:.4f}")

            else:
                ds_best_layer = best_layer
                print(f"[{dataset_name}] using GLOBAL best layer: {ds_best_layer}")

            # 4. define save prefix
            ds_prefix = (
                f"{dataset_name}_train{train_ratio}_seed{random_seed}_"
                f"bestlayer{ds_best_layer:02d}"
            )

            # paths
            ds_cum_path = os.path.join(out_dir, f"{ds_prefix}_test_cumulative.svg")
            ds_noncum_path = os.path.join(out_dir, f"{ds_prefix}_test_noncumulative.svg")
            ds_dist_path = os.path.join(out_dir, f"{ds_prefix}_test_distribution.svg")

            # 5. plot cumulative
            plot_cumulative_curve_for_indices(
                layer=ds_best_layer,
                indices=ds_test_idx,
                xs_by_layer=xs_by_layer,
                y_acc_by_layer=y_acc_by_layer,
                quantile_step=quantile_step,
               
                n_random_trials=n_random_trials,
                random_seed=random_seed + 11 * ds_best_layer,
                title=f"{dataset_name} (cumulative)",
                save_path=ds_cum_path,
                show=show,
            )

            # 6. plot noncumulative
            
            plot_noncumulative_curve_for_indices(
                layer=ds_best_layer,
                indices=ds_test_idx,
                xs_by_layer=xs_by_layer,
                y_acc_by_layer=y_acc_by_layer,
                quantile_step=quantile_step,
              
                n_random_trials=n_random_trials,
                random_seed=random_seed + 22 * ds_best_layer,
                title=f"{dataset_name} (noncumulative)",
                save_path=ds_noncum_path,
                show=show,
            )
            

            # 7. plot distribution
            plot_orthogonality_distribution_for_indices(
                layer=ds_best_layer,
                indices=ds_test_idx,
                xs_by_layer=xs_by_layer,
                bins=40,
                title=f"{dataset_name} orthogonality distribution",
                save_path=ds_dist_path,
                show=show,
            )
            

    return {
        "dataset_names": list(dataset_names),
        "selected_idx": selected_idx,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_auc_rows": train_auc_rows,
        "best_layer": best_layer,
        "best_train_auc": best_auc,
        "test_cumulative": test_cum,
        "test_error_recall": test_error_recall,
        "test_noncumulative": test_noncum,
        "test_scatter": test_scatter,
        "test_pr_auc": {
            "pr_auc": test_pr_stats["pr_auc"],
            "normalized_pr_auc": test_pr_stats["normalized_pr_auc"],
            "failure_rate": test_pr_stats["failure_rate"],
            "perm_p": test_pr_stats["perm_p"],
        },
        "saved_paths": {
            "cumulative": cumulative_path,
            "error_recall": error_recall_path,
            "noncumulative": noncumulative_path,
            "scatter": scatter_path,
            "pr_curve": pr_auc_path,
        },
    }





import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 6),

    # --- match LaTeX / paper ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",

    # --- sizes (tuned for paper) ---
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 10,
})
str_str = ['antonym-french', 'antonym-german', 'antonym-spanish', 
                    'landmark-country-capital', 
                    'park-country-capital', 'person-university-founder', 
                    'product-company-ceo', 'product-company-hq', 
                    'rgb-rot120-name',  'word-substring-reverse',]

num_num = ['plus-hundred-times-two', 'plus-ten-times-two', 'word-int-times-two','mod-twenty-times-two', ]
str_year = ['person-university-year', 'movie-director-birthyear', 'book-author-birthyear', 'song-artist-birthyear']
year_num=["director-birthyear-times-two","artist-birthyear-times-two"]#,"author-birthyear-times-two"]
num_str = [
         #"num-mod7-weekday",
          #"int-mod4-season",
           "int-plus5-parity", "int-plus5-str", 
          "int-plus2-str",
            "int-plus8-str",
            "int-mod9-str",
            "int-plus2-parity",
            "int-plus8-parity"]
#year_str=["director-birthyear-str","artist-birthyear-str","author-birthyear-str"]

bins={"all":str_str+num_num+str_year+year_num+num_str,
      #"str_str":str_str,
      #"num_num":num_num,
      #"str_str_num_num":str_str+num_num,
      #"str_year":str_year,
      #"year_num":year_num,
      #"num_str":num_str
    
    }

SV_WEIGHT_MODE = "sv"  # "sv": normalised by sum(sv); "sv_squared": normalised by sum(sv²); "none": direct sv-weighted sum
# _auto_cluster_center names this script's plots regardless of clustering
# method (manual vs. automatic), so append AUTO_CLUSTER_USING to keep today's
# real auto-clustered ("train"/"all") results from landing in the same
# directory as older manual-A/B/C runs of this same script.
out_dir = f'PAPER_PLOTS_multihop_{model_save_name}/multihop_onerun_allplots_{acc_cond}_auto_cluster_center_{AUTO_CLUSTER_USING}'

for bin_name in bins.keys():
    task_list = bins[bin_name]
    print(task_list)

    out = select_best_layer_by_train_auc_and_plot_test_for_datasets(
        dataset_names=task_list,
        xs_by_layer=xs_by_layer,
        y_acc_by_layer=y_acc_by_layer,
        eligible=eligible,
        out_dir=os.path.join(out_dir, bin_name),
        dataset_field="task",
        train_ratio=0.1,
        random_seed=0,
        quantile_step=0.1,
        subspace_center=False,
        sweep_subspace_var_prop=[0.85, 0.9, 0.95, 0.99],
        n_random_trials=20,
        show=True,
        show_ortho_range=False,
        sv_weight_mode=SV_WEIGHT_MODE,

        plot_every_layer_test=True,
        #fixed_layer_to_plot=14,

    )
