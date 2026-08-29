# Adversarial Concept Search (ACS)

Code for **"Adversarial Concept Search: Predicting Compositional Errors From Feature
Geometry"** ([paper PDF](ACS_paper.pdf)).

We predict *where* an LLM will fail on compositional tasks — multihop reasoning,
multilingual factual recall — using only the geometry of its internal representations,
without evaluating the composed input itself. The core quantity is **Compositional
Interference (CI)**: how non-orthogonal (close in angle) the atomic concept
representations `a_i` that make up a composition are. Near-orthogonal concepts compose
reliably; concepts encoded close together interfere, and interference predicts failure.

![Can we find challenging concept combinations for LLMs without explicitly evaluating them?](ACS_title_figure.pdf)

This repo has two stages per task, matching the paper's pipeline:

```
stage 1  save residual streams   (*_save_resid_stream*.py / dynamic_tyler_geometry.py)
            → run the model, cache last-token residuals + logits for the atomic queries
stage 2  compute CI + evaluate   (*_experiments*.py)
            → cluster-center residuals into concept representations a_i, compute CI,
              and correlate CI against composed-query accuracy (Figures 4–5 in the paper)
```

Stage 1 needs a GPU (loads the model). Stage 2 is CPU-only — it only reads what stage 1
wrote to disk.

`helpers.py` (`load_hooked_transformer_model(name)`), `multihop_auto_cluster.py`, and
`multilingual_experiment_auc_plots.py` are shared helpers, imported by the scripts below
rather than run directly.

---

## I. Multihop Reasoning

Section 4.1 of the paper: a composed query `g(f(x))` chains an atomic first-hop query `f`
(e.g. `author of 1984`) into an atomic second-hop query `g`
(`birthyear of [the author of 1984]`). We extract last-token residuals for the atomic
first- and second-hop queries, cluster-center them into concept representations `a_f` and
`a_g`, and use CI between them to predict whether the model composes them correctly.

- `multihop_datasets_save_resid_stream_logits.py` — stage 1
- `multihop_experiments_auto_cluster_center.py` — stage 2

**Pipeline:**
```bash
cd ACS
python multihop_datasets_save_resid_stream_logits.py --model_name llama-3b
python multihop_experiments_auto_cluster_center.py --model_name llama-3b --experiments_dir multihop_experiments
```

**Notes**
- `multihop_datasets_save_resid_stream_logits.py` pulls the `apoorvkh/composing-functions`
  HF dataset automatically — no local data needed. For every query it saves the last-token
  residual stream (all layers) for all three prompt types: `Qx_Fx` (first hop), `QFx_GFx`
  (second hop given the gold first-hop answer), and `Qx_GFx` (the full composed query).
  Output directory is fixed — `multihop_functions_resid_logits_{model_name}` (hyphens →
  underscores) — which stage 2 looks for automatically, so just keep `--model_name`
  consistent between the two commands.
- `multihop_experiments_auto_cluster_center.py` builds `a_f`/`a_g` from the stage-1 residuals,
  auto-clusters tasks by composition family, and computes CI to predict composed-query
  accuracy (the multihop curve in Figure 4a). Loads a tokenizer only — no model weights, no
  GPU. Writes auto-cluster plots (`.svg`/`.html`) and tables (`.json`) under
  `--experiments_dir` (default `multihop_experiments/`).

---

## II. Multilingual Fact Recall

Section 4.1 of the paper: for a factual query in target language `ℓ`, the active concept
set is a factual concept `q` (its English form) and a language concept `ℓ`. We estimate the
fact representation `a_q` from the English query and the language subspace `B_ℓ` via SVD
over residuals from OSCAR text in that language, then compute CI between `a_q` and `B_ℓ` to
predict cross-lingual transfer failure — without needing the translated input.

- `KLAR_datasets_save_resid_streams_logits.py` — stage 1a (KLAR factual queries)
- `dynamic_tyler_geometry.py` — stage 1b (OSCAR per-language background residuals)
- `multilingual_experiment_merged_lans.py` — stage 2

**Pipeline:**
```bash
cd ACS
python KLAR_datasets_save_resid_streams_logits.py --model_name llama-3b --n_shots 0 --prompt_template_which 0
python dynamic_tyler_geometry.py --model_name llama-3b
python multilingual_experiment_merged_lans.py --model llama-3b
```

**Notes**
- `KLAR_datasets_save_resid_streams_logits.py` reads `klar/klar/<lang>/<relation>.json`
  (run with CWD = `ACS/` so the glob resolves) and saves last-token residuals + predictions
  per query. Output directory is fixed —
  `klar/{model_name}_eval_save_all_layers_prompt{prompt_template_which}` — matching what
  stage 2 expects automatically.
- `dynamic_tyler_geometry.py` streams the (gated) `oscar-corpus/OSCAR-2109` dataset live —
  nothing local to copy in. This is the raw material stage 2 reduces to the per-language
  SVD subspace `B_ℓ`. `--model_name` selects the model; the language list is still hardcoded
  at the top of the script. Saves to `oscar_geometry_{model_name}/...`; large, safe to
  delete once stage 2 has built `oscar_subspaces_cache_{model_name}/` from it.
- `multilingual_experiment_merged_lans.py` needs stage 1a's KLAR output and either
  stage 1b's raw OSCAR residuals or an already-populated `oscar_subspaces_cache_{model_name}/`
  (the SVD cache is built lazily on first run, so you only need the raw OSCAR data once). It
  computes CI(`a_q`, `B_ℓ`) per language and evaluates it against fact-recall accuracy
  (Figures 4b–c, 5 in the paper). Writes under
  `multilingual_experiments_{model_name}/`.

---

## III. SCAN (synthetic proof-of-concept)

Section 3 of the paper: toy autoregressive Transformers trained from scratch on SCAN
(varying training coverage and model dimension), used to show — under full control of the
data distribution and model scale — that the angle between atomic-concept representations
(`walk`, `left`, `twice`, ...) predicts compositional generalization failure on held-out
instruction compositions.

> **Adapted from** [`ryeii/Representational-Homomorphism-for-Transformer-Language-Models`](https://github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models)
> (`he_probe/`). Only the files this pipeline actually uses are copied in here — no saved
> checkpoints, result JSONs, or figures from the source repo. Each copied file carries a
> one-line header pointing back to its source path in that repo.

- `SCAN/he_probe/experiment_scan.py` — stage 1: train toy Transformers on SCAN
- `SCAN/he_probe/analysis_atomic_avg.py` — stage 2: compute CI and evaluate against OOD accuracy
- `SCAN/make_summary_plots.py` — aggregates stage 2's per-checkpoint outputs into summary plots
- `SCAN/he_probe/gen_data_scan.py`, `SCAN/he_probe/transformers.py` — SCAN data generation and
  the toy `DecoderOnlyTransformer`, imported by both stages
- `SCAN/he_probe/experiment_scan_configs/` — training configs (sweep seeds, model size, etc.)

**Pipeline** (run from `SCAN/`, so `he_probe` resolves as a package):
```bash
cd ACS/SCAN

# stage 1 — train a sweep of checkpoints
python he_probe/experiment_scan.py --config he_probe/experiment_scan_configs/config_d32_sweep_seeds_fixed_seed_data2.json

# stage 2 — for each checkpoint, compute CI and evaluate OOD accuracy
python -m he_probe.analysis_atomic_avg \
    --checkpoint <path/to/checkpoint.pt> \
    --results_dir <base_results_dir from the stage-1 config> \
    --seed <seed> --d_model <d_model> --n_heads <n_heads> --d_ff <d_ff> --n_layers <n_layers> \
    --data_mode size_variation --size_variation_p <size_p> \
    --aggregation cumulative_coherence --bin_width 0.1 \
    --dev_source from_test_split --dev_fraction 0.1 \
    --select_best_layer cumulative

# optional — aggregate every stage-2 run under one results_dir into summary plots
python make_summary_plots.py --results_dir <base_results_dir>
```

**Notes**
- `experiment_scan.py --config ...` trains one model per (seed × architecture × coverage)
  combination in the config and saves a `.pt` checkpoint per run under `base_results_dir`
  (set in the config, e.g. `results_scan_sweep_seeds`). CLI flags of the same name (e.g.
  `--epochs`, `--lr`) override individual config fields without editing the JSON.
- `python -m he_probe.analysis_atomic_avg` takes one `--checkpoint` at a time. Its
  `--seed`/`--d_model`/`--n_heads`/`--d_ff`/`--n_layers`/`--size_variation_p` must match the
  values used to train that checkpoint (they're encoded in the checkpoint's filename). It
  extracts atomic-concept representations, computes CI (`--aggregation
  cumulative_coherence`), and evaluates CI against held-out compositional accuracy, writing
  `summary.json` and plots under `--results_dir/<size_p dir>/<--results_subdir>/<run_name>/`.
  It caches each checkpoint's per-example dev/test correctness the first time it runs
  generation, so re-running with different aggregation/analysis flags on the same checkpoint
  is fast. To sweep this over every checkpoint under a `base_results_dir`, loop over
  `find base_results_dir -name "*.pt"` and parse each filename for `seed`/`dmodel`/`nheads`/
  `dff`/`layer`/`size_p`, then call the command above per checkpoint (this is what
  `run_analysis_all_cumulative_coherence.sh` in the source repo does — not copied here, since
  it's cluster-submission glue rather than pipeline code).
- `make_summary_plots.py` reads every run's `summary.json` under `--results_dir` and produces
  cross-run bar/line charts (accuracy and orthogonality vs. layer, vs. coverage) under
  `--results_dir/Plots/`.

**Demo notebook** — `SCAN/scan_ci_demo.ipynb` walks through one representative checkpoint
(`d_model=12`, `8%` coverage, `seed=42`, best layer 6): a pairwise-CI lower-triangle heatmap and
a 2D PCA of its atomic-concept representations, then cumulative, noncumulative, and error-recall
curves, then the 10 SCAN test examples with the highest CI (the model's predicted hardest),
ranked high-to-low, with the gold action sequence and whether the model actually got each one
right. `d_model=12` was picked over the smallest `d_model=8` checkpoint because CI's signal is
visibly cleaner once the model isn't capacity-starved (96% vs. 47% accuracy; smooth,
near-monotonic curves; error-recall AUC-above-diagonal 0.224 vs. 0.029). The curve/heatmap/PCA
plots use the same numeric routines as `analysis_atomic_avg.py` (imported, not reimplemented)
but are styled to match `Representational-Homomorphism-.../PAPER_PLOTS_scan.ipynb`'s
`_plot_setup`/heatmap/PCA cells instead of that script's own plain plots — no permutation test,
so it runs in seconds. Needs no GPU/model reload: it runs entirely off
`SCAN/demo_data/size_p8_dmodel12/` (that checkpoint's cached atomic-concept representations,
per-example CI/correctness table, heatmap/PCA data, and summary metrics — copied in from
`results_scan_paper_new_dev/size_p8/` in the source repo, same attribution as above). Run from
`SCAN/` so `he_probe` resolves.
