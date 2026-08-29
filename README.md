# ACS

Two experiment pipelines: **Multihop** (composition-of-functions probing) and
**Multilingual** (KLAR + OSCAR cross-lingual factual-recall probing). Each pipeline is a
stage-1 (GPU, saves residual streams + logits to disk) → stage-2 (CPU, reads stage 1's
output and runs the actual geometry experiments) shape.

`helpers.py` (shared `load_hooked_transformer_model(name)`), `multihop_auto_cluster.py`,
and `geometric_feature_experiment_auc_plots.py` are imported helpers, not run directly.

---

## I. Multihop

- `multihop_datasets_save_resid_stream_logits.py` — stage 1
- `multihop_experiments_auto_cluster_center.py` — stage 2

**Pipeline:**
```bash
cd ACS
python multihop_datasets_save_resid_stream_logits.py --model_name llama-3b
python multihop_experiments_auto_cluster_center.py --model_name llama-3b
```

**Notes**
- `multihop_datasets_save_resid_stream_logits.py` pulls the `apoorvkh/composing-functions`
  HF dataset automatically — no local data needed. Output directory is fixed:
  `multihop_functions_resid_logits_{model_name}` (hyphens → underscores), which stage 2
  looks for automatically, so just keep `--model_name` consistent between the two commands.
- `multihop_experiments_auto_cluster_center.py` loads a tokenizer only — no model weights,
  no GPU. Writes plots (`.svg`) and tables (`.json`) under
  `weekly_meetings/geometric_features_experiments/`.

---

## II. Multilingual (KLAR + OSCAR)

- `KLAR_datasets_save_resid_streams_logits.py` — stage 1a (KLAR)
- `dynamic_tyler_geometry.py` — stage 1b (OSCAR)
- `geometric_feature_experiment_merged_lans.py` — stage 2

**Pipeline:**
```bash
cd ACS
python KLAR_datasets_save_resid_streams_logits.py --model_name llama-3b --n_shots 0 --prompt_template_which 0
python dynamic_tyler_geometry.py   # edit MODEL at the top of the script first — no CLI arg
python geometric_feature_experiment_merged_lans.py --model llama-3b
```

**Notes**
- `KLAR_datasets_save_resid_streams_logits.py` reads `klar/klar/<lang>/<relation>.json`
  (run with CWD = `ACS/` so the glob resolves). Output directory is fixed:
  `klar/{model_name}_eval_save_all_layers_prompt{prompt_template_which}`, matching what
  stage 2 expects automatically.
- `dynamic_tyler_geometry.py` streams the (gated) `oscar-corpus/OSCAR-2109` dataset live —
  nothing local to copy in. Model and the language list are hardcoded at the top of the
  script (edit directly, no CLI flags). Saves to `oscar_geometry_{MODEL}/...`; large, safe
  to delete once stage 2 has built `oscar_subspaces_cache_{MODEL}/` from it.
- `geometric_feature_experiment_merged_lans.py` needs stage 1a's KLAR output and either
  stage 1b's raw OSCAR `.pt` files or an already-populated `oscar_subspaces_cache_{MODEL}/`
  (the SVD cache is built lazily on first run, so you only need the raw OSCAR data once).
  Writes under `weekly_meetings/geometric_features_experiments_{MODEL}/`.
