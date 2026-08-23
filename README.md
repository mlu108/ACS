# ACS

Two experiment pipelines, pulled out of `dynamic_stress_test/` into their own folder:

- **Multihop** — composition-of-functions probing (e.g. "director → birth year → parity").
- **Multilingual (KLAR + OSCAR)** — cross-lingual factual-recall probing, using KLAR relation
  data and OSCAR per-language "background" residual streams to build language subspaces.

Each pipeline has the same two-stage shape:

```
[stage 1] *_save_resid_stream*.py / dynamic_tyler_geometry.py
              → runs the model, saves residual streams + logits to disk
[stage 2] *_experiments*.py
              → reads the saved residuals, runs the actual geometry / CI-prediction
                experiments, writes plots (svg) + tables (json/csv) under weekly_meetings/
```

Stage 1 is the expensive, GPU-bound part. Stage 2 is CPU-only analysis over whatever
stage 1 already wrote to disk — it does **not** need the model loaded (it stubs out
`model` as a plain config namespace; see each script's Inputs section).

## Folder layout

```
ACS/
├── README.md                                        (this file)
├── helpers.py                                        shared model-loading helper
├── multihop_auto_cluster.py                           helper, imported by multihop experiments
├── geometric_feature_experiment_auc_plots.py           helper, imported by geometric_feature_...
│
├── multihop_datasets_save_resid_stream_logits.py       [multihop stage 1]
├── multihop_experiments_auto_cluster_center.py         [multihop stage 2]
│
├── KLAR_datasets_save_resid_streams_logits.py           [multilingual stage 1a — KLAR]
├── dynamic_tyler_geometry.py                            [multilingual stage 1b — OSCAR]
├── geometric_feature_experiment_merged_lans.py          [multilingual stage 2]
│
└── klar/
    └── klar/<lang>/<relation>.json                      raw KLAR dataset (15M, copied in)
```

No OSCAR raw-data folder is copied in — see the OSCAR note in `dynamic_tyler_geometry.py`'s
section below.

## Before you run anything: set HF_TOKEN

`helpers.py` (`_maybe_login_hf`) and `dynamic_tyler_geometry.py` both call
`huggingface_hub.login(...)` / `load_dataset(..., token=...)` using `os.environ["HF_TOKEN"]`
— set that env var before running anything in this folder, e.g.:
```bash
export HF_TOKEN=hf_xxx...
```
(The original copies in `dynamic_stress_test/` — including `cpu_script_multihop.sh` — still
have the token hardcoded in plaintext; that's unrelated to this folder and untouched.)

## Helper modules (no inputs/outputs of their own)

| File | Used by | What it provides |
|---|---|---|
| `helpers.py` | both stage-1 KLAR/multihop scripts, `dynamic_tyler_geometry.py` | `load_hooked_transformer_model(name)` — resolves a short model alias (`llama-3b`, `qwen-14b`, `olmo-7b`, ...) to a HF repo id, logs into HF, loads a `transformer_lens` `TransformerBridge` + tokenizer |
| `multihop_auto_cluster.py` | `multihop_experiments_auto_cluster_center.py` | auto-clustering of multihop tasks into composition groups |
| `geometric_feature_experiment_auc_plots.py` | `geometric_feature_experiment_merged_lans.py` | AUC-matrix heatmap plotting utilities |

---

## Multihop pipeline

### 1. `multihop_datasets_save_resid_stream_logits.py` — save residual streams

**Inputs**
- HF dataset `apoorvkh/composing-functions` (auto-downloaded, no local file needed).
- Model, via `load_hooked_transformer_model(--model_name)`.
- No local data files required — this script is fully self-contained given internet + HF auth.

**CLI args** (all optional, `argparse`):

| Flag | Default | Notes |
|---|---|---|
| `--model_name` | `llama-3b` | passed straight to `helpers.load_hooked_transformer_model` |
| `--seed` | `12345` | |
| `--out_dir` | `composing_functions_save_resid_logits` | **override this** — see naming note below |
| `--max_new_tokens` | `5` | |
| `--dtype` | `float16` | `float16` or `float32`, dtype of the saved memmap |
| `--n_icl` | `10` | number of in-context examples per query |
| `--n_queries_per_task` | `None` (all) | |
| `--print_every` / `--print_first_k` | `50` / `10` | logging only |

**Naming requirement:** stage 2 (`multihop_experiments_auto_cluster_center.py`) hardcodes
its expected input directory as `multihop_functions_resid_logits_{model_save_name}`, where
`model_save_name` ∈ `{olmo_7b, llama_3b, qwen_14b}` (from that script's `model_name_dict`).
So run this script with, e.g.:
```
python multihop_datasets_save_resid_stream_logits.py \
    --model_name llama-3b --out_dir multihop_functions_resid_logits_llama_3b
```

**Outputs** (under `--out_dir`):
- `meta.json` — task/prompt-type id maps
- `task_id.npy`, `prompt_type_id.npy`, `local_query_idx.npy`, `global_row_idx.npy`, `match.npy`
- `resid_last_all_layers.memmap` — shape `[N, n_layers, d_model]`, dtype per `--dtype`
- `predictions.jsonl` — per-row prompt/target/prediction/match
- `summary.json` — N, accuracy, n_layers, d_model, d_vocab, task/prompt-type maps, file paths

### 2. `multihop_experiments_auto_cluster_center.py` — CI prediction experiments

**Inputs**
- `--model_name` (added in this reorg): accepts a full HF id, a short save-name
  (`olmo_7b`/`llama_3b`/`qwen_14b`), or a hyphenated alias (`olmo-7b`/`llama-3b`/`qwen-14b`).
  Default `allenai/OLMo-7B`.
- Requires `multihop_functions_resid_logits_{model_save_name}/` (stage 1's output) to already
  exist in the current working directory.
- Loads a tokenizer only (`AutoTokenizer`/`PreTrainedTokenizerFast`) — no model weights, no GPU.

**Outputs** — root: `weekly_meetings/geometric_features_experiments/`
- `auto_clusters/auto_clusters_{train|all}.json` — the learned task→cluster assignment
- `auto_clusters/auto_cluster_plots_{train|all}/` — interactive HTML cluster-visualization
  plots (only when `AUTO_CLUSTER_DEBUG_MODE = True`, the current default)
- Many further per-experiment `.svg`/`.json`/`.csv` files under the same `weekly_meetings/`
  root (cumulative-AUC curves, PR-AUC curves, corr-vs-layer plots, distance/scatter plots) —
  driven by the `DO_EXP*`-style flags near the bottom of the script, mirroring the structure
  used in `geometric_feature_experiment_merged_lans.py` below.

---

## Multilingual (KLAR + OSCAR) pipeline

### 1a. `KLAR_datasets_save_resid_streams_logits.py` — save KLAR residual streams

**Inputs**
- Local KLAR dataset at `klar/klar/<lang>/<relation>.json` (glob `klar/klar/*/*.json`) —
  **copied into this folder** at `ACS/klar/klar/`. Run this script with CWD = `ACS/` (or
  wherever `klar/klar/` sits relative to you) so the glob resolves.
- Model, via `load_hooked_transformer_model(--model_name)`.

**CLI args**:

| Flag | Default | Notes |
|---|---|---|
| `--model_name` | `llama-3b` | |
| `--seed` | `12345` | |
| `--out_dir` | `klar/llama_3b_eval_save_all_layers_nobatch` | **override this** — see naming note below |
| `--max_new_tokens` | `10` | |
| `--n_shots` | `0` | `0` or `3` |
| `--dtype` | `float16` | |
| `--prompt_template_which` | `0` | index into each relation JSON's `prompt_templates`; must match `prompt_num` used in stage 2 |
| `--print_every` / `--print_first_k` | `250` / `10` | logging only |

**Naming requirement:** stage 2 (`geometric_feature_experiment_merged_lans.py`) hardcodes
`RUN_DIR = f"klar/{MODEL}_eval_save_all_layers_prompt{prompt_num}/{MODEL}_{shot_tag_num}"`
(default `prompt_num=0`, `shot_tag_num='0shot'`). This script builds its run folder as
`{out_dir}/{model_name}_{n_shots}shot`, so to line up exactly:
```
python KLAR_datasets_save_resid_streams_logits.py \
    --model_name llama-3b --n_shots 0 --prompt_template_which 0 \
    --out_dir klar/llama-3b_eval_save_all_layers_prompt0
```
→ produces `klar/llama-3b_eval_save_all_layers_prompt0/llama-3b_0shot/`, matching stage 2's
`RUN_DIR` exactly.

**Outputs**:
- `{out_dir}/meta.json` — lang/relation id maps
- `{out_dir}/{model_name}_{n_shots}shot/`:
  - `indices.npy`, `lang_id.npy`, `rel_id.npy`, `match.npy`
  - `resid_last_all_layers.memmap` — shape `[N, n_layers, d_model]`
  - `predictions_{n_shots}shot.jsonl`
  - `summary_{n_shots}shot.json`

### 1b. `dynamic_tyler_geometry.py` — save OSCAR per-language background residuals

**Inputs**
- HF dataset `oscar-corpus/OSCAR-2109` (gated — streamed live via `load_dataset(...,
  streaming=True)`, requires the HF token noted above). **There is no local raw-OSCAR
  directory to copy in** — it's fetched fresh each run, so nothing was copied into `ACS/`
  for this. (`hf_datasets_cache/` next to the original script was empty — OSCAR isn't
  persisted locally even in the source layout.)
- Model **is not a CLI arg** in this script (unlike the others) — it's hardcoded at the top:
  ```python
  available_models = ['llama-1b','llama-3b','llama-7b','llama-8b','Qwen2_1.5b','Qwen2-0.5B','Qwen2.5-14B']
  MODEL = available_models[-1]   # currently picks 'Qwen2.5-14B'
  ```
  Edit that line directly to change which model's OSCAR residuals you're building.
- Languages are a hardcoded list (line ~218): french, japanese, korean, spanish, english,
  chinese, hungarian, dutch, russian, ukrainian, vietnamese (11 of the 12 languages KLAR
  covers — catalan is not included here).

**Outputs** (path was `dynamic_stress_test/oscar_geometry_{MODEL}/...` in the original;
edited here to drop that prefix so it's self-contained under `ACS/`):
```
oscar_geometry_{MODEL}/all_one_lan_data_resid_pre_oscar2109/{LANGUAGE}/LAYER_{layer}_{LANGUAGE}.pt
```
one `.pt` tensor per (language, layer), shape `[n_sequences * tokens_per_seq, d_model]`.
This is the raw residual data that stage 2 below reduces to per-language SVD subspaces
(and caches — see next section). It's large; feel free to delete it once
`oscar_subspaces_cache_{MODEL}/` has been built from it.

### 2. `geometric_feature_experiment_merged_lans.py` — CI prediction experiments

**Inputs**
- `--model` (argparse, `parse_known_args` so it's Jupyter-safe): one of `llama-1b, llama-3b,
  llama-7b, llama-8b, Qwen2_1.5b, Qwen2-0.5B, qwen_14b` (aliases like `llama_3b`/`qwen-14b`
  also accepted). Default `llama-3b`.
- KLAR data at `klar/{MODEL}_eval_save_all_layers_prompt{prompt_num}/{MODEL}_{shot_tag_num}/`
  — i.e. stage 1a's output, with `prompt_num=0`, `shot_tag_num='0shot'` by default (edit the
  `prompt_num`/`shot_tag` variables near the top of the script to change).
- OSCAR data — **either**:
  - `oscar_geometry_{MODEL}/all_one_lan_data_resid_pre_oscar2109/` (stage 1b's raw output —
    the script will build the SVD subspace from it and populate the cache below), **or**
  - `oscar_subspaces_cache_{MODEL}/` already populated (skips needing the raw `.pt` files
    at all — this is what you want if you're only re-running the analysis, not rebuilding
    subspaces from scratch).
- Local import: `geometric_feature_experiment_auc_plots.py`.

Current default experiment config (edit the block right after `# OSCAR cache roots` /
`valid_langs =` near the top if you want different settings): `target_kind='binary_correct'`,
`shot_tag='zeroshot'`, `prompt_num=0`, `mean_center_by_cluster=True`, `rep_kind='last'`,
`lang_mode=('center_oscar_and_uncentered_language_subspace','SVD',0.99)`,
`valid_langs=[es,fr,hu,ja,ko,nl,ru,uk,vi,zh]` (`en` excluded by default — see the
`valid_langs = valid_langs #+['en']` line if you need to include it, e.g. for debugging).
`DO_EXP1` and `DO_EXP2` run by default; `DO_EXP0/3/3B/4/5` are off.

**Outputs**:
- `oscar_subspaces_cache_{MODEL}/...` — cached per-language SVD subspaces (`Vk`, `mu`, `sv`
  `.npz` files), built lazily from the raw OSCAR `.pt` files the first time each
  (language, layer, var_prop, split) combination is needed. **This is the "cached repr"
  you'd otherwise have had to copy** — just run this script once against stage 1b's output
  and it populates itself.
- `weekly_meetings/geometric_features_experiments_{MODEL}/` — main plot/table root:
  ```
  prompt{N}_{shot_tag}_{rep_kind}_{lang_mode}_{target_kind}_{y_transform}_centerenfacts{bool}/
    {feature}_lowest_feature_first/
      exp1_dev_test_split/          exp1_*.svg, exp1_*.json, auc_tables/, pr_auc_tables/, ...
      exp2_...                      (per-relation subspace experiment outputs)
      ...
  ```
  (This is the same output family explored extensively earlier in this conversation —
  `exp1_en_fact_cumulative_coherence.svg` and friends live here.)

---

## Quick start: regenerating everything from scratch for one model

```bash
cd ACS
MODEL=llama-3b            # or llama_3b / llama-7b / qwen-14b / olmo-7b, per script's own alias rules

# Multihop
python multihop_datasets_save_resid_stream_logits.py \
    --model_name $MODEL --out_dir multihop_functions_resid_logits_llama_3b
python multihop_experiments_auto_cluster_center.py --model_name $MODEL

# Multilingual — KLAR
python KLAR_datasets_save_resid_streams_logits.py \
    --model_name $MODEL --n_shots 0 --prompt_template_which 0 \
    --out_dir klar/${MODEL}_eval_save_all_layers_prompt0

# Multilingual — OSCAR (edit MODEL at the top of the script first, no CLI arg)
python dynamic_tyler_geometry.py

# Multilingual — experiments (builds + caches OSCAR subspaces on first run)
python geometric_feature_experiment_merged_lans.py --model $MODEL
```

Re-running the last two commands for a different model only needs the raw OSCAR `.pt` data
and the KLAR eval directory for that model — the SVD-subspace cache and all plots regenerate
from those automatically.
