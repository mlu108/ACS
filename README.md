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
`geometric_feature_experiment_auc_plots.py` are shared helpers, imported by the scripts
below rather than run directly.

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
python multihop_experiments_auto_cluster_center.py --model_name llama-3b
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
  GPU. Writes plots (`.svg`) and tables (`.json`) under
  `weekly_meetings/geometric_features_experiments/`.

---

## II. Multilingual Fact Recall

Section 4.1 of the paper: for a factual query in target language `ℓ`, the active concept
set is a factual concept `q` (its English form) and a language concept `ℓ`. We estimate the
fact representation `a_q` from the English query and the language subspace `B_ℓ` via SVD
over residuals from OSCAR text in that language, then compute CI between `a_q` and `B_ℓ` to
predict cross-lingual transfer failure — without needing the translated input.

- `KLAR_datasets_save_resid_streams_logits.py` — stage 1a (KLAR factual queries)
- `dynamic_tyler_geometry.py` — stage 1b (OSCAR per-language background residuals)
- `geometric_feature_experiment_merged_lans.py` — stage 2

**Pipeline:**
```bash
cd ACS
python KLAR_datasets_save_resid_streams_logits.py --model_name llama-3b --n_shots 0 --prompt_template_which 0
python dynamic_tyler_geometry.py --model_name llama-3b
python geometric_feature_experiment_merged_lans.py --model llama-3b
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
- `geometric_feature_experiment_merged_lans.py` needs stage 1a's KLAR output and either
  stage 1b's raw OSCAR residuals or an already-populated `oscar_subspaces_cache_{model_name}/`
  (the SVD cache is built lazily on first run, so you only need the raw OSCAR data once). It
  computes CI(`a_q`, `B_ℓ`) per language and evaluates it against fact-recall accuracy
  (Figures 4b–c, 5 in the paper). Writes under
  `weekly_meetings/geometric_features_experiments_{model_name}/`.

---

## III. SCAN (synthetic proof-of-concept)

Section 3 of the paper: toy autoregressive Transformers trained from scratch on SCAN
(varying training coverage and model dimension), used to show — under full control of the
data distribution and model scale — that the angle between atomic-concept representations
(`walk`, `left`, `twice`, ...) predicts compositional generalization failure on held-out
instruction compositions.

*Code for this section is not yet in this repo — coming soon.*
