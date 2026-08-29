# ============================================================
# KLAR eval + save residual streams (NO BATCHING)
#
# Saves:
#   - resid_last_all_layers.memmap  [N, n_layers, d_model]
#   - indices.npy, lang_id.npy, rel_id.npy, match.npy
#   - predictions_{n_shots}shot.jsonl
# ============================================================

import os
import json
import glob
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset

from helpers import *  # expects load_hooked_transformer_model etc.

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama-3b")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_new_tokens", type=int, default=10)
parser.add_argument("--n_shots", type=int, default=0, choices=[0, 3])
parser.add_argument("--print_every", type=int, default=250)
parser.add_argument("--print_first_k", type=int, default=10)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
parser.add_argument("--prompt_template_which", type=int, default=0, choices=[0,1,2,3,4], help="Which prompt template to use (0/1/2).")

args = parser.parse_args()

MODEL_NAME = args.model_name
OUT_DIR = f"klar/{MODEL_NAME}_eval_save_all_layers_prompt{args.prompt_template_which}"
MAX_NEW_TOKENS = args.max_new_tokens
N_SHOTS = args.n_shots
DTYPE = np.float16 if args.dtype == "float16" else np.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# -----------------------------
# Relations / languages
# -----------------------------
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion",
]
valid_langs = ["en","ca","es","fr","hu","ja","ko","nl","ru","uk","vi","zh"]
valid_rels = set(relations)
rel2id = {r: i for i, r in enumerate(relations)}
id2rel = {i: r for r, i in rel2id.items()}
lang2id = {l: i for i, l in enumerate(valid_langs)}
id2lang = {i: l for l, i in lang2id.items()}

# -----------------------------
# Load KLAR JSONs
# -----------------------------
json_paths = glob.glob("klar/klar/*/*.json")
path_map = defaultdict(dict)  # rel -> lang -> path

for p in json_paths:
    lang = os.path.basename(os.path.dirname(p))
    rel = os.path.splitext(os.path.basename(p))[0]
    if (lang in valid_langs) and (rel in valid_rels):
        path_map[rel][lang] = p

# -----------------------------
# Build samples
# -----------------------------
samples = []
for rel, lang_paths in path_map.items():
    if not all(l in lang_paths for l in valid_langs):
        continue

    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
        loaded_samples = content["samples"]
        template = content["prompt_templates"][args.prompt_template_which].split("<mask>")[0] + "<mask>"

        for s in loaded_samples:
            samples.append({
                "subject": s["subject"],
                "object": s["object"],
                "language": lang,
                "relation": rel,
                "template": template,
                "index": int(s["index"]),   # NOTE: often shared across languages!
            })

def apply_prompt(ex):
    prompt = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "")
    ex["input"] = prompt.strip()
    ex["target"] = " " + ex["object"]
    return ex

dataset = Dataset.from_list([apply_prompt(x) for x in samples])
dataset_list = list(dataset)

print("Total examples:", len(dataset_list))

# -----------------------------
# Matching (same as your earlier)
# -----------------------------
def is_nontrivial_prefix(a: str, b: str) -> bool:
    a = a.lower().strip()
    b = b.lower().strip()
    return len(a) > 0 and b.startswith(a)

def match_pred_target(pred: str, target: str) -> bool:
    return is_nontrivial_prefix(pred, target) or is_nontrivial_prefix(target, pred)

# -----------------------------
# Few-shot prompt building (one-by-one)
# -----------------------------
def build_fewshot_prompt(ex, pool, n_shot: int):
    if n_shot <= 0:
        return ex["input"]

    candidates = [
        c for c in pool
        if c["language"] == ex["language"]
        and c["relation"] == ex["relation"]
        and c["index"] != ex["index"]
    ]
    demos = random.sample(candidates, k=min(n_shot, len(candidates)))
    return "".join([f"{d['input']}{d['target']}\n" for d in demos]) + ex["input"]

# -----------------------------
# Memmap creation
# -----------------------------
def create_memmaps(out_dir: str, N: int, n_layers: int, d_model: int):
    os.makedirs(out_dir, exist_ok=True)

    idx_path  = os.path.join(out_dir, "indices.npy")
    lang_path = os.path.join(out_dir, "lang_id.npy")
    rel_path  = os.path.join(out_dir, "rel_id.npy")
    ok_path   = os.path.join(out_dir, "match.npy")
    last_path = os.path.join(out_dir, "resid_last_all_layers.memmap")

    indices = np.empty((N,), dtype=np.int64)
    lang_id = np.empty((N,), dtype=np.int16)
    rel_id  = np.empty((N,), dtype=np.int16)
    match   = np.empty((N,), dtype=np.uint8)

    resid_last = np.memmap(last_path, mode="w+", dtype=DTYPE, shape=(N, n_layers, d_model))

    paths = {
        "indices": idx_path,
        "lang_id": lang_path,
        "rel_id": rel_path,
        "match": ok_path,
        "resid_last_all_layers": last_path,
    }

    return {
        "indices": indices,
        "lang_id": lang_id,
        "rel_id": rel_id,
        "match": match,
        "resid_last": resid_last,
        "paths": paths,
    }

def finalize_small_arrays(indices, lang_id, rel_id, match, idx_path, lang_path, rel_path, ok_path):
    np.save(idx_path, indices)
    np.save(lang_path, lang_id)
    np.save(rel_path, rel_id)
    np.save(ok_path, match)

# -----------------------------
# Collect all-layer resid_pre for ONE prompt (no batching)
# -----------------------------
@torch.no_grad()
def get_all_layers_last_resid_pre(model, input_ids_1d):
    """
    input_ids_1d: torch.LongTensor [T]
    Returns:
      last_stack: [n_layers, d_model]  (resid_pre at last token position)
    """
    n_layers = model.cfg.n_layers

    tokens = input_ids_1d.unsqueeze(0)  # [1, T]
    last_by_layer = [None] * n_layers

    def make_hook(layer_idx):
        def hook_fn(act, hook):
            last_by_layer[layer_idx] = act[0, -1, :].detach()
        return hook_fn

    hooks = [(f"blocks.{l}.hook_resid_pre", make_hook(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(tokens, fwd_hooks=hooks, return_type=None)

    last_stack = torch.stack(last_by_layer, dim=0)  # [L, D]
    return last_stack

# -----------------------------
# Greedy generation
# -----------------------------
@torch.no_grad()
def generate_one_completion_like_eval_py(model, tokenizer, prompt: str, max_new_tokens: int, device):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)  # [1, T]
    out_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_type="tokens",
    )
    decoded = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    pred = decoded[len(prompt):].split("\n")[0].strip()
    return pred

# -----------------------------
# Main evaluate + save (no batching)
# -----------------------------
@torch.no_grad()
def evaluate_and_save_all_layers_no_batch(
    model, tokenizer, dataset_list,
    out_dir: str,
    n_shot: int,
    max_new_tokens: int,
    device,
    print_every: int = 250,
    print_first_k: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.to(device)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab
    N = len(dataset_list)

    store = create_memmaps(out_dir=out_dir, N=N, n_layers=n_layers, d_model=d_model)

    indices = store["indices"]
    lang_id = store["lang_id"]
    rel_id  = store["rel_id"]
    match   = store["match"]
    resid_last = store["resid_last"]
    paths = store["paths"]

    pred_path = os.path.join(out_dir, f"predictions_{n_shot}shot.jsonl")
    pred_f = open(pred_path, "w", encoding="utf-8")

    correct = 0

    for row, ex in enumerate(tqdm(dataset_list, desc=f"[eval+save {n_shot}-shot | all layers | no batch]")):
        prompt = build_fewshot_prompt(ex, dataset_list, n_shot=n_shot)

        # --- (A) resid reps for the prompt
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids_1d = enc["input_ids"][0].to(device)
        last_stack = get_all_layers_last_resid_pre(model, input_ids_1d)

        resid_last[row, :, :] = last_stack.to(torch.float16).cpu().numpy()

        # --- (B) generation
        pred = generate_one_completion_like_eval_py(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )

        target = ex["target"]
        ok = match_pred_target(pred, target)

        # --- write small arrays
        indices[row] = int(ex["index"])
        lang_id[row] = int(lang2id[ex["language"]])
        rel_id[row]  = int(rel2id[ex["relation"]])
        match[row]   = 1 if ok else 0
        correct += int(ok)

        # --- log JSONL
        json.dump({
            "lang": ex["language"],
            "relation": ex["relation"],
            "index": int(ex["index"]),
            "input": ex["input"],
            "target": target,
            "prediction": pred,
            "match": bool(ok),
        }, pred_f, ensure_ascii=False)
        pred_f.write("\n")

        if row < print_first_k or (print_every > 0 and row % print_every == 0):
            print(f"{row}: pred={pred!r}  target={target!r}  match={ok}")

        # flush periodically
        if row > 0 and row % 2000 == 0:
            resid_last.flush()

    # final flush
    resid_last.flush()
    pred_f.close()

    finalize_small_arrays(
        indices, lang_id, rel_id, match,
        paths["indices"], paths["lang_id"], paths["rel_id"], paths["match"]
    )

    acc = correct / N if N else 0.0
    summary = {
        "n_shot": n_shot,
        "max_new_tokens": max_new_tokens,
        "N": N,
        "acc": float(acc),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "d_vocab": int(d_vocab),
        "paths": {
            **paths,
            "predictions_jsonl": pred_path,
        },
        "langs": valid_langs,
        "relations": relations,
    }

    with open(os.path.join(out_dir, f"summary_{n_shot}shot.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. acc={acc:.4f}  N={N}")
    print("Saved:", os.path.join(out_dir, f"summary_{n_shot}shot.json"))
    return summary

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"langs": valid_langs, "relations": relations, "lang2id": lang2id, "rel2id": rel2id},
            f, indent=2, ensure_ascii=False
        )

    model, tokenizer = load_hooked_transformer_model(MODEL_NAME)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_dir = os.path.join(OUT_DIR, f"{MODEL_NAME}_{N_SHOTS}shot")
    os.makedirs(run_dir, exist_ok=True)

    _summary = evaluate_and_save_all_layers_no_batch(
        model=model,
        tokenizer=tokenizer,
        dataset_list=dataset_list,
        out_dir=run_dir,
        n_shot=N_SHOTS,
        max_new_tokens=MAX_NEW_TOKENS,
        device=device,
        print_every=args.print_every,
        print_first_k=args.print_first_k,
    )
