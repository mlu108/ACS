import os
import sys
import json
import random
import argparse
from dataclasses import dataclass
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

from helpers import *  # expects load_hooked_transformer_model, etc.

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama-3b")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--out_dir", type=str, default="composing_functions_save_resid_logits")
parser.add_argument("--max_new_tokens", type=int, default=5)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
parser.add_argument("--n_icl", type=int, default=10)
parser.add_argument(
    "--n_queries_per_task",
    type=int,
    default=None,
    help="If omitted, use all eligible query examples in each task.",
)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--print_first_k", type=int, default=10)
args = parser.parse_args()

MODEL_NAME = args.model_name
OUT_DIR = args.out_dir
MAX_NEW_TOKENS = args.max_new_tokens
N_ICL = args.n_icl
N_QUERIES_PER_TASK = args.n_queries_per_task
DTYPE = np.float16 if args.dtype == "float16" else np.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Repro
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


# ============================================================
# Prompt container
# ============================================================
@dataclass
class PromptExample:
    context: list
    query: dict

    def get_prompt(
        self,
        query_type: str = "x",
        pred_type: str = "GFx",
        prompt_prefix: str = "",
        prompt_sep: str = "\n\n",
        trailing_space: bool = False,
    ) -> str:
        prompt = prompt_prefix
        for example in self.context:
            prompt += f"Q: {example[query_type]}\nA: {example[pred_type]}{prompt_sep}"
        prompt += f"Q: {self.query[query_type]}\nA:"
        if trailing_space:
            prompt += " "
        return prompt

def _get_next_global_row_idx(rows_by_task):
    all_existing_indices = []
    for task_rows in rows_by_task.values():
        for r in task_rows:
            all_existing_indices.append(r["global_row_idx"])
    return (max(all_existing_indices) + 1) if all_existing_indices else 0


def _deduplicate_examples_keep_first(rows):
    """
    Remove identical new examples.
    I treat examples as identical if all semantic fields match:
    task, x, Fx, Gx, GFx, FGx
    We keep the first occurrence's global_row_idx.
    """
    seen = set()
    deduped = []
    for r in rows:
        key = (
            r["task"],
            r["x"],
            r["Fx"],
            r["Gx"],
            r["GFx"],
            r["FGx"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _safe_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    x = str(x).strip()
    return int(x)



def generate_new_multihop_dataset_tasks(task_name, rows_by_task, N_examples=500, seed=42):
    """
    Generate new multihop dataset rows for the following tasks:

        - director-birthyear-str
        - director-birthyear-times-two
        - artist-birthyear-str
        - artist-birthyear-times-two
        - author-birthyear-str
        - author-birthyear-times-two
        - num-mod7-weekday
        - num-mod12-month
        - int-mod10-ten-letters
        - int-times3-parity
        - int-plus5-parity
        - int-times3-str
        - int-plus5-str
        - int-mod4-season

        New:
        - int-plus2-str
        - int-plus8-str
        - int-times2-str
        - int-mod9-str
        - int-plus2-parity
        - int-plus8-parity

    Returns:
        new_rows: list[dict]
    """
    rng = random.Random(seed)
    next_global_row_idx = _get_next_global_row_idx(rows_by_task)
    new_rows = []

    # -------------------------------------------------
    # 1) num-mod7-weekday
    # -------------------------------------------------
    if task_name == "num-mod7-weekday":
        weekday_map = {
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
            7: "Sunday",
        }

        for x in range(1, N_examples + 1):
            Fx = ((x - 1) % 7) + 1
            GFx = weekday_map[Fx]

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 2) num-mod12-month
    # -------------------------------------------------
    elif task_name == "num-mod12-month":
        month_map = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }

        for x in range(1, N_examples + 1):
            Fx = ((x - 1) % 12) + 1
            GFx = month_map[Fx]

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 3) int-mod10-ten-letters
    # x -> mod 10 -> uppercase letter A-J
    # -------------------------------------------------
    elif task_name == "int-mod10-ten-letters":
        import string
        letters = {i + 1: string.ascii_uppercase[i] for i in range(10)}  # A-J

        for x in range(1, N_examples + 1):
            Fx = ((x - 1) % 10) + 1
            GFx = letters[Fx]

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 4) int-times3-parity
    # x -> 3x -> parity
    # -------------------------------------------------
    elif task_name == "int-times3-parity":
        for x in range(1, N_examples + 1):
            Fx = 3 * x
            GFx = "even" if Fx % 2 == 0 else "odd"

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 5) int-plus5-parity
    # x -> x+5 -> parity
    # -------------------------------------------------
    elif task_name == "int-plus5-parity":
        for x in range(1, N_examples + 1):
            Fx = x + 5
            GFx = "even" if Fx % 2 == 0 else "odd"

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 6) int-times3-str
    # x -> 3x -> English words
    # -------------------------------------------------
    elif task_name == "int-times3-str":
        for x in range(1, N_examples + 1):
            Fx = 3 * x
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 7) int-plus5-str
    # x -> x+5 -> English words
    # -------------------------------------------------
    elif task_name == "int-plus5-str":
        for x in range(1, N_examples + 1):
            Fx = x + 5
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 8) int-mod4-season
    # x -> mod 4 -> season
    # 1->Spring, 2->Summer, 3->Fall, 4->Winter
    # -------------------------------------------------
    elif task_name == "int-mod4-season":
        season_map = {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter",
        }

        for x in range(1, N_examples + 1):
            Fx = ((x - 1) % 4) + 1
            GFx = season_map[Fx]

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 9) int-plus2-str
    # x -> x+2 -> English words
    # -------------------------------------------------
    elif task_name == "int-plus2-str":
        for x in range(1, N_examples + 1):
            Fx = x + 2
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 10) int-plus8-str
    # x -> x+8 -> English words
    # -------------------------------------------------
    elif task_name == "int-plus8-str":
        for x in range(1, N_examples + 1):
            Fx = x + 8
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 11) int-times2-str
    # x -> 2x -> English words
    # -------------------------------------------------
    elif task_name == "int-times2-str":
        for x in range(1, N_examples + 1):
            Fx = 2 * x
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 12) int-mod9-str
    # x -> mod 9 -> English words
    # use 1..9 cyclically
    # -------------------------------------------------
    elif task_name == "int-mod9-str":
        for x in range(1, N_examples + 1):
            Fx = ((x - 1) % 9) + 1
            GFx = num2words(Fx)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 13) int-plus2-parity
    # x -> x+2 -> parity
    # -------------------------------------------------
    elif task_name == "int-plus2-parity":
        for x in range(1, N_examples + 1):
            Fx = x + 2
            GFx = "even" if Fx % 2 == 0 else "odd"

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 14) int-plus8-parity
    # x -> x+8 -> parity
    # -------------------------------------------------
    elif task_name == "int-plus8-parity":
        for x in range(1, N_examples + 1):
            Fx = x + 8
            GFx = "even" if Fx % 2 == 0 else "odd"

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x),
                "Fx": str(Fx),
                "Gx": None,
                "GFx": GFx,
                "FGx": None,
            })
            next_global_row_idx += 1

    # -------------------------------------------------
    # 15) author-birthyear-*
    # source task: book-author-birthyear
    # x = old Fx
    # Fx = old GFx
    # -------------------------------------------------
    elif task_name in {"author-birthyear-str", "author-birthyear-times-two"}:
        source_task = "book-author-birthyear"
        if source_task not in rows_by_task:
            raise ValueError(f"Source task not found in rows_by_task: {source_task}")

        for old in rows_by_task[source_task]:
            x_val = old["Fx"]       # author
            Fx_val = old["GFx"]     # birthyear

            try:
                year_int = _safe_int(Fx_val)
            except Exception:
                continue

            if task_name == "author-birthyear-str":
                GFx_val = num2words(year_int, to="year")
            else:  # author-birthyear-times-two
                GFx_val = str(2 * year_int)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x_val),
                "Fx": str(Fx_val),
                "Gx": None,
                "GFx": GFx_val,
                "FGx": None,
            })
            next_global_row_idx += 1

        new_rows = _deduplicate_examples_keep_first(new_rows)

    # -------------------------------------------------
    # 16) artist-birthyear-*
    # source task: song-artist-birthyear
    # -------------------------------------------------
    elif task_name in {"artist-birthyear-str", "artist-birthyear-times-two"}:
        source_task = "song-artist-birthyear"
        if source_task not in rows_by_task:
            raise ValueError(f"Source task not found in rows_by_task: {source_task}")

        for old in rows_by_task[source_task]:
            x_val = old["Fx"]       # artist
            Fx_val = old["GFx"]     # birthyear

            try:
                year_int = _safe_int(Fx_val)
            except Exception:
                continue

            if task_name == "artist-birthyear-str":
                GFx_val = num2words(year_int, to="year")
            else:  # artist-birthyear-times-two
                GFx_val = str(2 * year_int)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x_val),
                "Fx": str(Fx_val),
                "Gx": None,
                "GFx": GFx_val,
                "FGx": None,
            })
            next_global_row_idx += 1

        new_rows = _deduplicate_examples_keep_first(new_rows)

    # -------------------------------------------------
    # 17) director-birthyear-*
    # source task: movie-director-birthyear
    # -------------------------------------------------
    elif task_name in {"director-birthyear-str", "director-birthyear-times-two"}:
        source_task = "movie-director-birthyear"
        if source_task not in rows_by_task:
            raise ValueError(f"Source task not found in rows_by_task: {source_task}")

        for old in rows_by_task[source_task]:
            x_val = old["Fx"]       # director
            Fx_val = old["GFx"]     # birthyear

            try:
                year_int = _safe_int(Fx_val)
            except Exception:
                continue

            if task_name == "director-birthyear-str":
                GFx_val = num2words(year_int, to="year")
            else:  # director-birthyear-times-two
                GFx_val = str(2 * year_int)

            new_rows.append({
                "global_row_idx": next_global_row_idx,
                "task": task_name,
                "x": str(x_val),
                "Fx": str(Fx_val),
                "Gx": None,
                "GFx": GFx_val,
                "FGx": None,
            })
            next_global_row_idx += 1

        new_rows = _deduplicate_examples_keep_first(new_rows)

    else:
        raise ValueError(f"Task not implemented: {task_name}")

    # Print random 20 examples for sanity check
    n_print = min(20, len(new_rows))
    print(f"\nGenerated {len(new_rows)} rows for task '{task_name}'")
    if n_print > 0:
        print(f"Random {n_print} examples:")
        for r in rng.sample(new_rows, n_print):
            print(r)

    return new_rows


def generate_all_new_multihop_datasets(task_names,rows_by_task, N_examples=500, seed=42):

    all_new_rows_by_task = {}
    for task_name in task_names:
        all_new_rows_by_task[task_name] = generate_new_multihop_dataset_tasks(
            task_name=task_name,
            rows_by_task=rows_by_task,
            N_examples=N_examples,
            seed=seed,
        )
    return all_new_rows_by_task


# ============================================================
# Load dataset
# ============================================================
from num2words import num2words

# only evaluate the new string-version tasks
#all_tasks = [
#        "director-birthyear-str",
#        "director-birthyear-times-two",
#        "artist-birthyear-str",
#        "artist-birthyear-times-two",
#        "author-birthyear-str",
#        "author-birthyear-times-two",
#        "num-mod7-weekday",
#        "num-mod12-month",
#"int-plus2-str",
#        "int-plus8-str",
#        "int-times2-str",
#        "int-mod9-str",
#        "int-plus2-parity",
#        "int-plus8-parity",
#    ]
# ============================================================
# Load dataset
# ============================================================
dataset = load_dataset("apoorvkh/composing-functions")
train_data = dataset["train"]

rows = []
for i in range(len(train_data)):
    rows.append({
        "global_row_idx": i,
        "task": train_data[i]["task"],
        "x": train_data[i]["x"],
        "Fx": train_data[i]["Fx"],
        "Gx": train_data[i]["Gx"],
        "GFx": train_data[i]["GFx"],
        "FGx": train_data[i]["FGx"],
    })

old_rows_by_task = defaultdict(list)
for r in rows:
    old_rows_by_task[r["task"]].append(r)

orig_tasks = sorted(old_rows_by_task.keys())
print(orig_tasks)


new_tasks = [
        "int-plus5-parity",
        "int-plus2-str",
        "int-plus8-str",
        "int-plus5-str",
        "artist-birthyear-times-two",
    ]
# task2id/id2task are built after all_tasks is assembled below




prompt_types = ["Qx_Fx", "QFx_GFx", "Qx_GFx"]
prompt_type2id = {p: i for i, p in enumerate(prompt_types)}
id2prompt_type = {i: p for p, i in prompt_type2id.items()}



rows_by_task = generate_all_new_multihop_datasets(
    new_tasks,
    old_rows_by_task,
    N_examples=1000,
    seed=42,
)
# merge orig task rows so rows_by_task covers everything
rows_by_task.update(old_rows_by_task)

# ============================================================
# Build eval examples
# ============================================================
def _sample_context_for_query(task_rows, query_pos, n_icl, seed):
    """
    Sample n_icl context examples from task_rows excluding the query example itself.
    Sampling is deterministic for each query position.
    """
    candidate_positions = [i for i in range(len(task_rows)) if i != query_pos]
    if len(candidate_positions) < n_icl:
        return None

    rng = random.Random(seed + 1000003 * query_pos)
    sampled_positions = rng.sample(candidate_positions, n_icl)
    sampled_positions.sort()  # stable prompt ordering
    return [task_rows[i] for i in sampled_positions]


def build_examples_for_task(task_name, task_rows, n_icl=10, n_queries=None, seed=0):
    """
    For each task:
      - each query gets its own randomly sampled n_icl context examples
      - if n_queries is None, use all examples in the task as queries
      - otherwise use the first n_queries eligible query examples
      - for each query row, create 3 prompt types
    """
    if len(task_rows) <= n_icl:
        return []

    if n_queries is None:
        query_positions = list(range(len(task_rows)))
    else:
        query_positions = list(range(min(n_queries, len(task_rows))))

    out = []
    kept_query_idx = 0

    for query_pos in query_positions:
        query_row = task_rows[query_pos]

        context = _sample_context_for_query(
            task_rows=task_rows,
            query_pos=query_pos,
            n_icl=n_icl,
            seed=seed,
        )
        if context is None:
            continue

        pe = PromptExample(context=context, query=query_row)

        base_meta = {
            "task": task_name,
            "task_id": task2id[task_name],
            "local_query_idx": kept_query_idx,
            "global_row_idx": query_row["global_row_idx"],
            "query_fields": {
                "x": query_row["x"],
                "Fx": query_row["Fx"],
                "Gx": query_row["Gx"],
                "GFx": query_row["GFx"],
                "FGx": query_row["FGx"],
            },
            "context_global_row_idxs": [ex["global_row_idx"] for ex in context],
        }

        out.append({
            **base_meta,
            "prompt_type": "Qx_Fx",
            "prompt_type_id": prompt_type2id["Qx_Fx"],
            "target": " " + query_row["Fx"],
            "input": pe.get_prompt(query_type="x", pred_type="Fx", trailing_space=True),
        })

        out.append({
            **base_meta,
            "prompt_type": "QFx_GFx",
            "prompt_type_id": prompt_type2id["QFx_GFx"],
            "target": " " + query_row["GFx"],
            "input": pe.get_prompt(query_type="Fx", pred_type="GFx", trailing_space=True),
        })

        out.append({
            **base_meta,
            "prompt_type": "Qx_GFx",
            "prompt_type_id": prompt_type2id["Qx_GFx"],
            "target": " " + query_row["GFx"],
            "input": pe.get_prompt(query_type="x", pred_type="GFx", trailing_space=True),
        })

        kept_query_idx += 1

    return out
all_tasks = new_tasks + orig_tasks
task2id = {t: i for i, t in enumerate(all_tasks)}
id2task = {i: t for t, i in task2id.items()}
dataset_list = []
for task_name in all_tasks:
    task_examples = build_examples_for_task(
        task_name=task_name,
        task_rows=rows_by_task[task_name],
        n_icl=N_ICL,
        n_queries=N_QUERIES_PER_TASK,
        seed=args.seed,
    )
    dataset_list.extend(task_examples)

print(f"Total saved examples: {len(dataset_list)}")
print(f"Num tasks: {len(all_tasks)}")
print(f"Prompt types: {prompt_types}")
print(f"n_queries_per_task: {N_QUERIES_PER_TASK} (None means all examples)")


# ============================================================
# Matching
# ============================================================
def normalize_text(s: str) -> str:
    return s.strip().lower()

def match_pred_target(pred: str, target: str) -> bool:
    pred_n = normalize_text(pred)
    target_n = normalize_text(target)
    return pred_n == target_n

def is_nontrivial_prefix(a: str, b: str) -> bool:
    a = a.lower().strip()
    b = b.lower().strip()
    return len(a) > 0 and b.startswith(a)

def match_bidir_prefix(pred: str, target: str) -> bool:
    return is_nontrivial_prefix(pred, target) or is_nontrivial_prefix(target, pred)


# ============================================================
# Save helpers
# ============================================================
def create_memmaps(out_dir, N, n_layers, d_model, d_vocab):
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "task_id": os.path.join(out_dir, "task_id.npy"),
        "prompt_type_id": os.path.join(out_dir, "prompt_type_id.npy"),
        "local_query_idx": os.path.join(out_dir, "local_query_idx.npy"),
        "global_row_idx": os.path.join(out_dir, "global_row_idx.npy"),
        "match": os.path.join(out_dir, "match.npy"),
        "resid_last_all_layers": os.path.join(out_dir, "resid_last_all_layers.memmap"),
    }

    task_id = np.empty((N,), dtype=np.int32)
    prompt_type_id = np.empty((N,), dtype=np.int16)
    local_query_idx = np.empty((N,), dtype=np.int32)
    global_row_idx = np.empty((N,), dtype=np.int32)
    match = np.empty((N,), dtype=np.uint8)

    resid_last = np.memmap(
        paths["resid_last_all_layers"],
        mode="w+",
        dtype=DTYPE,
        shape=(N, n_layers, d_model),
    )

    return {
        "task_id": task_id,
        "prompt_type_id": prompt_type_id,
        "local_query_idx": local_query_idx,
        "global_row_idx": global_row_idx,
        "match": match,
        "resid_last": resid_last,
        "paths": paths,
    }


def finalize_small_arrays(store):
    np.save(store["paths"]["task_id"], store["task_id"])
    np.save(store["paths"]["prompt_type_id"], store["prompt_type_id"])
    np.save(store["paths"]["local_query_idx"], store["local_query_idx"])
    np.save(store["paths"]["global_row_idx"], store["global_row_idx"])
    np.save(store["paths"]["match"], store["match"])


# ============================================================
# Residual extraction
# ============================================================
@torch.no_grad()
def get_all_layers_last_resid_pre(model, input_ids_1d):
    """
    input_ids_1d: [T]
    returns: [n_layers, d_model]
    """
    n_layers = model.cfg.n_layers
    tokens = input_ids_1d.unsqueeze(0)

    last_by_layer = [None] * n_layers

    def make_hook(layer_idx):
        def hook_fn(act, hook):
            # With a sharded model (device_map="auto"), different layers'
            # activations can live on different GPUs, so move each to CPU
            # here rather than failing later when stacking across layers.
            last_by_layer[layer_idx] = act[0, -1, :].detach().cpu()
        return hook_fn

    hooks = [(f"blocks.{l}.hook_resid_pre", make_hook(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(tokens, fwd_hooks=hooks, return_type=None)

    last_stack = torch.stack(last_by_layer, dim=0)
    return last_stack


# ============================================================
# Generation + first-step logits
# ============================================================
@torch.no_grad()
def generate_one_completion_and_first_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    tokens = enc["input_ids"].to(device)

    # Use the bridge's cached generate() instead of a manual no-KV-cache loop:
    # recomputing a full forward over the ever-growing sequence at every step
    # is O(max_new_tokens) full-length forwards, which is brutal on top of
    # OLMo's forced eager attention (unfused, unlike Qwen/Llama's default
    # SDPA). use_past_kv_cache=True does one prefill + single-token forwards.
    gen_out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_past_kv_cache=True,
        stop_at_eos=True,
        output_logits=True,
        return_type="tokens",
        verbose=False,
    )
    tokens = gen_out.sequences
    first_step_logits = gen_out.logits[0][0].detach()

    decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
    pred = decoded[len(prompt):].split("\n")[0].strip()

    return pred, first_step_logits


# ============================================================
# Main eval + save
# ============================================================
@torch.no_grad()
def evaluate_and_save_no_batch(
    model,
    tokenizer,
    dataset_list,
    out_dir,
    max_new_tokens,
    device,
    print_every=50,
    print_first_k=10,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.to(device)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab
    N = len(dataset_list)

    store = create_memmaps(out_dir, N, n_layers, d_model, d_vocab)

    pred_path = os.path.join(out_dir, "predictions.jsonl")
    pred_f = open(pred_path, "w", encoding="utf-8")

    correct = 0
    print(f"Evaluating and saving to {out_dir} ...", flush=True)
    for row, ex in enumerate(tqdm(dataset_list, desc="[eval+save composing-functions]")):
        prompt = ex["input"]

        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids_1d = enc["input_ids"][0].to(device)
        last_stack = get_all_layers_last_resid_pre(model, input_ids_1d)
        store["resid_last"][row, :, :] = last_stack.to(torch.float32).cpu().numpy().astype(DTYPE, copy=False)

        pred, first_logits_t = generate_one_completion_and_first_logits(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        target = ex["target"]
        ok = match_pred_target(pred, target)

        # first-token accuracy: does gold start with the argmax token at step 0?
        first_tok_id = int(torch.argmax(first_logits_t).item())
        first_tok_text = tokenizer.decode([first_tok_id], skip_special_tokens=False).strip().lower()
        gold_norm = target.strip().lower()
        ok_first_token = bool(len(first_tok_text) > 0 and gold_norm.startswith(first_tok_text))

        # bidirectional prefix accuracy: pred starts with gold OR gold starts with pred
        ok_bidir = bool(match_bidir_prefix(pred, target))

        store["task_id"][row] = int(ex["task_id"])
        store["prompt_type_id"][row] = int(ex["prompt_type_id"])
        store["local_query_idx"][row] = int(ex["local_query_idx"])
        store["global_row_idx"][row] = int(ex["global_row_idx"])
        store["match"][row] = 1 if ok else 0

        correct += int(ok)

        record = {
            "row": row,
            "task": ex["task"],
            "task_id": int(ex["task_id"]),
            "prompt_type": ex["prompt_type"],
            "prompt_type_id": int(ex["prompt_type_id"]),
            "local_query_idx": int(ex["local_query_idx"]),
            "global_row_idx": int(ex["global_row_idx"]),
            "context_global_row_idxs": ex["context_global_row_idxs"],
            "input": ex["input"],
            "target": target,
            "prediction": pred,
            "match": bool(ok),
            "match_first_token": ok_first_token,
            "match_bidir_prefix": ok_bidir,
            "query_fields": ex["query_fields"],
        }


        if row < 20:
            print(json.dumps(record, ensure_ascii=False), flush=True)

        json.dump(record, pred_f, ensure_ascii=False)
        pred_f.write("\n")

        if row % 30 == 0:
            pred_f.flush()

        if row > 0 and row % 1000 == 0:
            store["resid_last"].flush()

    store["resid_last"].flush()
    pred_f.close()

    finalize_small_arrays(store)

    acc = correct / N if N else 0.0

    summary = {
        "N": N,
        "acc": float(acc),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "d_vocab": int(d_vocab),
        "dtype": args.dtype,
        "model_name": MODEL_NAME,
        "max_new_tokens": int(max_new_tokens),
        "n_icl": int(N_ICL),
        "n_queries_per_task": None if N_QUERIES_PER_TASK is None else int(N_QUERIES_PER_TASK),
        "prompt_types": prompt_types,
        "tasks": all_tasks,
        "task2id": task2id,
        "prompt_type2id": prompt_type2id,
        "paths": {
            **store["paths"],
            "predictions_jsonl": pred_path,
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. acc={acc:.4f}  N={N}")
    print("Saved:", os.path.join(out_dir, "summary.json"))

    return summary


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = {
        "tasks": all_tasks,
        "task2id": task2id,
        "id2task": id2task,
        "prompt_types": prompt_types,
        "prompt_type2id": prompt_type2id,
        "id2prompt_type": id2prompt_type,
        "n_icl": N_ICL,
        "n_queries_per_task": None if N_QUERIES_PER_TASK is None else int(N_QUERIES_PER_TASK),
        "random_context_per_query": True,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    model, tokenizer = load_hooked_transformer_model(MODEL_NAME)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    _summary = evaluate_and_save_no_batch(
        model=model,
        tokenizer=tokenizer,
        dataset_list=dataset_list,
        out_dir=OUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
        device=device,
        print_every=args.print_every,
        print_first_k=args.print_first_k,
    )


"""
later we can reconstruct:
mask = (
    (task_id == task2id["antonym-spanish"]) &
    (prompt_type_id == prompt_type2id["Qx_GFx"]) &
    (local_query_idx == 3)
)
"""