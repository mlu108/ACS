from __future__ import annotations
#from demo_load_datasets_model import *
#from demo_helpers import *
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from typing import Dict, Any, List, Optional
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import time, json, csv, argparse
import torch
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
#from demo_load_datasets_model import *
#from helpers import *



import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt






from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from typing import Dict, Any, List, Optional

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression

from pathlib import Path

import time, json


import torch




import matplotlib as mpl


def load_relation_by_index_lang_from_jsonl(jsonl_path):
    """
    Returns:
        relation_by_index_lang[(index, lang)] = relation
    Only keeps rows that actually contain a 'relation' field.
    """
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "relation" not in ex:
                continue
            idx = int(ex["index"])
            lang = str(ex["lang"])
            out[(idx, lang)] = ex["relation"]
    return out


def load_translation_vectors(
    run_dir: str,
    model,
    *,
    
    use_resid: str = "last",  # "last" uses resid_last_all_layers.memmap
    dtype_out=np.float32,
    prefer_filled_N: bool = True,
    verbose: bool = True,
    
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute (or load cached) mean last-prompt-token resid stream vectors across all layers,
    separately for each language, using the *saved matched* subset (MAX_MATCH_PER_LANG per language).

    Expects run_dir to contain outputs from your capped saving script:
      - resid_last_all_layers.memmap  (shape [Ncap, n_layers, d_model])
      - resid_mean_all_layers.memmap  (optional)
      - indices.npy, lang_id.npy, match.npy
      - filled_N.npy (optional but recommended)
      - meta.json (optional; not required)

    Saves to:
      translation_vectors_dir/layer{layer:02d}/{lang}.npy

    Returns:
      vecs[layer][lang] -> np.ndarray [d_model]
    """

    assert use_resid in ("last", "mean")
    translation_vectors_dir = os.path.join(run_dir,"translation_vectors_dir")
    # -----------------------
    # Load small arrays
    # -----------------------
    idx_path = os.path.join(run_dir, "indices.npy")
    lang_path = os.path.join(run_dir, "lang_id.npy")
    match_path = os.path.join(run_dir, "match.npy")

    indices = np.load(idx_path)          # [Ncap]
    lang_id = np.load(lang_path)         # [Ncap]
    match = np.load(match_path)          # [Ncap]

    Ncap = int(indices.shape[0])

    filled_N = None
    filled_path = os.path.join(run_dir, "filled_N.npy")
    if prefer_filled_N and os.path.exists(filled_path):
        filled_N = int(np.load(filled_path)[0])
        if verbose:
            print(f"[load_translation_vectors] Using filled_N={filled_N} / Ncap={Ncap}")
    else:
        if verbose:
            print(f"[load_translation_vectors] No filled_N.npy (or prefer_filled_N=False). Using all Ncap={Ncap}")

    # -----------------------
    # Determine langs + mapping
    # -----------------------
    # Try to load meta.json if present; otherwise infer from lang_id array
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        valid_langs = meta["langs"]
        lang2id = meta["lang2id"]
    else:
        # fallback: infer set of ids present
        uniq = sorted(set(int(x) for x in lang_id.tolist() if int(x) >= 0))
        # We can't recover string codes without meta.json, so require it.
        raise FileNotFoundError(
            f"meta.json not found in {run_dir}. "
            f"Please keep meta.json so we can map lang_id -> lang string."
        )

    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)

    # -----------------------
    # Memmap load
    # -----------------------
    if use_resid == "last":
        resid_path = os.path.join(run_dir, "resid_last_all_layers.memmap")
    else:
        resid_path = os.path.join(run_dir, "resid_mean_all_layers.memmap")

    resid = np.memmap(
        resid_path,
        mode="r",
        dtype=np.float16 if str(getattr(model.cfg, "dtype", "")).endswith("16") else np.float16,
        shape=(Ncap, n_layers, d_model),
    )

    # only consider the filled prefix if we have it
    sl = slice(0, filled_N) if (filled_N is not None) else slice(None)

    indices_sl = indices[sl]
    lang_id_sl = lang_id[sl]
    match_sl = match[sl]

    # saved rows criterion
    saved_mask = (match_sl.astype(np.int64) == 1) & (indices_sl.astype(np.int64) >= 0) & (lang_id_sl.astype(np.int64) >= 0)

    # -----------------------
    # Compute / load cache
    # -----------------------
    vecs: Dict[int, Dict[str, np.ndarray]] = {l: {} for l in range(n_layers)}
    os.makedirs(translation_vectors_dir, exist_ok=True)

    for layer in range(n_layers):
        layer_dir = os.path.join(translation_vectors_dir, f"layer{layer:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for lang in valid_langs:
            out_path = os.path.join(layer_dir, f"{lang}.npy")

            if os.path.exists(out_path):
                vec = np.load(out_path).astype(dtype_out, copy=False)
                vecs[layer][lang] = vec
                continue

            lid = int(lang2id[lang])
            m = saved_mask & (lang_id_sl.astype(np.int64) == lid)
            rows = np.nonzero(m)[0]

            if rows.size == 0:
                # no saved examples (shouldn’t happen if you capped 100 each, but be safe)
                if verbose:
                    print(f"[warn] layer={layer:02d} lang={lang}: no saved rows; writing NaNs")
                vec = np.full((d_model,), np.nan, dtype=dtype_out)
                np.save(out_path, vec)
                vecs[layer][lang] = vec
                continue

            # IMPORTANT: rows are relative to the sliced arrays; map back into memmap indices:
            # If we sliced [0:filled_N], rows already align with the first part of resid.
            # So we can index resid[rows, layer, :]
            X = resid[rows, layer, :].astype(np.float32, copy=False)  # [n, d]
            vec = X.mean(axis=0).astype(dtype_out, copy=False)        # [d]

            np.save(out_path, vec)
            vecs[layer][lang] = vec

        if verbose and (layer % 4 == 0 or layer == n_layers - 1):
            print(f"[load_translation_vectors] finished layer {layer:02d}/{n_layers-1:02d}")

    return vecs



def _oscar_cache_path(oscar_cache_root: str, lang: str, layer: int, subspace_method: str,
                      var_prop: float, max_rows: int, oscar_split: str = 'all') -> Path:
    var_tag = str(var_prop).replace(".", "_")
    dir_name = f"{subspace_method}_VAR_{var_tag}"
    if oscar_split != 'all':
        dir_name = f"{dir_name}_{oscar_split}"
    return Path(oscar_cache_root) / dir_name / lang / f"layer{layer}_cap{max_rows}.npz"

def _save_oscar_subspace_npz(path: Path, Vk: np.ndarray, mu: np.ndarray, meta: dict, sv: np.ndarray | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(Vk=Vk.astype(np.float32), mu=mu.astype(np.float32))
    if sv is not None:
        arrays["sv"] = sv.astype(np.float32)
    np.savez_compressed(path, **arrays, **{f"__meta__{k}": str(v) for k, v in meta.items()})

def _load_oscar_subspace_npz(path: Path):
    npz = np.load(path, allow_pickle=False)
    Vk = npz["Vk"]
    mu = npz["mu"]
    sv = npz["sv"] if "sv" in npz.files else None
    meta = {}
    for k in npz.files:
        if k.startswith("__meta__"):
            meta[k.replace("__meta__", "")] = npz[k].item()
    return Vk, mu, sv, meta


def compute_oscar_subspace_given_method_and_mean_vec(
    resids_path: Path,
    subspace_method: str = 'SVD',
    var_prop: float = 0.90,
    max_rows: int = None,
    verbose: bool = True,
    row_start: int = 0,
    row_end: Optional[int] = None,
):
    """
    Compute OSCAR subspace using either SVD or PCA.

    Returns
    -------
    Vk : np.ndarray, shape [d, k]
        Orthonormal basis capturing >= var_prop variance.
    mu : np.ndarray, shape [d]
        Mean vector.
    meta : dict
        {'k': k, 'n_used': N_used, 'd': d, 'method': subspace_method}
    """

    t0 = time.time()
    emb = torch.load(resids_path, map_location="cpu")

    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)

    N, d = emb.shape

    # Apply split window FIRST (for dev/test Oscar splits)
    if row_start != 0 or row_end is not None:
        row_end_actual = row_end if row_end is not None else N
        emb = emb[row_start:row_end_actual]
        N = emb.shape[0]

    # -----------------------------------------------------
    # Row subsampling
    # -----------------------------------------------------
    if max_rows is not None and N > max_rows:
        emb = emb[:max_rows]
        N_used = max_rows
    else:
        N_used = N

    # -----------------------------------------------------
    # Centering
    # -----------------------------------------------------
    mu = emb.mean(dim=0, keepdim=True)
    centered = emb - mu

    # -----------------------------------------------------
    # SVD METHOD
    # -----------------------------------------------------
    if subspace_method.upper() == "SVD":
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        sv2 = S ** 2
        cum = torch.cumsum(sv2, dim=0) / sv2.sum()
        k = int((cum >= var_prop).nonzero(as_tuple=True)[0][0].item() + 1)

        Vk = Vt[:k, :].T.contiguous()   # [d, k]

        if verbose:
            print(f"   [SVD] rows={N_used}, d={d}, var≥{var_prop} → k={k}  "
                  f"({time.time() - t0:.2f}s)")

        return (
            Vk.detach().cpu().numpy(),
            mu.squeeze(0).detach().cpu().numpy(),
            S[:k].detach().cpu().numpy(),
            {"k": k, "n_used": N_used, "d": d, "method": "SVD"}
        )

    # -----------------------------------------------------
    # PCA METHOD
    # -----------------------------------------------------
    elif subspace_method.upper() == "PCA":
        # Use float64 for numerical stability
        centered = centered.to(torch.float64)

        if N_used <= 1:
            raise ValueError(f"PCA needs at least 2 rows, got N_used={N_used}")

        if not torch.isfinite(centered).all():
            bad = (~torch.isfinite(centered)).sum().item()
            raise ValueError(f"centered contains {bad} non-finite values before PCA")

        # Covariance matrix: [d, d]
        cov = (centered.T @ centered) / (N_used - 1)

        # Force exact symmetry to avoid backend eigh issues from tiny asymmetries
        cov = 0.5 * (cov + cov.T)

        if not torch.isfinite(cov).all():
            bad = (~torch.isfinite(cov)).sum().item()
            raise ValueError(f"covariance contains {bad} non-finite values")

        try:
            # Eigen decomposition (ascending)
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError as e:
            if verbose:
                print(f"   [PCA] torch.linalg.eigh failed, falling back to numpy.linalg.eigh: {e}")

            cov_np = cov.detach().cpu().numpy()
            eigvals_np, eigvecs_np = np.linalg.eigh(cov_np)
            eigvals = torch.from_numpy(eigvals_np).to(torch.float64)
            eigvecs = torch.from_numpy(eigvecs_np).to(torch.float64)

        # Reverse (largest first)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        # Clamp tiny negative eigenvalues from numerical noise
        eigvals = torch.clamp(eigvals, min=0.0)

        total_var = eigvals.sum()
        if total_var <= 0 or not torch.isfinite(total_var):
            raise ValueError(f"Invalid total variance in PCA: {total_var}")

        cum = torch.cumsum(eigvals, dim=0) / total_var
        k = int((cum >= var_prop).nonzero(as_tuple=True)[0][0].item() + 1)
        k = max(1, min(k, eigvecs.shape[1]))

        Vk = eigvecs[:, :k].contiguous()   # [d, k]

        if verbose:
            sym_err = (cov - cov.T).abs().max().item()
            print(
                f"   [PCA] rows={N_used}, d={d}, var≥{var_prop} → k={k}  "
                f"(sym_err={sym_err:.3e}, {time.time() - t0:.2f}s)"
            )

        return (
            Vk.detach().cpu().numpy(),
            mu.squeeze(0).to(torch.float64).detach().cpu().numpy(),
            {"k": k, "n_used": N_used, "d": d, "method": "PCA"}
        )
    # -----------------------------------------------------
    # Unsupported method
    # -----------------------------------------------------
    else:
        raise ValueError(
            f"subspace_method must be 'SVD' or 'PCA', got: {subspace_method}"
        )


def oscar_W_cached(lang: str, layer: int, subspace_method: str, var_prop: float,
                    oscar_resids_root: str,
                    disk_cache_root: str,
                    max_rows: int = 8000,
                    verbose: bool = True,
                    oscar_split: str = 'all'):
    """
    Returns (WL, mu) where WL is (d,k). Uses on-disk cache; computes and saves if missing.
    Cache key includes (lang, layer, var_prop, max_rows, oscar_split).
    subspace_method=['SVD','PCA']
    oscar_split: 'all' uses rows [0:max_rows]; 'dev' uses [0:max_rows//2]; 'test' uses [max_rows//2:max_rows].
    """
    half = max_rows // 2
    if oscar_split == 'dev':
        row_start, row_end = 0, half
    elif oscar_split == 'test':
        row_start, row_end = half, max_rows
    else:
        row_start, row_end = 0, None

    cache_path = _oscar_cache_path(disk_cache_root, lang, layer, subspace_method, var_prop, max_rows, oscar_split)
    if cache_path.exists():
        Vk, mu, sv, meta = _load_oscar_subspace_npz(cache_path)
        if sv is not None or Vk is None:
            k = Vk.shape[1] if Vk is not None else (meta.get("rank_k") or meta.get("k"))
            if verbose:
                print(f"    [cache-hit] {lang} L{layer} VAR={var_prop} cap={max_rows} → {cache_path.name}  | rank={k}")
            return Vk, mu, sv
        if verbose:
            print(f"    [cache-stale] {lang} L{layer}: sv missing, recomputing → {cache_path.name}")
    else:
        if verbose:
            print(f"{cache_path} doesn't exist")

    src = Path(oscar_resids_root) / lang / f"LAYER_{layer}_{lang}.pt"
    if not src.exists():
        if verbose:
            print(f"    [MISS] {lang} L{layer} residuals missing: {src}")
        return None, None, None

    if verbose:
        split_info = f", split={oscar_split}" if oscar_split != 'all' else ''
        print(f"    [build] {lang} L{layer} from {src.name} (cap={max_rows}, VAR={var_prop}{split_info})")
    Vk, mu, sv, stats = compute_oscar_subspace_given_method_and_mean_vec(
        src, subspace_method, var_prop, max_rows, verbose=verbose,
        row_start=row_start, row_end=row_end,
    )

    k = int(Vk.shape[1]) if Vk is not None else int(stats.get("rank_k") or stats.get("k") or 0)
    meta = {
        "language": lang,
        "layer": layer,
        "var_prop": var_prop,
        "max_rows": max_rows,
        **(stats or {}),
        "rank_k": k,
    }
    _save_oscar_subspace_npz(cache_path, Vk, mu, meta, sv=sv)
    if verbose:
        print(f"    [save] {cache_path}  | rank={k}")
    return Vk, mu, sv


def _oscar_global_mean_cache_path(disk_cache_root: str, layer: int, max_rows: int) -> Path:
    return Path(disk_cache_root) / "global_means" / f"layer{layer}_cap{max_rows}.npz"


def _oscar_global_mean_cached(
    layer: int,
    all_full_langs: list,
    oscar_resids_root: str,
    disk_cache_root: str,
    max_rows: int = 8000,
    verbose: bool = True,
) -> np.ndarray:
    """Weighted grand mean of Oscar residuals across all languages at `layer`."""
    cache_path = _oscar_global_mean_cache_path(disk_cache_root, layer, max_rows)
    if cache_path.exists():
        npz = np.load(cache_path)
        if verbose:
            print(f"    [cache-hit] global mean L{layer} cap={max_rows} → {cache_path.name}")
        return npz["grand_mean"].astype(np.float64)

    total_sum = None
    total_count = 0
    for full_lang in tqdm(all_full_langs, desc=f"global mean L{layer}", leave=False):
        src = Path(oscar_resids_root) / full_lang / f"LAYER_{layer}_{full_lang}.pt"
        if not src.exists():
            if verbose:
                print(f"    [skip] global mean: {src} not found")
            continue
        emb = torch.load(src, map_location="cpu")
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        N = emb.shape[0]
        if max_rows is not None and N > max_rows:
            emb = emb[:max_rows]
            N = max_rows
        lang_sum = emb.to(torch.float64).sum(dim=0).numpy()
        total_sum = lang_sum if total_sum is None else total_sum + lang_sum
        total_count += N

    if total_sum is None or total_count == 0:
        raise RuntimeError(f"No Oscar residuals found for layer={layer} in {oscar_resids_root}")

    grand_mean = (total_sum / total_count).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, grand_mean=grand_mean)
    if verbose:
        print(f"    [save] global mean L{layer} → {cache_path}  (langs={len(all_full_langs)}, rows={total_count})")
    return grand_mean.astype(np.float64)


def _oscar_global_centered_cache_path(
    disk_cache_root: str, lang: str, layer: int, subspace_method: str,
    var_prop: float, max_rows: int, per_lang_center: bool,
) -> Path:
    var_tag = str(var_prop).replace(".", "_")
    center_tag = "globalcent_langcent" if per_lang_center else "globalcent_only"
    return Path(disk_cache_root) / f"{subspace_method}_VAR_{var_tag}_{center_tag}" / lang / f"layer{layer}_cap{max_rows}.npz"


def oscar_W_global_centered_cached(
    lang: str,
    layer: int,
    subspace_method: str,
    var_prop: float,
    oscar_resids_root: str,
    disk_cache_root: str,
    grand_mean: np.ndarray,
    per_lang_center: bool,
    max_rows: int = 8000,
    verbose: bool = True,
) -> tuple:
    """
    Globally-centered Oscar subspace.

    Globally centers lang's reps by grand_mean, then optionally per-lang centers
    before SVD. Returns (WL, muL) where muL is the per-lang mean of globally-centered
    data (None when per_lang_center=False).
    """
    cache_path = _oscar_global_centered_cache_path(
        disk_cache_root, lang, layer, subspace_method, var_prop, max_rows, per_lang_center
    )
    if cache_path.exists():
        Vk, mu, sv, meta = _load_oscar_subspace_npz(cache_path)
        if sv is not None or Vk is None:
            k = Vk.shape[1] if Vk is not None else (meta.get("rank_k") or meta.get("k"))
            if verbose:
                print(f"    [cache-hit] global-cent {lang} L{layer} per_lang_center={per_lang_center} → {cache_path.name} | rank={k}")
            return Vk, (mu if per_lang_center else None), sv
        if verbose:
            print(f"    [cache-stale] global-cent {lang} L{layer}: sv missing, recomputing → {cache_path.name}")

    src = Path(oscar_resids_root) / lang / f"LAYER_{layer}_{lang}.pt"
    if not src.exists():
        if verbose:
            print(f"    [MISS] global-cent {lang} L{layer}: {src} not found")
        return None, None

    emb = torch.load(src, map_location="cpu")
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    N = emb.shape[0]
    if max_rows is not None and N > max_rows:
        emb = emb[:max_rows]
        N = max_rows

    emb = emb.to(torch.float64)
    grand_mean_t = torch.from_numpy(np.asarray(grand_mean, dtype=np.float64))
    globally_centered = emb - grand_mean_t[None, :]

    lang_mean = globally_centered.mean(dim=0)
    if per_lang_center:
        data_for_svd = globally_centered - lang_mean[None, :]
    else:
        data_for_svd = globally_centered

    if subspace_method.upper() != "SVD":
        raise ValueError(f"oscar_W_global_centered_cached only supports SVD, got: {subspace_method}")

    U, S, Vt = torch.linalg.svd(data_for_svd, full_matrices=False)
    sv2 = S ** 2
    cum = torch.cumsum(sv2, dim=0) / sv2.sum()
    k = int((cum >= var_prop).nonzero(as_tuple=True)[0][0].item() + 1)
    Vk = Vt[:k, :].T.contiguous()
    Vk_np = Vk.detach().cpu().numpy().astype(np.float32)
    mu_np = lang_mean.detach().cpu().numpy().astype(np.float32)
    sv_np = S[:k].detach().cpu().numpy().astype(np.float32)

    meta = {
        "language": lang, "layer": layer, "var_prop": var_prop,
        "max_rows": max_rows, "per_lang_center": per_lang_center,
        "n_used": N, "rank_k": int(k),
    }
    _save_oscar_subspace_npz(cache_path, Vk_np, mu_np, meta, sv=sv_np)
    if verbose:
        print(f"    [save] global-cent {lang} L{layer} per_lang_center={per_lang_center} → {cache_path} | rank={k}")
    return Vk_np, (mu_np if per_lang_center else None), sv_np




abbr_to_full_LANGUAGE_CODE_MAP = {
    "en": "english",
    "zh": "chinese",
    "fr": "french",
    "ja": "japanese",
    "ko": "korean",
    "es": "spanish",
    "ca": "catalan",
    "hu": "hungarian",
    "nl": "dutch",
    "ru": "russian",
    "uk": "ukrainian",
    "vi": "vietnamese",
}

# -------------------------
# HELPERS
# -------------------------
def _orthonormalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """QR-orthonormalize columns; returns [d,k] with orthonormal columns (or empty)."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] == 0:
        return W
    if not np.isfinite(W).all():
        return W
    Q, R = np.linalg.qr(W)
    diag = np.abs(np.diag(R))
    keep = diag > eps
    if keep.sum() == 0:
        return W[:, :0]
    return Q[:, keep]

def task_dir_name(shot_tag, rep_kind, method, threshold):
    return f"klar_english_task_{shot_tag}_{rep_kind}_{method}_{threshold}"

def load_summary(run_dir: str, shot_tag: str) -> dict:
    """
    Your saver writes summary_{n_shot}shot.json. We'll map shot_tag -> n_shot.
    """
    n_shot = {"zeroshot": 0, "threeshot": 3}[shot_tag]
    p = os.path.join(run_dir, f"summary_{n_shot}shot.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def open_resid_memmap(run_dir: str, rep_kind: str, summary: dict):
    """
    resid_last_all_layers.memmap / resid_mean_all_layers.memmap
    shape: [N, n_layers, d_model]
    dtype: float16 (per your saver) :contentReference[oaicite:1]{index=1}
    """
    N = int(summary["N"])
    n_layers = int(summary["n_layers"])
    d_model = int(summary["d_model"])
    if rep_kind == "last":
        path = os.path.join(run_dir, "resid_last_all_layers.memmap")
    elif rep_kind == "mean":
        path = os.path.join(run_dir, "resid_mean_all_layers.memmap")
    else:
        raise ValueError(rep_kind)

    mm = np.memmap(path, mode="r", dtype=np.float16, shape=(N, n_layers, d_model))
    return mm  # do not copy entire thing

def load_run_small_arrays(run_dir: str):
    indices = np.load(os.path.join(run_dir, "indices.npy"))
    lang_id = np.load(os.path.join(run_dir, "lang_id.npy"))
    match   = np.load(os.path.join(run_dir, "match.npy"))
    return indices, lang_id, match


from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, json
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import pearsonr, spearmanr

@dataclass
class LanSepResult:
    lang: str
    n: int
    pos_rate: float
    beta1: float
    beta0: float
    auc: float


# -----------------------------
# Small utilities
# -----------------------------
def load_correct_token_targets_jsonl_by_index_lang(jsonl_path: str):
    """
    Returns: (index, lang) -> record
    """
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            idx = ex.get("index", None)
            lang = ex.get("lang", None)
            if idx is None or lang is None:
                continue
            out[(int(idx), str(lang))] = ex
    return out


def extract_targets_for_index_lang(
    indices: np.ndarray,
    lang_id: np.ndarray,
    targets_by_index_lang: dict,
    *,
    target_kind: str,   # "prob" or "rank"
    rank_kind: str = "rank_if_in_topk_else_lower_bound",
    top_k_examined: int = 5000,
):
    """
    Build y aligned to rows using (index, lang) key.


    """
    all_saved_langs = ["en","ca","es","fr","hu","ja","ko","nl","ru","uk","vi","zh"]

    lang2id = {l:i for i,l in enumerate(all_saved_langs)}
    id2lang = {i:l for l,i in lang2id.items()}

    y = np.full((indices.shape[0],), np.nan, dtype=np.float64)
  
    for i in range(indices.shape[0]):
        idx = int(indices[i])
        lang = id2lang[int(lang_id[i])]
        rec = targets_by_index_lang.get((idx, lang))
        if rec is None:
            continue

        if target_kind == "prob":
            # new field name
            p = rec.get("prob", None)
            if p is None:
                p = 0.0
            y[i] = float(p)
    
        
        elif target_kind == "rec_rank":
            if rank_kind == "rank_if_in_topk_else_nan":
                r = rec.get("rank", None)           
                if r is None:
                    y[i] = np.nan   # truly missing
                else:
                    y[i] = 1.0 / float(r)

            elif rank_kind == "rank_if_in_topk_else_lower_bound":
                r = rec.get("rank", None)

                if r is not None:
                    y[i] = 1.0 / float(r)
                else:
                    # use lower bound (e.g. topk+1)
                    rlb = rec.get("rank_lower_bound", None)
                    if rlb is None or rlb <= 0:
                        y[i] = 0.0   # safest fallback
                    else:
                        y[i] = 1.0 / float(rlb)

            else:
                raise ValueError(f"Unknown rank_kind={rank_kind}")
        else:
            raise ValueError(f"Unknown target_kind={target_kind}")

    return y
# MULTIPLE LAYERS TOGETHER PEARSON




# =============================
# Regression helpers
# =============================

def _cv_regression_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int = 5,
    seed: int = 0,
    ridge_alpha: float = 1.0,
) -> Tuple[float, float]:
    """Returns (mean_cv_r2, mean_cv_mse)."""
    X = np.asarray(X)
    y = np.asarray(y)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    r2s, mses = [], []

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=ridge_alpha, random_state=seed)),
    ])

    for tr, te in kf.split(X):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        mses.append(mean_squared_error(y[te], pred))

    return float(np.mean(r2s)), float(np.mean(mses))


def _cv_isotonic_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int = 5,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Monotone (non-decreasing) nonlinear regressor: IsotonicRegression.
    Returns (mean_cv_r2, mean_cv_mse).
    """
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    r2s, mses = [], []

    for tr, te in kf.split(x):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x[tr], y[tr])
        pred = iso.predict(x[te])

        r2s.append(r2_score(y[te], pred))
        mses.append(mean_squared_error(y[te], pred))

    return float(np.mean(r2s)), float(np.mean(mses))


# =============================
# Result dataclass
# =============================

@dataclass
class LanRegResult:
    lang: str          # e.g. "ja" or "ja|correct"
    n: int
    y_mean: float

    pearson_r_ortho: float
    pearson_p_ortho: float
    spearman_r_ortho: float
    spearman_p_ortho: float
    kendall_tau_ortho: float
    kendall_p_ortho: float

    cv_r2_ortho: float
    cv_mse_ortho: float
    cv_r2_isotonic: float
    cv_mse_isotonic: float

    pearson_r_base: float
    pearson_p_base: float
    spearman_r_base: float
    spearman_p_base: float
    cv_r2_base: float
    cv_mse_base: float



# =============================
# Fit regressors + plots
# =============================

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def fit_language_regressors_from_out(
    out: Dict[str, Dict[str, Any]],
    *,
    min_n: int = 100,
    cv_splits: int = 5,
    cv_seed: int = 0,
    ridge_alpha: float = 1.0,
    plot_each_language: bool = False,
    plot_dir: str = "plot_dir",
    target_kind: str = "",
    y_transform: str = "",
    rep_kind: str = "",
    lang_mode=None,
    task_mode=None,
) -> List[LanRegResult]:

    results: List[LanRegResult] = []

    if plot_each_language:
        os.makedirs(plot_dir, exist_ok=True)

    def _safe_pearson(x, y):
        if np.std(x) > 0 and np.std(y) > 0:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        return np.nan, np.nan

    def _safe_spearman(x, y):
        if np.std(x) > 0 and np.std(y) > 0:
            sp = spearmanr(x, y)
            # scipy returns either (corr, p) or an object depending on version;
            # handle both robustly
            corr = getattr(sp, "correlation", None)
            pval = getattr(sp, "pvalue", None)
            if corr is None:  # older tuple form
                corr, pval = sp
            return float(corr), float(pval)
        return np.nan, np.nan

    def _process_one_pack(lang_label: str, pack: Dict[str, Any]) -> Optional[LanRegResult]:
        x = np.asarray(pack["x_ortho"], dtype=np.float64)
        Xb = np.asarray(pack["X_base"], dtype=np.float64)
        y = np.asarray(pack["y"], dtype=np.float64)

        n = int(len(y))
        if n < min_n:
            return None

        # Correlations (ortho)
        pr_o, p_pr_o = _safe_pearson(x, y)
        sr_o, p_sr_o = _safe_spearman(x, y)

        # Kendall tau (ortho)
        if np.std(x) > 0 and np.std(y) > 0:
            kt = kendalltau(x, y, nan_policy="omit")
            tau_o = float(kt.correlation) if kt.correlation is not None else np.nan
            p_tau = float(kt.pvalue) if kt.pvalue is not None else np.nan
        else:
            tau_o, p_tau = np.nan, np.nan

        # Isotonic CV
        if np.isfinite(x).all() and np.isfinite(y).all() and (np.unique(x).size >= 2):
            r2_iso, mse_iso = _cv_isotonic_scores(x, y, cv_splits=cv_splits, seed=cv_seed)
        else:
            r2_iso, mse_iso = np.nan, np.nan

        # Baseline ridge (fit + correlation on in-sample preds)
        pipe_base = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=ridge_alpha, random_state=cv_seed)),
        ])
        pipe_base.fit(Xb, y)
        yhat_b = pipe_base.predict(Xb)

        pr_b, p_pr_b = _safe_pearson(yhat_b, y)
        sr_b, p_sr_b = _safe_spearman(yhat_b, y)

        # CV scores
        r2_o, mse_o = _cv_regression_scores(
            x.reshape(-1, 1), y,
            cv_splits=cv_splits, seed=cv_seed, ridge_alpha=ridge_alpha
        )
        r2_b, mse_b = _cv_regression_scores(
            Xb, y,
            cv_splits=cv_splits, seed=cv_seed, ridge_alpha=ridge_alpha
        )

        # Plot + Save (one figure per pack)
        if plot_each_language and np.isfinite(pr_o):
            plt.figure()
            plt.scatter(x, y, alpha=0.5)

            if np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(np.min(x), np.max(x), 200)
                ys = slope * xs + intercept
                plt.plot(xs, ys)

            plt.xlabel("x_ortho")
            plt.ylabel("y")
            plt.title(f"{lang_label} | n={n}\nPearson r={pr_o:.3f} (p={p_pr_o:.2g})")
            plt.grid(alpha=0.3)

            filename = f"{lang_label}.svg"
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path, format="svg", bbox_inches="tight")
            plt.close()
            print(f"Saved: {save_path}")

        return LanRegResult(
            lang=lang_label,
            n=n,
            y_mean=float(np.mean(y)),

            pearson_r_ortho=float(pr_o),
            pearson_p_ortho=float(p_pr_o),
            spearman_r_ortho=float(sr_o),
            spearman_p_ortho=float(p_sr_o),

            kendall_tau_ortho=float(tau_o),
            kendall_p_ortho=float(p_tau),

            cv_r2_ortho=float(r2_o),
            cv_mse_ortho=float(mse_o),
            cv_r2_isotonic=float(r2_iso),
            cv_mse_isotonic=float(mse_iso),

            pearson_r_base=float(pr_b),
            pearson_p_base=float(p_pr_b),
            spearman_r_base=float(sr_b),
            spearman_p_base=float(p_sr_b),

            cv_r2_base=float(r2_b),
            cv_mse_base=float(mse_b),
        )

    for L, pack in out.items():
        if isinstance(pack, dict) and ("splits" in pack) and isinstance(pack["splits"], dict):
            for split_name, split_pack in pack["splits"].items():
                lang_label = f"{L}|{split_name}"
                res = _process_one_pack(lang_label, split_pack)
                if res is not None:
                    results.append(res)
        else:
            res = _process_one_pack(L, pack)
            if res is not None:
                results.append(res)

    results.sort(key=lambda r: (r.cv_r2_ortho, r.cv_r2_base), reverse=True)
    return results

from collections import defaultdict


def _transform_y(y: np.ndarray, y_transform: str) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    if y_transform == "identity":
        return y
    if y_transform == "logit":
        eps = 1e-6
        yy = np.clip(y, eps, 1 - eps)
        return np.log(yy) - np.log(1 - yy)
    if y_transform == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unknown y_transform={y_transform}")

def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return np.full_like(v, np.nan, dtype=np.float64)
    return (v / n).astype(np.float64, copy=False)

def _cos_vec_vec(A: np.ndarray, b_unit: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    An = np.linalg.norm(A, axis=1)
    dot = A @ b_unit
    return dot / np.clip(An, eps, None)

def _cos_vec_subspace(A: np.ndarray, W_ortho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    AW = A @ W_ortho          # [n,k]
    proj = AW @ W_ortho.T     # [n,d]
    proj_n = np.linalg.norm(proj, axis=1)
    a_n = np.linalg.norm(A, axis=1)
    return proj_n / np.clip(a_n, eps, None)

def _per_basis_abs_cos(A: np.ndarray, W_ortho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-basis absolute cosines [n, k]: |a·w_j| / ||a||."""
    a_n = np.linalg.norm(A, axis=1, keepdims=True)
    return np.abs(A @ W_ortho) / np.clip(a_n, eps, None)

def _per_basis_angles(A: np.ndarray, W_ortho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-basis-vector angles [n, k]: arccos(|a_i · w_j| / ||a_i||)."""
    return np.arccos(np.clip(_per_basis_abs_cos(A, W_ortho, eps), 0.0, 1.0))

def _vec_subspace_mean_angle(A: np.ndarray, W_ortho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Mean angle to each basis vector: mean_j arccos(|a·w_j|/||a||)."""
    return _per_basis_angles(A, W_ortho, eps).mean(axis=1)

def _vec_subspace_min_angle(A: np.ndarray, W_ortho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min angle to any basis vector: min_j arccos(|a·w_j|/||a||)."""
    return _per_basis_angles(A, W_ortho, eps).min(axis=1)

def _vec_subspace_mean_weighted_angle(
    A: np.ndarray, W_ortho: np.ndarray, sv_weights: np.ndarray, eps: float = 1e-12,
    sv_weight_mode: str = "sv",
) -> np.ndarray:
    """Singular-value-weighted mean angle.
    "sv": sum_j(sv_j * angle_j) / sum(sv); "sv_squared": same with sv²; "none": direct sum_j(sv_j * angle_j)."""
    angles = _per_basis_angles(A, W_ortho, eps)              # [n,k]
    k = angles.shape[1]
    sv = np.asarray(sv_weights[:k], dtype=np.float64)
    if sv_weight_mode == "sv_squared":
        w = sv ** 2
        w_sum = w.sum()
        return (angles @ w) / w_sum if w_sum >= eps else angles.mean(axis=1)
    elif sv_weight_mode == "sv":
        w_sum = sv.sum()
        return (angles @ sv) / w_sum if w_sum >= eps else angles.mean(axis=1)
    else:  # "none": direct sv-weighted sum, no normalisation
        return angles @ sv

def load_binary_correct_targets_by_index_lang(jsonl_path: str):
    """
    JSONL lines must contain: index, lang, and match (or correct).
    Returns dict[(index, lang)] -> 0/1
    """
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            idx = int(ex["index"])
            lang = ex["lang"]
            if "match" in ex:
                val = 1 if bool(ex["match"]) else 0
            elif "correct" in ex:
                val = 1 if bool(ex["correct"]) else 0
            else:
                raise KeyError("binary_correct requires 'match' or 'correct' field in JSONL")
            out[(idx, lang)] = val
    return out


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from typing import Optional

def _softmax_rows(Z: np.ndarray, alpha: float = 5.0) -> np.ndarray:
    """Row-wise softmax with temperature-like sharpness alpha."""
    Z = np.asarray(Z, dtype=np.float64)
    Zs = alpha * Z
    Zs = Zs - np.max(Zs, axis=1, keepdims=True)
    E = np.exp(Zs)
    return E / np.clip(E.sum(axis=1, keepdims=True), 1e-12, None)


def aggregate_ortho_matrix(
    O: np.ndarray,
    op: str,
    *,
    soft_alpha: float = 5.0,
) -> np.ndarray:
    """
    O: [n, L] orthogonality per datapoint (rows) per layer (cols)
    returns x: [n]
    """
    op = str(op).lower()
    if op == "mean":
        return np.nanmean(O, axis=1)
    if op == "min":
        return np.nanmin(O, axis=1)
    if op == "max":
        return np.nanmax(O, axis=1)
    if op in ("softweighted_mean", "soft_weighted_mean", "softmean", "softmax_mean"):
        w = _softmax_rows(O, alpha=soft_alpha)
        return np.sum(w * O, axis=1)
    raise ValueError(f"Unknown op={op}. Expected mean/min/max/softweighted_mean")


def _split_lang_label(lang_label: str) -> Tuple[str, str]:
    """
    "ja|correct" -> ("ja","correct")
    "ja"         -> ("ja","all")
    """
    if "|" in lang_label:
        a, b = lang_label.split("|", 1)
        return a, b
    return lang_label, "all"


# -----------------------------
# Collect pearson across layers
# -----------------------------
def _ridge_oof_metrics_and_coeffs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float,
    cv_splits: int,
    seed: int,
) -> RidgeFitSummary:
    """
    Returns:
      - OOF pearson/spearman
      - mean CV r2/mse
      - coefficients mapped back to original feature scale (so you can interpret per-layer)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    yhat = np.full_like(y, np.nan, dtype=np.float64)
    r2s, mses = [], []

    for tr, te in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=ridge_alpha, random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        yhat[te] = pred
        r2s.append(r2_score(y[te], pred))
        mses.append(mean_squared_error(y[te], pred))

    m = np.isfinite(yhat) & np.isfinite(y)
    if m.sum() < 3 or np.std(yhat[m]) == 0 or np.std(y[m]) == 0:
        pr = np.nan
        sr = np.nan
    else:
        pr = float(pearsonr(yhat[m], y[m])[0])
        sr = float(spearmanr(yhat[m], y[m]).correlation)

    cv_r2 = float(np.mean(r2s)) if len(r2s) else np.nan
    cv_mse = float(np.mean(mses)) if len(mses) else np.nan

    # full-data coefficients (stable), then unscale them
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = Ridge(alpha=ridge_alpha, random_state=seed)
    reg.fit(Xs, y)

    coef_raw = reg.coef_.astype(np.float64)  # scaled-feature space

    scale = scaler.scale_.astype(np.float64)
    mean = scaler.mean_.astype(np.float64)

    coef_unscaled = coef_raw / scale
    intercept_unscaled = float(reg.intercept_ - np.sum(coef_raw * mean / scale))

    return RidgeFitSummary(
        pearson_oof=pr,
        spearman_oof=sr,
        cv_r2=cv_r2,
        cv_mse=cv_mse,
        coef_unscaled=coef_unscaled,
        intercept_unscaled=intercept_unscaled,
    )


def print_layer_coeffs_table(
    layers: List[int],
    coef: np.ndarray,
    *,
    topk: int = 10,
    precision: int = 5,
):
    layers = list(layers)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    assert len(layers) == len(coef), "layers and coef must align"

    print(f"{'layer':>6}  {'coef':>14}  {'sign':>6}  {'|coef|':>14}")
    for L, c in zip(layers, coef):
        sign = "+" if c >= 0 else "-"
        print(f"{L:6d}  {c:14.{precision}f}  {sign:>6}  {abs(c):14.{precision}f}")

    k = min(topk, len(coef))
    idx = np.argsort(np.abs(coef))[::-1][:k]
    print(f"\nTop-{k} layers by |coef|:")
    print(f"{'rank':>6}  {'layer':>6}  {'coef':>14}  {'|coef|':>14}")
    for r, j in enumerate(idx, start=1):
        print(f"{r:6d}  {layers[j]:6d}  {coef[j]:14.{precision}f}  {abs(coef[j]):14.{precision}f}")


from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np


@dataclass
class MultiLayerOrthoFitRow:
    lang: str                  # e.g. "ja" or "ja|correct"
    n: int
    pearson_oof: float
    spearman_oof: float
    cv_r2: float
    cv_mse: float
    coef_unscaled: Optional[np.ndarray] = None   # len(layers) if returned
    intercept_unscaled: Optional[float] = None

#outs_by_layer_from_english: Dict[str, Dict[str, Any]]
#outs_by_layer_to_english: Dict[str, Dict[str, Any]]
#Given these two, i want to use all layer's orthogoanlity values in these two dicts, 
#how can i process tehm and run fit_multilayer_ortho 
#or do i have to change fit_multilayer_ortho?
def fit_multilayer_ortho(
    outs_by_layer: Dict[str, Dict[str, Any]],
    
    *,
    layers: List[int],
    rep_kind: str,
    target_kind: str,
    y_transform: str,
    lang_mode,
    ridge_alpha: float = 1.0,
    cv_splits: int = 5,
    cv_seed: int = 0,
    min_n: int = 100,
    return_coeffs: bool = True,
    assert_same_y: bool = True,
    y_rtol: float = 1e-8,
    y_atol: float = 1e-10,

) -> List[MultiLayerOrthoFitRow]:
    """
    Fits Ridge on X_ortho_layers -> y for EACH language (and split if present).

    Returns a list of MultiLayerOrthoFitRow that contains metrics:
      - pearson_oof (OOF yhat vs y)
      - spearman_oof
      - cv_r2, cv_mse
    Optionally includes:
      - coef_unscaled (per-layer coefficients, len(layers))
      - intercept_unscaled

    Printing is intentionally NOT done here — so you can decide later:
      - print only metrics
      - print coeffs
      - save to csv, etc.
    """
    #organize the outs_by_layer
    first_layer = layers[0]
    base_out = outs_by_layer[first_layer]

    out: Dict[str, Dict[str, Any]] = {}

    def _stack_pooled(lang: str):
        x_cols = []
        y_ref = None

        for layer in layers:
            pack = outs_by_layer[layer].get(lang, None)
            if pack is None:
                return None  # missing this lang at some layer

            x = np.asarray(pack["x_ortho"], dtype=np.float64)
            y = np.asarray(pack["y"], dtype=np.float64)

            if y_ref is None:
                y_ref = y
           
            if y.shape != y_ref.shape or not np.allclose(y, y_ref, rtol=1e-8, atol=1e-10, equal_nan=False):
                raise ValueError(f"[multi-layer join] y mismatch for lang={lang} between layers {first_layer} and {layer}")

            x_cols.append(x)

        # shape: [n, m]
        X = np.stack(x_cols, axis=1)
        return X, y_ref

    def _stack_split(lang: str, split_name: str):
        x_cols = []
        y_ref = None

        for layer in layers:
            pack = outs_by_layer[layer].get(lang, None)
            if pack is None or "splits" not in pack:
                return None
            sp = pack["splits"].get(split_name, None)
            if sp is None:
                return None

            x = np.asarray(sp["x_ortho"], dtype=np.float64)
            y = np.asarray(sp["y"], dtype=np.float64)

            if y_ref is None:
                y_ref = y
            elif assert_same_y:
                if y.shape != y_ref.shape or not np.allclose(y, y_ref, rtol=y_rtol, atol=y_atol, equal_nan=False):
                    raise ValueError(f"[multi-layer join] y mismatch for lang={lang}|{split_name} between layers {first_layer} and {layer}")

            x_cols.append(x)

        X = np.stack(x_cols, axis=1)
        return X, y_ref

    # 3) build the multi-layer out by language
    for L, pack0 in base_out.items():
        # split mode
        if "splits" in pack0:
            splits_out = {}
            for split_name in pack0["splits"].keys():
                got = _stack_split(L, split_name)
                if got is None:
                    continue
                X, y = got
                if len(y) < min_n:
                    continue
                splits_out[split_name] = {
                    "X_ortho_layers": X,
                    "y": y,
                    "meta": {
                        **pack0["splits"][split_name].get("meta", {}),
                        "layers": layers,
                        "rep_kind": rep_kind,
                        "target_kind": target_kind,
                        "y_transform": y_transform,
                        "lang_mode": lang_mode,
                    },
                }
            if len(splits_out) == 0:
                continue
            out[L] = {
                "splits": splits_out,
                "meta": {
                    **pack0.get("meta", {}),
                    "layers": layers,
                    "rep_kind": rep_kind,
                    "target_kind": target_kind,
                    "y_transform": y_transform,
                    "lang_mode": lang_mode,
                    "separate_correct_incorrect_examples": True,
                },#loa
            }
        # pooled mode
        else:
            got = _stack_pooled(L)
            if got is None:
                continue
            X, y = got
            if len(y) < min_n:
                continue
            out[L] = {
                "X_ortho_layers": X,
                "y": y,
                "meta": {
                    **pack0.get("meta", {}),
                    "layers": layers,
                    "rep_kind": rep_kind,
                    "target_kind": target_kind,
                    "y_transform": y_transform,
                    "lang_mode": lang_mode,
                    "separate_correct_incorrect_examples": False,
                },
            }


    rows: List[MultiLayerOrthoFitRow] = []

    def _process_one(lang_label: str, X: np.ndarray, y: np.ndarray) -> Optional[MultiLayerOrthoFitRow]:
        if len(y) < min_n:
            return None

        fit = _ridge_oof_metrics_and_coeffs(
            X, y,
            ridge_alpha=ridge_alpha,
            cv_splits=cv_splits,
            seed=cv_seed,
        )

        if return_coeffs:
            coef = fit.coef_unscaled
            intercept = fit.intercept_unscaled
        else:
            coef = None
            intercept = None

        return MultiLayerOrthoFitRow(
            lang=lang_label,
            n=int(len(y)),
            pearson_oof=float(fit.pearson_oof),
            spearman_oof=float(fit.spearman_oof),
            cv_r2=float(fit.cv_r2),
            cv_mse=float(fit.cv_mse),
            coef_unscaled=coef,
            intercept_unscaled=intercept,
        )

    for L, pack in sorted(out.items()):
        # split mode
        if "splits" in pack:
            for split_name, sp in pack["splits"].items():
                X = sp["X_ortho_layers"]
                y = sp["y"]
                r = _process_one(f"{L}|{split_name}", X, y)
                if r is not None:
                    rows.append(r)
        # pooled mode
        else:
            X = pack["X_ortho_layers"]
            y = pack["y"]
            r = _process_one(L, X, y)
            if r is not None:
                rows.append(r)

    return rows


# -------------------------------
# Optional: printing helpers
# -------------------------------

def top5_layers_rank_aggregate(
    rows,
    *,
    layers,
    exclude_langs=("en",),
    k: int = 5,
    weight_by_n: bool = False,
    use_abs: bool = True,
) -> Tuple[List[Tuple[int, float]], Dict[int, float]]:
    """
    Rank-aggregate layer importance across languages using coef_unscaled.

    rows: output of fit_multilayer_ortho(... return_coeffs=True)
    layers: list/range of layers used as features (order must match coef vector)
    exclude_langs: languages to exclude (default excludes English)
    weight_by_n: if True, weight each language's votes by its n
    use_abs: if True, importance uses |coef|; else uses raw coef

    Returns:
      topk: list of (layer, score) for top-k layers
      layer2score: full dict layer->score
    """
    layers = list(layers)
    Lnum = len(layers)

    layer2score = defaultdict(float)
    layer2ranks_sum = defaultdict(float)  # optional diagnostics: sum of ranks
    used_langs = 0

    for r in rows:
        lang = r.lang.split("|")[0]  # handle possible "xx|correct"
        if lang in exclude_langs:
            continue
        if r.coef_unscaled is None:
            continue

        coef = np.asarray(r.coef_unscaled, dtype=float).reshape(-1)
        if coef.shape[0] != Lnum:
            continue

        imp = np.abs(coef) if use_abs else coef

        # ranks: 1 = most important
        order = np.argsort(imp)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, Lnum + 1)

        w = float(r.n) if weight_by_n else 1.0

        # Borda: higher points for better ranks
        # points = (Lnum - rank) so best gets Lnum-1, worst gets 0
        for i, layer in enumerate(layers):
            rank_i = int(ranks[i])
            points = (Lnum - rank_i)
            layer2score[layer] += w * points
            layer2ranks_sum[layer] += w * rank_i

        used_langs += 1

    if used_langs == 0:
        return [], dict(layer2score)

    ranked = sorted(layer2score.items(), key=lambda x: x[1], reverse=True)
    topk = ranked[:k]
    return topk, dict(layer2score)


def _base_lang(lang_label: str) -> str:
    # handles "ja|correct" etc.
    return lang_label.split("|", 1)[0]



#def sweep_topk_and_collect_lang_metrics_given_two_groups()
#outs_by_layer_into_english=outs_by_layer
#outs_by_layer_from_english=outs_by_layer_from_english
#given these two outs_by_layer_into_english, and outs_by_layer_from_english
#i want to iterate through top


def nice_label_from_run_dir(run_dir: str) -> str:
    # legend label: include prompt1 if present + shot count
    s = run_dir.lower()
    shot = "0shot" if "0shot" in s else ("3shot" if "3shot" in s else "shot?")
    prompt = "prompt1_" if "prompt1" in s else ""
    return f"{prompt}{shot}"
import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any







# -----------------------------
# Plot: "max pearson" picked on train, shown on test (bars per language)
# -----------------------------
def _pearson_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = x[mask]; y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)

def _resolve_y_layer(layers, y_layer):
    if y_layer == "last":
        return int(max(layers))
    if y_layer == "first":
        return int(min(layers))
    return int(y_layer)

# -----------------------------
# Train/test split
# Returns split ids so we can do cumulative curves on TEST
# Also stores both max-Pearson and min-Pearson selected layers.
# -----------------------------
def tt_select_best_layer_with_split(
    outs_by_layer: dict,
    *,
    valid_langs,
    layers=None,
    seed: int = 0,
    y_layer="last",
    min_n: int = 100,
    train_frac: float = 0.2,
    select_objective: str = "max_r",  # kept for backward compat but ignored; selection is by PP
    x_bar: str = "percentile",        # "percentile", "fixedrange", or "pct_scaled"
    bin_width: float = 0.05,
    acc_map: dict | None = None,      # optional {(idx, lang): 0/1} for accuracy-based PP
):
    """
    Selects best/worst layer per language by highest/lowest cumulative predictive power (PP)
    computed on the training split.

    PP = 1 - auc / avg_acc  (or baseline_area variant for fixedrange/pct_scaled).
    x_bar controls which cumulative x-axis mode is used for the PP calculation.

    Returns:
      tt[lang] = {
        "common_ids": np.ndarray[int] length n,
        "train_mask": np.ndarray[bool] length n,
        "test_mask": np.ndarray[bool] length n,

        "best_layer": int,
        "best_train_pp": float,
        "test_r": float,
        "test_p": float,

        "worst_layer": int,
        "worst_train_pp": float,
        "worst_test_r": float,
        "worst_test_p": float,

        "n": int,
        "n_train": int,
        "n_test": int,
        "y_layer_used": int,
      }
    """
    if layers is None:
        layers = sorted(outs_by_layer.keys())
    else:
        layers = list(layers)

    if len(layers) == 0:
        return {}

    yL = _resolve_y_layer(layers, y_layer)
    rng = np.random.default_rng(seed)
    tt = {}

    if x_bar not in ("percentile", "fixedrange", "pct_scaled"):
        raise ValueError("x_bar must be 'percentile', 'fixedrange', or 'pct_scaled'")

    for lang in list(valid_langs):
        # require all candidate x-layers and y-layer to exist
        if any((L not in outs_by_layer) or (lang not in outs_by_layer[L]) for L in layers):
            print("any((L not in outs_by_layer) or (lang not in outs_by_layer[L]) for L in layers)")
            continue
        if (yL not in outs_by_layer) or (lang not in outs_by_layer[yL]):
            print("(yL not in outs_by_layer) or (lang not in outs_by_layer[yL])")
            continue

        # -------------------------
        # Align example ids across all candidate layers
        # -------------------------
        idx_sets = []
        for L in layers:
            idxL = np.asarray(outs_by_layer[L][lang]["indices"], dtype=np.int64)
            idx_sets.append(set(map(int, idxL)))

        common_ids = sorted(set.intersection(*idx_sets))
        n = len(common_ids)
        if n < min_n:
            continue
        common_ids = np.asarray(common_ids, dtype=np.int64)

        # -------------------------
        # Build y aligned to common_ids using y-layer
        # -------------------------
        idx_y = np.asarray(outs_by_layer[yL][lang]["indices"], dtype=np.int64)
        pos_y = {int(ex_id): i for i, ex_id in enumerate(idx_y)}
        y_arr = np.asarray(outs_by_layer[yL][lang]["y"], dtype=np.float64)

        if not all(int(ex_id) in pos_y for ex_id in common_ids):
            continue

        y = np.asarray(
            [float(y_arr[pos_y[int(ex_id)]]) for ex_id in common_ids],
            dtype=np.float64,
        )

        # -------------------------
        # Prebuild x per layer aligned to common_ids
        # -------------------------
        X_by_L = {}
        valid_all_layers = True
        for L in layers:
            idx_x = np.asarray(outs_by_layer[L][lang]["indices"], dtype=np.int64)
            pos_x = {int(ex_id): i for i, ex_id in enumerate(idx_x)}
            x_arr = np.asarray(outs_by_layer[L][lang]["x_ortho"], dtype=np.float64)

            if not all(int(ex_id) in pos_x for ex_id in common_ids):
                valid_all_layers = False
                break

            X_by_L[L] = np.asarray(
                [float(x_arr[pos_x[int(ex_id)]]) for ex_id in common_ids],
                dtype=np.float64,
            )

        if not valid_all_layers:
            continue

        # -------------------------
        # Train/test split
        # -------------------------
        perm = rng.permutation(n)
        n_train = int(np.floor(train_frac * n))
        n_train = max(1, min(n - 1, n_train))

        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # -------------------------
        # Build y for PP (use accuracy labels if acc_map provided, else pack y)
        # -------------------------
        if acc_map is not None:
            y_for_pp = np.array(
                [float(acc_map.get((int(ex_id), lang), np.nan)) for ex_id in common_ids],
                dtype=np.float64,
            )
        else:
            y_for_pp = y

        # -------------------------
        # Pick best and worst layers on TRAIN by cumulative PP
        # -------------------------
        best_L = None
        best_score = -np.inf
        best_train_pp = np.nan

        worst_L = None
        worst_score = np.inf
        worst_train_pp = np.nan

        for L in layers:
            x_tr = X_by_L[L][train_mask]
            y_tr = y_for_pp[train_mask]

            pp_tr = _compute_train_pp(x_tr, y_tr, x_bar, bin_width)
            if not np.isfinite(pp_tr):
                continue

            if pp_tr > best_score:
                best_score = float(pp_tr)
                best_L = int(L)
                best_train_pp = float(pp_tr)

            if pp_tr < worst_score:
                worst_score = float(pp_tr)
                worst_L = int(L)
                worst_train_pp = float(pp_tr)

        if best_L is None or worst_L is None:
            continue

        # -------------------------
        # Evaluate selected layers on TEST
        # -------------------------
        best_test_r, best_test_p = _pearson_safe(
            X_by_L[best_L][test_mask],
            y[test_mask],
        )

        worst_test_r, worst_test_p = _pearson_safe(
            X_by_L[worst_L][test_mask],
            y[test_mask],
        )

        tt[lang] = {
            "common_ids": common_ids,
            "train_mask": train_mask,
            "test_mask": test_mask,

            "best_layer": int(best_L),
            "best_train_pp": float(best_train_pp),
            "test_r": float(best_test_r),
            "test_p": float(best_test_p),

            "worst_layer": int(worst_L),
            "worst_train_pp": float(worst_train_pp),
            "worst_test_r": float(worst_test_r),
            "worst_test_p": float(worst_test_p),

            "n": int(n),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "y_layer_used": int(yL),
            "x_bar": x_bar,
        }

    return tt

# -----------------------------
# Cumulative curve on TEST
# -----------------------------


def _curve_values_from_sorted_y(
    y_sorted: np.ndarray,
    steps: np.ndarray,
    *,
    cumulative_mode: str = "cumulative",
) -> tuple[list[float], list[float]]:
    """
    Given y sorted by x_ortho from low->high, compute either:
      - cumulative prefix means over [0:k)
      - non-cumulative bin means over [k_prev:k)
    Returns (xs, ys), where xs are the step fractions.
    """
    y_sorted = np.asarray(y_sorted, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    n = int(y_sorted.size)
    if n <= 0:
        return [], []

    mode = str(cumulative_mode).lower()
    if mode not in ("cumulative", "non_cumulative", "non-cumulative", "noncumulative"):
        raise ValueError("cumulative_mode must be 'cumulative' or 'non_cumulative'")

    xs, ys = [], []
    prev_k = 0
    for frac in steps:
        k = int(np.ceil(float(frac) * n))
        k = max(1, min(n, k))
        xs.append(float(frac))
        if mode == "cumulative":
            seg = y_sorted[:k]
        else:
            lo = prev_k
            hi = k
            if hi <= lo:
                lo = max(0, hi - 1)
            seg = y_sorted[lo:hi]
            prev_k = k
        ys.append(float(np.mean(seg)) if seg.size > 0 else np.nan)
    return xs, ys




# -----------------------------
# Cumulative AUC helpers
# -----------------------------

def _cumulative_curve_full_from_sorted_y(y_sorted: np.ndarray):
    """
    Build the full cumulative-mean curve over x in [0, 1].

    IMPORTANT: because the y-axis is a cumulative mean, the left endpoint is
    copied from the first cumulative value rather than forced to 0.
    """
    y_sorted = np.asarray(y_sorted, dtype=np.float64)
    y_sorted = y_sorted[np.isfinite(y_sorted)]
    n = int(y_sorted.size)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    c = np.cumsum(y_sorted) / np.arange(1, n + 1, dtype=np.float64)
    x = np.arange(1, n + 1, dtype=np.float64) / float(n)

    x_full = np.concatenate([[0.0], x])
    c_full = np.concatenate([[c[0]], c])
    return x_full, c_full


def _cumulative_auc_from_sorted_y(y_sorted: np.ndarray) -> float:
    """
    True trapezoid AUC of the cumulative-mean curve over [0, 1].
    Convention: lower AUC is better.
    """
    xs, ys = _cumulative_curve_full_from_sorted_y(y_sorted)
    if xs.size < 2:
        return float("nan")
    return float(np.trapz(ys, xs))


def _baseline_auc_from_y(y: np.ndarray) -> float:
    """
    Expected AUC under random ordering. Since the expected cumulative mean at
    every rank is mean(y), the expected AUC is mean(y).
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def _perfect_auc_from_y_lower_is_better(y: np.ndarray) -> float:
    """
    Best possible AUC when lower AUC is better: sort y ascending.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return _cumulative_auc_from_sorted_y(np.sort(y))


def _recoverability_lower_auc_better(test_auc: float, baseline_auc: float, perfect_auc: float) -> float:
    """
    Fraction of possible improvement over random baseline recovered by the signal.

        rec = (baseline_auc - test_auc) / (baseline_auc - perfect_auc)

    1.0 = perfect ordering, 0.0 = random ordering, <0 = worse than random.
    """
    denom = baseline_auc - perfect_auc
    if not np.isfinite(denom) or abs(float(denom)) < 1e-12:
        return float("nan")
    return float((baseline_auc - test_auc) / denom)


def _auc_summary_lower_auc_better_from_sorted_y(y_sorted_by_signal: np.ndarray):
    """
    Given y sorted by the geometric signal, return test/baseline/perfect AUC
    and recoverability under the lower-AUC-is-better convention.
    """
    y_sorted_by_signal = np.asarray(y_sorted_by_signal, dtype=np.float64)
    y_sorted_by_signal = y_sorted_by_signal[np.isfinite(y_sorted_by_signal)]
    if y_sorted_by_signal.size == 0:
        return {
            "test_auc": float("nan"),
            "baseline_auc": float("nan"),
            "perfect_auc": float("nan"),
            "recoverability": float("nan"),
        }

    test_auc = _cumulative_auc_from_sorted_y(y_sorted_by_signal)
    baseline_auc = _baseline_auc_from_y(y_sorted_by_signal)
    perfect_auc = _perfect_auc_from_y_lower_is_better(y_sorted_by_signal)
    rec = _recoverability_lower_auc_better(test_auc, baseline_auc, perfect_auc)
    return {
        "test_auc": float(test_auc),
        "baseline_auc": float(baseline_auc),
        "perfect_auc": float(perfect_auc),
        "recoverability": float(rec),
    }


def _train_pp_percentile(x_ortho: np.ndarray, y: np.ndarray) -> float:
    """PP = 1 - auc/avg_acc for percentile x-axis. Higher PP = more predictive power."""
    m = np.isfinite(x_ortho) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x_f, y_f = x_ortho[m], y[m]
    avg_acc = float(np.mean(y_f))
    if avg_acc < 1e-9:
        return float("nan")
    order = np.argsort(x_f)
    auc = _cumulative_auc_from_sorted_y(y_f[order])
    if not np.isfinite(auc):
        return float("nan")
    return 1.0 - auc / avg_acc


def _train_pp_fixedrange(x_ortho: np.ndarray, y: np.ndarray, *, bin_width: float = 0.05) -> float:
    """PP = 1 - auc_r/baseline_area for fixedrange x-axis. Higher PP = more predictive power."""
    m = np.isfinite(x_ortho) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x_f, y_f = x_ortho[m], y[m]
    min_o, max_o = float(x_f.min()), float(x_f.max())
    x_range = max_o - min_o
    if x_range < 1e-12:
        return float("nan")
    avg_acc = float(np.mean(y_f))
    baseline_area = avg_acc * x_range
    if baseline_area < 1e-9:
        return float("nan")
    step = bin_width * x_range
    thresholds = np.arange(min_o + step, max_o + step * 0.5, step)
    xs_r, ys_r = [], []
    for thr in thresholds:
        mask = x_f <= thr
        if mask.sum() == 0:
            continue
        xs_r.append(float(min(thr, max_o)))
        ys_r.append(float(np.mean(y_f[mask])))
    if len(xs_r) < 2:
        return float("nan")
    auc_r = float(np.trapz(ys_r, xs_r))
    if not np.isfinite(auc_r):
        return float("nan")
    return 1.0 - auc_r / baseline_area


def _train_pp_pct_scaled(x_ortho: np.ndarray, y: np.ndarray, *, bin_width: float = 0.05) -> float:
    """PP = 1 - auc_s/baseline_area_s for pct_scaled x-axis. Higher PP = more predictive power."""
    m = np.isfinite(x_ortho) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x_f, y_f = x_ortho[m], y[m]
    avg_acc = float(np.mean(y_f))
    if avg_acc < 1e-9:
        return float("nan")
    order = np.argsort(x_f)
    x_sorted, y_sorted = x_f[order], y_f[order]
    n = len(x_sorted)
    steps = np.arange(bin_width, 1.0 + 1e-9, bin_width)
    xs_s, ys_s = [], []
    for frac in steps:
        k = max(1, min(n, int(np.ceil(frac * n))))
        xs_s.append(float(x_sorted[k - 1]))
        ys_s.append(float(np.mean(y_sorted[:k])))
    if len(xs_s) < 2:
        return float("nan")
    xs_s = np.array(xs_s)
    ys_s = np.array(ys_s)
    x_range_s = float(xs_s[-1] - xs_s[0])
    if x_range_s < 1e-12:
        return float("nan")
    auc_s = float(np.trapz(ys_s, xs_s))
    baseline_area_s = avg_acc * x_range_s
    if baseline_area_s < 1e-9 or not np.isfinite(auc_s):
        return float("nan")
    return 1.0 - auc_s / baseline_area_s


def _compute_train_pp(
    x_ortho: np.ndarray, y: np.ndarray, x_bar: str, bin_width: float = 0.05
) -> float:
    """Dispatch to the appropriate PP computation for the given x_bar mode."""
    if x_bar == "percentile":
        return _train_pp_percentile(x_ortho, y)
    elif x_bar == "fixedrange":
        return _train_pp_fixedrange(x_ortho, y, bin_width=bin_width)
    elif x_bar == "pct_scaled":
        return _train_pp_pct_scaled(x_ortho, y, bin_width=bin_width)
    else:
        raise ValueError(f"x_bar must be 'percentile', 'fixedrange', or 'pct_scaled'; got {x_bar!r}")


def _resolve_curve_y_from_pack(
    *,
    lang: str,
    y_default: np.ndarray,
    idxs: np.ndarray | None = None,
    y_acc_by_lang=None,
    y_accuracy_default: np.ndarray | None = None,
    require_len: int | None = None,
) -> tuple[np.ndarray, str]:
    """
    Resolve accuracy y for plotting. Accepts either:
      - y_acc_by_lang[lang] as an array aligned to idxs/common_ids
      - y_acc_by_lang[lang] as a dict {index -> 0/1}
      - y_acc_by_lang as a dict {(index, lang) -> 0/1}
    If none of the above are available, falls back to y_default only if it is binary.
    """
    y_default = np.asarray(y_default, dtype=np.float64)

    if y_accuracy_default is not None:
        y_acc0 = np.asarray(y_accuracy_default, dtype=np.float64)
        if require_len is not None and len(y_acc0) != int(require_len):
            raise ValueError(f"{lang}: y_accuracy_default length {len(y_acc0)} != expected length {require_len}")
        return y_acc0, "Mean accuracy"

    if y_acc_by_lang is not None:
        if isinstance(y_acc_by_lang, dict) and lang in y_acc_by_lang:
            src = y_acc_by_lang[lang]
            if isinstance(src, dict):
                if idxs is None:
                    raise ValueError(f"{lang}: idxs required when y_acc_by_lang[lang] is a dict")
                y_acc = np.asarray([float(src[int(i)]) for i in idxs], dtype=np.float64)
            else:
                y_acc = np.asarray(src, dtype=np.float64)
                if require_len is not None and len(y_acc) != int(require_len):
                    raise ValueError(f"{lang}: y_acc length {len(y_acc)} != expected length {require_len}")
            return y_acc, "Mean accuracy"

        if isinstance(y_acc_by_lang, dict) and idxs is not None:
            try:
                y_acc = np.asarray([float(y_acc_by_lang[(int(i), lang)]) for i in idxs], dtype=np.float64)
                return y_acc, "Mean accuracy"
            except Exception:
                pass

    finite = y_default[np.isfinite(y_default)]
    if finite.size > 0:
        uniq = np.unique(finite)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            return y_default, "Mean accuracy"

    raise ValueError(
        f"Need accuracy labels for {lang}. Pass y_acc_by_lang aligned to the plotting ids, "
        "or use binary y in pack['y']."
    )

def build_y_acc_by_lang_from_predictions_jsonl(
    predictions_jsonl_path,
    tt,
):
    """
    predictions_jsonl must contain fields:
        - "index"
        - "lang"
        - "match" (0/1)

    Returns:
        y_acc_by_lang[lang] aligned with tt[lang]["common_ids"]
    """

    # 1) Load JSONL into dict
    acc_map = {}  # (index, lang) -> 0/1

    with open(predictions_jsonl_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            idx = int(ex["index"])
            lang = ex["lang"]
            match = int(ex["match"])
            acc_map[(idx, lang)] = match

    # 2) Align to tt
    y_acc_by_lang = {}

    for lang, info in tt.items():
        common_ids = info["common_ids"]

        acc_vec = []
        for idx in common_ids:
            key = (int(idx), lang)
            if key not in acc_map:
                raise ValueError(f"Missing {(idx, lang)} in predictions JSONL.")
            acc_vec.append(float(acc_map[key]))

        y_acc_by_lang[lang] = np.array(acc_vec, dtype=np.float64)

    return y_acc_by_lang

def _cos_vec_random_subspace_per_example_approx(
    X,
    k,
    eps=1e-12,
    show_progress=True,
):
    """
    Approximate cosine between each vector and a fresh random k-dim subspace.

    Uses column-normalized Gaussian vectors instead of full QR
    (much faster, good enough for diagnostic experiments).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    out = np.empty(n, dtype=np.float64)
    x_n = np.linalg.norm(X, axis=1)

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="random_subspace_per_example")

    for i in iterator:
        W = np.random.randn(d, k)

        # normalize columns (approximate orthonormal basis)
        W = W / np.clip(np.linalg.norm(W, axis=0, keepdims=True), eps, None)

        coeff = X[i] @ W
        proj_like = np.linalg.norm(coeff) / np.sqrt(k)

        out[i] = proj_like / max(x_n[i], eps)

    return np.clip(out, 0.0, 1.0)


def single_geometry_computation(feature, X1, vL_or_WL, eps: float = 1e-12, sv_weights=None,
                                sv_weight_mode: str = "sv"):
    """
    Compute one scalar geometric feature per example.

    Parameters
    ----------
    feature : str
        Name of geometric feature.
    X1 : np.ndarray
        [n, d] base vectors.
    vL_or_WL : np.ndarray | None
        Either:
            - vector signal vL of shape [d]
            - subspace basis WL of shape [d, k], assumed orthonormal
            - None for features that depend only on X1
    sv_weights : array-like | None
        Singular values for each basis column; required by
        vec_subspace_mean_weighted_by_variance.
    sv_weight_mode : str
        "sv": normalised by sum(sv); "sv_squared": normalised by sum(sv²); "none": direct sv-weighted sum.
    """
    feature = str(feature).lower()

    if vL_or_WL is None:
        raise ValueError(f"Feature '{feature}' requires a language-side signal, but got None.")

    arr = np.asarray(vL_or_WL)
    is_vector = (arr.ndim == 1)
    is_subspace = (arr.ndim == 2)

    # vec_vec_orthogonality: cumulative coherence's two-concept reduction (just the
    # angle between the two vectors) — used when the language-side signal is a
    # single vector rather than a subspace.
    if feature == "vec_vec_orthogonality":
        if not is_vector:
            raise ValueError(f"Feature '{feature}' requires a vector signal.")
        cos = _cos_vec_vec(X1, vL_or_WL)
        return 1.0 - abs(cos)

    elif feature == "vec_subspace_angle_by_cumulative_coherence":
        # 1 - abs(mean(sim_vec)) where sim_vec[i] = |v·w_i|/||v|| — the paper's CI
        # metric between a vector and a subspace.
        if not is_subspace:
            raise ValueError(f"Feature '{feature}' requires a subspace signal.")
        W = np.asarray(vL_or_WL, dtype=np.float64)
        norms = np.linalg.norm(X1, axis=1, keepdims=True)  # [n,1]
        abs_cos = np.abs(X1 @ W) / np.clip(norms, eps, None)  # [n, k]
        return 1.0 - np.abs(abs_cos.mean(axis=1))  # [n]

    else:
        raise ValueError(f"Unknown feature: {feature}")
def geometry_computation(feature, X1, vL_or_WL, eps: float = 1e-12, sv_weights=None,
                         sv_weight_mode: str = "sv"):
    return single_geometry_computation(feature, X1, vL_or_WL, eps=eps, sv_weights=sv_weights,
                                       sv_weight_mode=sv_weight_mode)
        



def compute_geometric_feature(
    model,
    run_dir: str,
    shot_tag: str,
    valid_langs: list,
    layer: int,
    rep_kind: str,
    lang_mode,
    *,
    targets_jsonl_path: str,
    target_kind: str = "prob",
    rank_kind: str = "rank_if_in_topk_else_lower_bound",
    y_transform: str = "identity",
    feature="vec_subspace_angle_by_cumulative_coherence",
    translation_data_dirs: dict,
    prefer_filled_N: bool = True,
    min_n: int = 100,
    separate_correct_incorrect_examples: bool = False,
    base_or_random_mode: str = "en_fact",
    random_seed: int = 0,
    mean_center_by_cluster: bool = False,
    rel_map_targets=None,
    rel_map_preds=None,
    oscar_split: str = 'all',
    store_X_base: bool = True,
    sv_weight_mode: str = "sv",
):
    from typing import Dict, Any
    import numpy as np

    def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < eps:
            out = np.zeros_like(v)
            out[0] = 1.0
            return out
        return v / n

    def _cos_vec_vec(A: np.ndarray, b_unit: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        b_unit = np.asarray(b_unit, dtype=np.float64)
        An = np.linalg.norm(A, axis=1)
        dot = A @ b_unit
        return dot / np.clip(An, eps, None)

    def _cos_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        An = np.linalg.norm(A, axis=1)
        Bn = np.linalg.norm(B, axis=1)
        denom = np.clip(An * Bn, eps, None)
        return np.sum(A * B, axis=1) / denom

    def _affine_project(X: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        Xc = X - mu[None, :]
        return (Xc @ W) @ W.T + mu[None, :]

    def _subspace_feature(X: np.ndarray, W: np.ndarray, feature_name: str, eps: float = 1e-12) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        proj = (X @ W) @ W.T
        proj_norm = np.linalg.norm(proj, axis=1)
        x_norm = np.linalg.norm(X, axis=1)
        align = proj_norm / np.clip(x_norm, eps, None)
        residual = X - proj
        ortho_resid = np.linalg.norm(residual, axis=1) / np.clip(x_norm, eps, None)
        if feature_name == "vec_subspace_alignment":
            return align
        elif feature_name == "vec_subspace_orthogonality":
            return ortho_resid
        else:
            raise ValueError(f"Unsupported subspace feature: {feature_name}")

    def _rand_vec_by_idx(
        d: int,
        *,
        seed: int,
        idx: int,
        same_norm: bool = True,
        ref_v: np.ndarray | None = None,
        eps: float = 1e-12,
    ) -> np.ndarray:
        s = (int(seed) & 0xFFFFFFFF)
        s ^= (int(idx) * 1315423911) & 0xFFFFFFFF
        rng = np.random.default_rng(s)
        v = rng.standard_normal(size=(d,)).astype(np.float64)
        u = _safe_unit(v, eps=eps)
        if not same_norm:
            return u
        if ref_v is None:
            raise ValueError("same_norm=True requires ref_v to scale to.")
        ref_n = float(np.linalg.norm(np.asarray(ref_v, dtype=np.float64)))
        if (not np.isfinite(ref_n)) or ref_n < eps:
            raise ValueError(f"ref_v has non-finite or near-zero norm: {ref_n}")
        return u * ref_n

    def _rand_subspace(d: int, k: int, *, seed: int, lang: str, layer: int, idx: int | None = None) -> np.ndarray:
        s = (int(seed) & 0xFFFFFFFF)
        for c in lang.encode("utf-8"):
            s = (s * 16777619) ^ int(c)
            s &= 0xFFFFFFFF
        s ^= (int(layer) * 2654435761) & 0xFFFFFFFF
        if idx is not None:
            s ^= (int(idx) * 2246822519) & 0xFFFFFFFF
        rng = np.random.default_rng(s)
        A = rng.standard_normal(size=(d, k)).astype(np.float64)
        return _orthonormalize(A)

    def _rand_vector_same_norm(
        d: int,
        ref_v: np.ndarray | None,
        *,
        seed: int,
        lang: str,
        layer: int,
        same_norm: bool = True,
        eps: float = 1e-12,
    ) -> np.ndarray | None:
        target_norm = 1.0
        if same_norm and ref_v is not None:
            n = float(np.linalg.norm(np.asarray(ref_v, dtype=np.float64)))
            if np.isfinite(n) and n > eps:
                target_norm = n
        s = (int(seed) & 0xFFFFFFFF)
        for c in lang.encode("utf-8"):
            s = (s * 16777619) ^ int(c)
            s &= 0xFFFFFFFF
        s ^= (int(layer) * 2654435761) & 0xFFFFFFFF
        rng = np.random.default_rng(s)
        v = rng.standard_normal(size=(d,)).astype(np.float64)
        v = _safe_unit(v, eps=eps)
        if v is None or (not np.isfinite(v).all()):
            return None
        return v * target_norm

    mode = str(base_or_random_mode).lower()
    if mode == "base":
        mode = "en_fact"

    ALL_SAVED_LANGS = ["en", "ca", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"]
    lang2id = {l: i for i, l in enumerate(ALL_SAVED_LANGS)}
    id2lang = {i: l for l, i in lang2id.items()}

    indices, lang_id, match = load_run_small_arrays(run_dir)
    summary = load_summary(run_dir, shot_tag)

    y_all = None
    y_all_prob = None

    if target_kind == "binary_correct":
        binary_map = load_binary_correct_targets_by_index_lang(targets_jsonl_path)
        y_all = np.full(len(indices), np.nan, dtype=np.float64)
        for r in range(len(indices)):
            idx = int(indices[r])
            lid = int(lang_id[r])
            if lid < 0:
                continue
            lang = id2lang.get(lid, None)
            if lang is None:
                continue
            key = (idx, lang)
            if key in binary_map:
                y_all[r] = float(binary_map[key])
        if y_transform != "identity":
            y_all = _transform_y(y_all, y_transform)
    else:
        targets_by_index_lang = load_correct_token_targets_jsonl_by_index_lang(targets_jsonl_path)
        if target_kind == "delta_nll":
            y_all_prob = extract_targets_for_index_lang(
                indices=indices,
                lang_id=lang_id,
                targets_by_index_lang=targets_by_index_lang,
                target_kind="prob",
                rank_kind=rank_kind,
            )
        else:
            y_all = extract_targets_for_index_lang(
                indices=indices,
                lang_id=lang_id,
                targets_by_index_lang=targets_by_index_lang,
                target_kind=target_kind,
                rank_kind=rank_kind,
            )
            y_all = _transform_y(y_all, y_transform)

    resid_mm = open_resid_memmap(run_dir, rep_kind, summary)
    H_layer = resid_mm[:, layer, :]
    d_model = int(H_layer.shape[1])

    EN_LID = lang2id["en"]
    en_rows = np.where(lang_id == EN_LID)[0]
    en_correct_rows = en_rows[match[en_rows] == 1]
    if en_correct_rows.size == 0:
        raise ValueError("No EN-correct rows found.")

    en_row_by_index = {}
    for r in en_correct_rows.tolist():
        en_row_by_index[int(indices[r])] = int(r)
    en_correct_indices = sorted(en_row_by_index.keys())

    row_by_lang_index = {}
    for r in range(len(indices)):
        lid = int(lang_id[r])
        if lid < 0:
            continue
        idx = int(indices[r])
        row_by_lang_index[(lid, idx)] = int(r)

    relation_by_lang_index = {}
    for r in range(len(indices)):
        lid = int(lang_id[r])
        if lid < 0:
            continue
        idx = int(indices[r])
        lang = id2lang.get(lid, None)
        if lang is None:
            continue

        rel = _resolve_relation(
            idx=idx,
            lang=lang,
            rel_map_targets=rel_map_targets,
            rel_map_preds=rel_map_preds,
        )
        relation_by_lang_index[(lid, idx)] = None if rel is None else str(rel)

    lang_mode_str = str(lang_mode).lower()
    _is_mean_translation_mode = (
        isinstance(lang_mode, (tuple, list)) and len(lang_mode) >= 1
        and str(lang_mode[0]).lower() == "mean_translation_vector"
    )
    use_translation_vector = ("translation_vector" in lang_mode_str) and not _is_mean_translation_mode
    use_both_translation_vectors = (lang_mode_str == "translation_vector_from_english_and_into_english")
    no_language_signal = (mode == "en_fact_norm")

    if mode == "random_translation_vector" and (not use_translation_vector):
        raise ValueError("mode='random_translation_vector' requires translation_vector lang_mode.")

    if mode.startswith("fixed_translation_vector_") and (not use_translation_vector):
        raise ValueError("mode='fixed_translation_vector_{lan}' requires translation_vector lang_mode.")

    use_subspace_mode = (
        (isinstance(lang_mode, (tuple, list))
         and len(lang_mode) >= 3
         and str(lang_mode[0]).lower() in (
             "language_subspace",
             "language_subspace_meanshifted",
             "subspace",
             "en_subspace",
             "language_subspace_mean_vector",
             "mean_translation_vector",
             "center_oscar_and_uncentered_language_subspace",
             "center_oscar_and_language_subspace",
             "center_oscar_and_language_subspace_meanshifted",
         ))
        or (isinstance(lang_mode, str) and lang_mode.lower() == "english_subspace")
    )

    subspace_mode_name = None
    if isinstance(lang_mode, (tuple, list)) and len(lang_mode) >= 1:
        subspace_mode_name = str(lang_mode[0]).lower()

    use_mean_shift = (subspace_mode_name in (
        "language_subspace_meanshifted",
        "center_oscar_and_language_subspace_meanshifted",
    ))
    use_WL_only = (subspace_mode_name in (
        "language_subspace",
        "center_oscar_and_uncentered_language_subspace",
        "center_oscar_and_language_subspace",
    ))
    use_muL_only = (subspace_mode_name == "language_subspace_mean_vector")
    use_mean_translation = (subspace_mode_name == "mean_translation_vector")

    use_center_oscar = subspace_mode_name in (
        "center_oscar_and_uncentered_language_subspace",
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )
    use_center_oscar_per_lang_center = subspace_mode_name in (
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )

    if not (use_translation_vector or use_subspace_mode):
        raise ValueError("lang_mode must contain 'translation_vector' or be a subspace mode.")

    if (mode in ("random_subspace", "random_subspace_per_example", "random_both") or "fixed_subspace" in mode) and use_translation_vector:
        raise ValueError(f"base_or_random_mode='{mode}' is not compatible with translation_vector lang_mode.")

    if (mode == "random_translation_vector" or mode.startswith("fixed_translation_vector_")) and use_subspace_mode:
        raise ValueError(f"base_or_random_mode='{mode}' is not compatible with subspace lang_mode.")

    fixed_translation_lang = None
    if mode.startswith("fixed_translation_vector_"):
        fixed_translation_lang = mode[len("fixed_translation_vector_"):]
        if fixed_translation_lang not in ALL_SAVED_LANGS:
            raise ValueError(f"Unknown language in mode='{base_or_random_mode}'.")
        if fixed_translation_lang == "en":
            raise ValueError("fixed_translation_vector_en is not valid.")

    fixed_subspace_lang = None
    if "fixed_subspace" in mode:
        fixed_subspace_lang = mode.split("_")[-1]
        if fixed_subspace_lang not in ALL_SAVED_LANGS:
            raise ValueError(f"Unknown language in mode='{base_or_random_mode}'.")

    translation_vecs_by_layer = None
    translation_vecs_by_layer_from_english = None
    translation_vecs_by_layer_into_english = None

    if use_translation_vector:
        if use_both_translation_vectors:
            translation_vecs_by_layer_from_english = load_translation_vectors(
                run_dir=translation_data_dirs["translation_vector_from_english"],
                model=model,
                use_resid="last",
                prefer_filled_N=prefer_filled_N,
                verbose=False,
            )
            translation_vecs_by_layer_into_english = load_translation_vectors(
                run_dir=translation_data_dirs["translation_vector_into_english"],
                model=model,
                use_resid="last",
                prefer_filled_N=prefer_filled_N,
                verbose=False,
            )
        else:
            translation_vecs_by_layer = load_translation_vectors(
                run_dir=translation_data_dirs[lang_mode],
                model=model,
                use_resid="last",
                prefer_filled_N=prefer_filled_N,
                verbose=False,
            )

    lang_method = None
    varp = None
    if use_subspace_mode:
        if isinstance(lang_mode, str):
            lang_method = "SVD"
            varp = 0.90
        else:
            lang_method = str(lang_mode[1])
            varp = float(lang_mode[2])

    v_by_lang: Dict[str, Any] = {}
    v_from_en_by_lang: Dict[str, Any] = {}
    v_into_en_by_lang: Dict[str, Any] = {}

    W_by_lang: Dict[str, Any] = {}
    mu_by_lang: Dict[str, Any] = {}
    sv_by_lang: Dict[str, Any] = {}
    k_by_lang: Dict[str, Any] = {}

    if not no_language_signal:
        if use_translation_vector:
            if use_both_translation_vectors:
                for L in valid_langs:
                    v_from_real = translation_vecs_by_layer_from_english[layer].get(L, None)
                    v_into_real = translation_vecs_by_layer_into_english[layer].get(L, None)

                    v_from_en_by_lang[L] = None if v_from_real is None else _safe_unit(v_from_real.astype(np.float64, copy=False))
                    v_into_en_by_lang[L] = None if v_into_real is None else _safe_unit(v_into_real.astype(np.float64, copy=False))

                if fixed_translation_lang is not None:
                    v_from_fixed_real = translation_vecs_by_layer_from_english[layer].get(fixed_translation_lang, None)
                    v_into_fixed_real = translation_vecs_by_layer_into_english[layer].get(fixed_translation_lang, None)

                    v_from_fixed = None if v_from_fixed_real is None else _safe_unit(v_from_fixed_real.astype(np.float64, copy=False))
                    v_into_fixed = None if v_into_fixed_real is None else _safe_unit(v_into_fixed_real.astype(np.float64, copy=False))

                    if v_from_fixed is None or not np.isfinite(v_from_fixed).all():
                        raise ValueError(
                            f"No valid fixed translation vector (from English) for '{fixed_translation_lang}' at layer={layer}."
                        )
                    if v_into_fixed is None or not np.isfinite(v_into_fixed).all():
                        raise ValueError(
                            f"No valid fixed translation vector (into English) for '{fixed_translation_lang}' at layer={layer}."
                        )

                    for L in valid_langs:
                        v_from_en_by_lang[L] = v_from_fixed
                        v_into_en_by_lang[L] = v_into_fixed

                elif mode == "random_translation_vector":
                    for L in valid_langs:
                        v_from_real = translation_vecs_by_layer_from_english[layer].get(L, None)
                        v_into_real = translation_vecs_by_layer_into_english[layer].get(L, None)

                        v_from_rand = _rand_vector_same_norm(
                            d_model, v_from_real, seed=random_seed, lang=L, layer=layer, same_norm=True
                        )
                        v_into_rand = _rand_vector_same_norm(
                            d_model, v_into_real, seed=random_seed + 100003, lang=L, layer=layer, same_norm=True
                        )

                        v_from_en_by_lang[L] = None if v_from_rand is None else _safe_unit(v_from_rand.astype(np.float64, copy=False))
                        v_into_en_by_lang[L] = None if v_into_rand is None else _safe_unit(v_into_rand.astype(np.float64, copy=False))

            else:
                for L in valid_langs:
                    v_real = translation_vecs_by_layer[layer].get(L, None)
                    v_by_lang[L] = None if v_real is None else _safe_unit(v_real.astype(np.float64, copy=False))

                if fixed_translation_lang is not None:
                    v_fixed_real = translation_vecs_by_layer[layer].get(fixed_translation_lang, None)
                    v_fixed = None if v_fixed_real is None else _safe_unit(v_fixed_real.astype(np.float64, copy=False))
                    if v_fixed is None or not np.isfinite(v_fixed).all():
                        raise ValueError(f"No valid fixed translation vector for '{fixed_translation_lang}' at layer={layer}.")
                    for L in valid_langs:
                        v_by_lang[L] = v_fixed

                elif mode == "random_translation_vector":
                    for L in valid_langs:
                        v_real = translation_vecs_by_layer[layer].get(L, None)
                        v_rand = _rand_vector_same_norm(
                            d_model, v_real, seed=random_seed, lang=L, layer=layer, same_norm=True
                        )
                        v_by_lang[L] = None if v_rand is None else _safe_unit(v_rand.astype(np.float64, copy=False))

        elif use_center_oscar:
            all_full_langs = list(abbr_to_full_LANGUAGE_CODE_MAP.values())
            grand_mean = _oscar_global_mean_cached(
                layer, all_full_langs, oscar_resids_root, oscar_cache_root, max_oscar_rows, verbose=False,
            )
            for L in tqdm(valid_langs, desc=f"global-cent subspaces L{layer}", leave=False):
                full = abbr_to_full_LANGUAGE_CODE_MAP[L]
                WL, muL, svL = oscar_W_global_centered_cached(
                    lang=full,
                    layer=layer,
                    subspace_method=lang_method,
                    var_prop=varp,
                    oscar_resids_root=oscar_resids_root,
                    disk_cache_root=oscar_cache_root,
                    grand_mean=grand_mean,
                    per_lang_center=use_center_oscar_per_lang_center,
                    max_rows=max_oscar_rows,
                    verbose=False,
                )
                WL = None if WL is None else _orthonormalize(WL)
                muL = None if muL is None else np.asarray(muL, dtype=np.float64)
                svL = None if svL is None else np.asarray(svL, dtype=np.float64)
                W_by_lang[L] = WL
                mu_by_lang[L] = muL
                sv_by_lang[L] = svL
                k_by_lang[L] = None if WL is None else int(WL.shape[1])

            if fixed_subspace_lang is not None:
                full_lan = abbr_to_full_LANGUAGE_CODE_MAP[fixed_subspace_lang]
                W_FIXED, MU_FIXED, SV_FIXED = oscar_W_global_centered_cached(
                    lang=full_lan,
                    layer=layer,
                    subspace_method=lang_method,
                    var_prop=varp,
                    oscar_resids_root=oscar_resids_root,
                    disk_cache_root=oscar_cache_root,
                    grand_mean=grand_mean,
                    per_lang_center=use_center_oscar_per_lang_center,
                    max_rows=max_oscar_rows,
                    verbose=False,
                )
                W_FIXED = None if W_FIXED is None else _orthonormalize(W_FIXED)
                MU_FIXED = None if MU_FIXED is None else np.asarray(MU_FIXED, dtype=np.float64)
                SV_FIXED = None if SV_FIXED is None else np.asarray(SV_FIXED, dtype=np.float64)
                k_fixed = None if W_FIXED is None else int(W_FIXED.shape[1])
                for L in valid_langs:
                    W_by_lang[L] = W_FIXED
                    mu_by_lang[L] = MU_FIXED
                    sv_by_lang[L] = SV_FIXED
                    k_by_lang[L] = k_fixed

            if mode in ("random_subspace", "random_both"):
                for L in valid_langs:
                    k = k_by_lang.get(L, None)
                    if k is None or k <= 0:
                        W_by_lang[L] = None
                        continue
                    W_by_lang[L] = _rand_subspace(d_model, k, seed=random_seed, lang=L, layer=layer)

        else:
            for L in valid_langs:
                full = abbr_to_full_LANGUAGE_CODE_MAP[L]
                WL, muL, svL = oscar_W_cached(
                    lang=full,
                    layer=layer,
                    subspace_method=lang_method,
                    var_prop=varp,
                    oscar_resids_root=oscar_resids_root,
                    disk_cache_root=oscar_cache_root,
                    max_rows=max_oscar_rows,
                    verbose=False,
                    oscar_split=oscar_split,
                )
                WL = None if WL is None else _orthonormalize(WL)
                muL = None if muL is None else np.asarray(muL, dtype=np.float64)
                svL = None if svL is None else np.asarray(svL, dtype=np.float64)

                W_by_lang[L] = WL
                mu_by_lang[L] = muL
                sv_by_lang[L] = svL
                k_by_lang[L] = None if WL is None else int(WL.shape[1])

            if fixed_subspace_lang is not None:
                full_lan = abbr_to_full_LANGUAGE_CODE_MAP[fixed_subspace_lang]
                W_FIXED, MU_FIXED, SV_FIXED = oscar_W_cached(
                    lang=full_lan,
                    layer=layer,
                    subspace_method=lang_method,
                    var_prop=varp,
                    oscar_resids_root=oscar_resids_root,
                    disk_cache_root=oscar_cache_root,
                    max_rows=max_oscar_rows,
                    verbose=False,
                    oscar_split=oscar_split,
                )
                W_FIXED = None if W_FIXED is None else _orthonormalize(W_FIXED)
                MU_FIXED = None if MU_FIXED is None else np.asarray(MU_FIXED, dtype=np.float64)
                SV_FIXED = None if SV_FIXED is None else np.asarray(SV_FIXED, dtype=np.float64)
                k_fixed = None if W_FIXED is None else int(W_FIXED.shape[1])

                for L in valid_langs:
                    W_by_lang[L] = W_FIXED
                    mu_by_lang[L] = MU_FIXED
                    sv_by_lang[L] = SV_FIXED
                    k_by_lang[L] = k_fixed

            if mode in ("random_subspace", "random_both"):
                for L in valid_langs:
                    k = k_by_lang.get(L, None)
                    if k is None or k <= 0:
                        W_by_lang[L] = None
                        continue
                    W_by_lang[L] = _rand_subspace(d_model, k, seed=random_seed, lang=L, layer=layer)

    out: Dict[str, Dict[str, Any]] = {}

    for L in valid_langs:
        if L not in lang2id:
            continue
        lid = lang2id[L]

        base_vecs = []
        y_list = []
        matchL_list = []
        idx_list = []
        relation_list = []

        for idx in en_correct_indices:
            key = (lid, idx)
            if key not in row_by_lang_index:
                continue
            rL = row_by_lang_index[key]
            rEN = en_row_by_index[idx]

            if target_kind == "delta_nll":
                if y_all_prob is None:
                    raise ValueError("Internal error: y_all_prob is None for target_kind='delta_nll'")
                pL = float(y_all_prob[rL])
                pEN = float(y_all_prob[rEN])
                if (not np.isfinite(pL)) or (not np.isfinite(pEN)):
                    continue
                eps = 1e-12
                pL = max(pL, eps)
                pEN = max(pEN, eps)
                yv = (-np.log(pL)) - (-np.log(pEN))
                if y_transform != "identity":
                    yv = float(_transform_y(np.asarray([yv], dtype=np.float64), y_transform)[0])
            else:
                if y_all is None:
                    raise ValueError("Internal error: y_all is None for non-delta target_kind")
                yv = y_all[rL]
                if not np.isfinite(yv):
                    continue

            if mode in ("random_vector", "random_both"):
                x1 = _rand_vec_by_idx(
                    d_model, seed=random_seed, idx=int(idx), same_norm=True, ref_v=H_layer[rEN]
                )
                if x1 is None or not np.isfinite(x1).all():
                    continue
            else:
                x1 = H_layer[rEN].astype(np.float64, copy=False)

            rel = relation_by_lang_index.get((lid, int(idx)), None)

            base_vecs.append(x1)
            y_list.append(float(yv))
            matchL_list.append(int(match[rL]))
            idx_list.append(int(idx))
            relation_list.append(None if rel is None else str(rel))

        if len(y_list) < min_n:
            continue

        X1 = np.stack(base_vecs, axis=0).astype(np.float64, copy=False)
        yL = np.asarray(y_list, dtype=np.float64)
        matchL = np.asarray(matchL_list, dtype=np.int8)
        relation_arr = np.asarray(relation_list, dtype=object)

        if mean_center_by_cluster:
            cluster_mean = np.mean(X1, axis=0)
            X1 = X1 - cluster_mean

        if use_translation_vector:
            if use_both_translation_vectors:
                v_from = v_from_en_by_lang.get(L, None)
                v_into = v_into_en_by_lang.get(L, None)

                if v_from is None or not np.isfinite(v_from).all():
                    continue
                if v_into is None or not np.isfinite(v_into).all():
                    continue

                ortho_from = geometry_computation(feature, X1, v_from)
                ortho_into = geometry_computation(feature, X1, v_into)
                ortho = ortho_from + ortho_into
            else:
                vL = v_by_lang.get(L, None)
                if vL is None or not np.isfinite(vL).all():
                    continue
                ortho = geometry_computation(feature, X1, vL)

        else:
            muL = mu_by_lang.get(L, None)
            WL = W_by_lang.get(L, None)

            if WL is None and not use_mean_translation:
                continue

            if use_WL_only:
                ortho = geometry_computation(feature, X1, WL, sv_weights=sv_by_lang.get(L, None),
                                             sv_weight_mode=sv_weight_mode)
            elif use_muL_only:
                ortho = geometry_computation(feature, X1, muL)

            elif use_mean_translation:
                # direction = unit(mu_L - mu_ENFACT), the vector from the
                # en-fact cluster centre to the language cluster centre
                if muL is None or not np.isfinite(muL).all():
                    continue
                mu_enfact = X1.mean(axis=0)
                direction = muL - mu_enfact
                n_dir = np.linalg.norm(direction)
                if not np.isfinite(n_dir) or n_dir < 1e-12:
                    continue
                direction_unit = (direction / n_dir).astype(np.float64)
                ortho = geometry_computation(feature, X1, direction_unit)

            elif use_mean_shift:
                if muL is None or not np.isfinite(muL).all():
                    continue

                X1_geom = X1 - muL[None, :]

                if mode == "random_subspace_per_example":
                    k = k_by_lang.get(L, None)
                    if k is None or k <= 0:
                        continue
                    cos = _cos_vec_random_subspace_per_example_approx(X1_geom, k)
                    ortho = 1.0 - cos
                else:
                    ortho = _subspace_feature(X1_geom, WL, feature)

            else:
                if mode == "random_subspace_per_example":
                    k = k_by_lang.get(L, None)
                    if k is None or k <= 0:
                        continue
                    cos = _cos_vec_random_subspace_per_example_approx(X1, k)
                    ortho = 1.0 - cos
                else:
                    ortho = _subspace_feature(X1, WL, feature)

        m = np.isfinite(ortho) & np.isfinite(yL)
        if int(m.sum()) < min_n:
            print(f"{L}: only {int(m.sum())} valid examples after filtering; skipping (need at least {min_n})")
            continue

        ortho = ortho[m].astype(np.float64)
        yL = yL[m].astype(np.float64)
        X_base = X1[m].astype(np.float64)
        idxs_m = np.asarray(idx_list, dtype=np.int64)[m]
        relation_m = relation_arr[m]

        meta_common = {
            "lang": L,
            "layer": int(layer),
            "rep_kind": rep_kind,
            "target_kind": target_kind,
            "rank_kind": rank_kind if target_kind == "rec_rank" else None,
            "y_transform": y_transform,
            "lang_mode": lang_mode,
            "base_or_random_mode": mode,
            "random_seed": int(random_seed),
            "n": int(len(yL)),
            "mean_centered_by_cluster": mean_center_by_cluster,
        }

        if separate_correct_incorrect_examples:
            matchL_m = matchL[m]
            splits = {}
            for split_name, split_val in (("correct", 1), ("incorrect", 0)):
                ms = (matchL_m == split_val)
                if int(ms.sum()) < min_n:
                    continue
                splits[split_name] = {
                    "x_ortho": ortho[ms],
                    "X_base": X_base[ms],
                    "y": yL[ms],
                    "indices": idxs_m[ms],
                    "relation": relation_m[ms],
                    "meta": {**meta_common, "split": split_name, "n": int(ms.sum())},
                }
            if not splits:
                continue
            out[L] = {"splits": splits, "meta": {**meta_common, "separate_correct_incorrect_examples": True}}
        else:
            pack = {
                "x_ortho": ortho,
                "X_base": X_base,
                "y": yL,
                "indices": idxs_m,
                "relation": relation_m,
                "meta": {**meta_common, "separate_correct_incorrect_examples": False},
            }
            # Always compute the vec-subspace cumulative-coherence feature when WL is
            # available, under its own name — used directly by callers that look it up
            # by key (e.g. EXP1_SRC_TGT_X_KEY / EXP1_TABLE_X_KEY), independent of
            # whichever `feature` drove x_ortho above.
            WL_coh = W_by_lang.get(L, None)
            if WL_coh is not None and WL_coh.ndim == 2 and WL_coh.shape[1] > 0:
                pack["vec_subspace_angle_by_cumulative_coherence"] = geometry_computation(
                    "vec_subspace_angle_by_cumulative_coherence", X_base, WL_coh)
            out[L] = pack

    if not store_X_base:
        for pack in out.values():
            if "splits" in pack:
                for sp in pack["splits"].values():
                    sp.pop("X_base", None)
            else:
                pack.pop("X_base", None)

    return out


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from scipy.stats import pearsonr

def plot_cumulative_all_layers_all_langs(
    outs_by_layer,
    tt,
    *,
    valid_langs,
    y_mode="accuracy",
    y_acc_by_lang=None,
    y_layer="last",
    quantile_step=0.1,
    cumulative_mode: str = "cumulative",
    title: str | None = None,
    save_path: str | None = None,
    save_data_path: str | None = None,
    calculate_auc: bool = True,
    auc_fmt: str = ".3f",
    show: bool = True,
    fixed_layers_to_plot=None,  # if specified, use these layers directly on TEST
    y_lim=(0.1, 0.85),
    lang_colors: dict | None = None,
    x_key: str = "x_ortho",
    pb_corr_subset: str | None = None,  # "all" or "test": compute point-biserial r(y_binary, x)
    pb_corr_save_path: str | None = None,
):
    """
    Plot one selected-layer curve per language.

    Layer-selection convention:
      * cumulative_mode == "cumulative": select the layer with lowest TRAIN
        cumulative AUC, unless fixed_layers_to_plot is provided.
      * cumulative_mode == "non_cumulative": STILL select layers from the
        cumulative trend. If tt[lang]["lowest_AUC_layer"] exists, use it;
        otherwise compute the same cumulative TRAIN-AUC selection internally.

    Cumulative legend reports:
        auc, avg_acc, PP, p(perm)

    Non-cumulative legend reports:
        R, p

    AUC convention:
      * lower AUC is better;
      * baseline AUC = mean(y), expected under random ordering;
      * perfect AUC = full trapezoid AUC after sorting y ascending;
      * recoverability = (baseline_auc - test_auc) / (baseline_auc - perfect_auc).
    """
    if len(outs_by_layer) == 0:
        print("outs_by_layer is empty.")
        return

    layers = sorted(outs_by_layer.keys())
    if len(layers) == 0:
        print("No layers found in outs_by_layer.")
        return

    steps = np.arange(quantile_step, 1.0 + 1e-9, quantile_step)
    mode = str(cumulative_mode).lower()
    is_cumulative = mode == "cumulative"
    mode_label = "cumulative" if is_cumulative else "non_cumulative"

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0
    final_y_label = None
    store_auc_for_langs = {}
    store_baseline_auc_for_langs = {}
    store_pp_for_langs = {}
    store_perm_p_for_langs = {}
    store_selected_layer_for_langs = {}
    store_r_for_langs = {}
    store_p_for_langs = {}
    store_rpb_for_langs = {}
    store_rpb_p_for_langs = {}
    curves_data = {}
    if pb_corr_subset is not None:
        from scipy.stats import pointbiserialr as _pointbiserialr

    def _resolve_y_all_for_lang(lang, common_ids):
        nonlocal final_y_label
        y_acc_default = None
        for L_probe in layers:
            if L_probe not in outs_by_layer or lang not in outs_by_layer[L_probe]:
                continue
            pack_probe = outs_by_layer[L_probe][lang]
            if isinstance(pack_probe, dict) and ("y_accuracy" in pack_probe):
                idx_y = np.asarray(pack_probe["indices"], dtype=np.int64)
                pos_y = {int(ex_id): i for i, ex_id in enumerate(idx_y)}
                y_arr_acc = np.asarray(pack_probe["y_accuracy"], dtype=np.float64)
                if all(int(ex_id) in pos_y for ex_id in common_ids):
                    y_acc_default = np.asarray(
                        [float(y_arr_acc[pos_y[int(ex_id)]]) for ex_id in common_ids],
                        dtype=np.float64,
                    )
                    break

        y_all, y_label = _resolve_curve_y_from_pack(
            lang=lang,
            y_default=np.full(len(common_ids), np.nan, dtype=np.float64),
            idxs=common_ids,
            y_acc_by_lang=y_acc_by_lang,
            y_accuracy_default=y_acc_default,
            require_len=len(common_ids),
        )
        final_y_label = y_label
        return y_all, y_label

    def _x_all_for_layer(lang, layer, common_ids):
        if layer not in outs_by_layer or lang not in outs_by_layer[layer]:
            return None
        pack = outs_by_layer[layer][lang]
        if x_key not in pack:
            return None
        idx_x = np.asarray(pack["indices"], dtype=np.int64)
        pos_x = {int(ex_id): i for i, ex_id in enumerate(idx_x)}
        if not all(int(ex_id) in pos_x for ex_id in common_ids):
            return None
        return np.asarray(
            [float(pack[x_key][pos_x[int(ex_id)]]) for ex_id in common_ids],
            dtype=np.float64,
        )

    def _train_auc_for_layer(lang, layer, common_ids, train_mask, y_all):
        x_all = _x_all_for_layer(lang, layer, common_ids)
        if x_all is None:
            return np.nan
        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        m = np.isfinite(x_train) & np.isfinite(y_train)
        if int(m.sum()) < 2:
            return np.nan
        order = np.argsort(x_train[m])
        y_sorted_train = y_train[m][order]
        return _cumulative_auc_from_sorted_y(y_sorted_train)

    for lang in valid_langs:
        if lang not in tt:
            continue

        d = tt[lang]
        common_ids = np.asarray(d["common_ids"], dtype=np.int64)
        train_mask = np.asarray(d["train_mask"], dtype=bool)
        test_mask = np.asarray(d["test_mask"], dtype=bool)
        if common_ids.size == 0:
            continue

        y_all, y_label = _resolve_y_all_for_lang(lang, common_ids)
        if y_all is None:
            continue

        # --------------------------------------------------
        # Choose layer. Non-cumulative curves use the same best layer selected
        # by cumulative TRAIN-AUC.
        # --------------------------------------------------
        best_layer = None
        best_train_auc = np.nan

        if fixed_layers_to_plot is not None:
            best_layer = fixed_layers_to_plot.get(lang, None)
            if best_layer is None:
                continue
            best_layer = int(best_layer)
            if best_layer not in outs_by_layer or lang not in outs_by_layer[best_layer]:
                continue
            best_train_auc = _train_auc_for_layer(lang, best_layer, common_ids, train_mask, y_all)
        else:
            # For non-cumulative plotting, prefer a layer selected earlier by a
            # cumulative call on the same tt. If unavailable, compute it here.
            if (not is_cumulative) and ("lowest_AUC_layer" in d):
                best_layer = int(d["lowest_AUC_layer"])
                best_train_auc = _train_auc_for_layer(lang, best_layer, common_ids, train_mask, y_all)
            else:
                best_train_auc = np.inf
                for L in layers:
                    if L not in outs_by_layer or lang not in outs_by_layer[L]:
                        continue
                    auc_tr = _train_auc_for_layer(lang, L, common_ids, train_mask, y_all)
                    if lang == "ja" and is_cumulative:
                        print(f"Layer {L}: train cumulative AUC = {auc_tr}")
                    if np.isfinite(auc_tr) and auc_tr < best_train_auc:
                        best_train_auc = float(auc_tr)
                        best_layer = int(L)

        if best_layer is None:
            continue
        tt[lang]["lowest_AUC_layer"] = int(best_layer)
        store_selected_layer_for_langs[lang] = int(best_layer)

        # --------------------------------------------------
        # Build TEST curve for selected layer
        # --------------------------------------------------
        x_all = _x_all_for_layer(lang, best_layer, common_ids)
        if x_all is None:
            continue

        x_test = x_all[test_mask]
        y_test = y_all[test_mask]
        m_te = np.isfinite(x_test) & np.isfinite(y_test)
        if int(m_te.sum()) == 0:
            continue

        x_test = x_test[m_te]
        y_test = y_test[m_te]
        order = np.argsort(x_test)
        y_sorted_test = y_test[order]
        if y_sorted_test.size == 0:
            continue

        # Point-biserial correlation: r(y_binary, CI), where CI ("confidence
        # index") = const - x_key is the order-reversing transform used
        # everywhere else in this file (e.g. CI = 1 - x_ortho; see
        # replot_exp2_scatter_from_json). Pearson/point-biserial correlation
        # is invariant to affine transforms up to sign, so r(y, CI) is just
        # -r(y, x_key) — we compute on raw x_key and flip the sign rather
        # than materializing CI explicitly.
        _rpb_val, _rpb_p = float("nan"), float("nan")
        if pb_corr_subset is not None:
            if pb_corr_subset == "all":
                _m_pb = np.isfinite(x_all) & np.isfinite(y_all)
                _x_pb, _y_pb = x_all[_m_pb], y_all[_m_pb]
            else:  # "test"
                _x_pb, _y_pb = x_test, y_test
            if len(_x_pb) >= 3 and len(np.unique(_y_pb)) >= 2:
                try:
                    _rpb_val, _rpb_p = _pointbiserialr(_y_pb, _x_pb)
                    _rpb_val, _rpb_p = -float(_rpb_val), float(_rpb_p)
                except Exception:
                    pass
            store_rpb_for_langs[lang] = _rpb_val
            store_rpb_p_for_langs[lang] = _rpb_p

        xs, ys = _curve_values_from_sorted_y(
            y_sorted_test,
            steps,
            cumulative_mode=cumulative_mode,
        )

        label = f"{lang}(L{best_layer})"

        if is_cumulative and calculate_auc:
            summ = _auc_summary_lower_auc_better_from_sorted_y(y_sorted_test)
            test_auc = summ["test_auc"]
            base_auc = summ["baseline_auc"]
            pp = (1.0 - test_auc / base_auc) if (np.isfinite(base_auc) and base_auc > 1e-9) else float("nan")
            rng = np.random.default_rng(42)
            n_perm = 1000
            null_aucs = np.empty(n_perm)
            for _pi in range(n_perm):
                x_perm = rng.permutation(x_test)
                null_aucs[_pi] = _cumulative_auc_from_sorted_y(y_test[np.argsort(x_perm)])
            perm_p = float(np.mean(null_aucs <= test_auc))
            store_auc_for_langs[lang] = test_auc
            store_baseline_auc_for_langs[lang] = base_auc
            store_pp_for_langs[lang] = pp
            store_perm_p_for_langs[lang] = perm_p
            perm_p_str = "p(perm)<0.01" if perm_p < 0.01 else f"p(perm)={perm_p:.3f}"
            label = (
                f"{label} | auc={test_auc:{auc_fmt}} | avg_acc={base_auc:{auc_fmt}} | "
                f"PP={pp:{auc_fmt}} | {perm_p_str}"
            )
            if pb_corr_subset is not None:
                _rpb_p_str = "p<0.01" if _rpb_p < 0.01 else f"p={_rpb_p:.3f}"
                label = f"{label} | rpb(acc,CI)={_rpb_val:.3f} {_rpb_p_str}"
            curves_data[lang] = {
                "xs": list(xs), "ys": list(ys), "label": label, "layer": int(best_layer),
                "auc": float(test_auc), "avg_acc": float(base_auc),
                "pp": float(pp), "perm_p": float(perm_p),
                **( {"rpb": _rpb_val, "rpb_p": _rpb_p} if pb_corr_subset is not None else {}),
            }
        elif (not is_cumulative):
            xs_arr = np.asarray(xs, dtype=np.float64)
            ys_arr = np.asarray(ys, dtype=np.float64)
            m_nc = np.isfinite(xs_arr) & np.isfinite(ys_arr)
            if m_nc.sum() >= 3 and np.std(ys_arr[m_nc]) > 1e-12:
                r_val, p_val = pearsonr(xs_arr[m_nc], ys_arr[m_nc])
                p_str = "p<0.01" if p_val < 0.01 else f"p={p_val:.3f}"
            else:
                r_val, p_val = float("nan"), float("nan")
                p_str = "p=nan"
            store_r_for_langs[lang] = r_val
            store_p_for_langs[lang] = p_val
            label = f"{label} | R={r_val:.3f} {p_str}"
            curves_data[lang] = {
                "xs": list(xs), "ys": list(ys), "label": label, "layer": int(best_layer),
                "r": float(r_val), "p": float(p_val),
            }
        else:
            curves_data[lang] = {"xs": list(xs), "ys": list(ys), "label": label, "layer": int(best_layer)}

        _color = lang_colors.get(lang) if lang_colors else None
        ax.plot(xs, ys, label=label, alpha=0.9, **({} if _color is None else {"color": _color}))
        plotted += 1

    if plotted == 0:
        print("No selected-layer curves to plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Bins Sorted from High to Low CI")
    ax.set_ylabel(final_y_label if final_y_label is not None else y_mode)
    ax.set_xlim(quantile_step - 1e-9, 1.0)
    ax.set_ylim(min(0,y_lim[0]), y_lim[1]+0.1)
    ax.grid(alpha=0.3)
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    if title is None:
        if fixed_layers_to_plot is None:
            title = f"All languages at cumulative train-selected layer ({mode_label}, TEST only)"
        else:
            title = f"All languages at fixed selected layers ({mode_label}, TEST only)"
    ax.set_title(title)

    ax.legend(
        ncol=1, fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="svg")
        print("Saved:", save_path)

    if save_data_path is not None:
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        _plot_meta = {
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": final_y_label if final_y_label is not None else y_mode,
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "cumulative_mode": mode_label,
            "curves": curves_data,
        }
        if store_rpb_for_langs:
            _plot_meta["pb_corr"] = {
                lang: {"rpb": store_rpb_for_langs[lang], "p": store_rpb_p_for_langs[lang]}
                for lang in store_rpb_for_langs
            }
        with open(save_data_path, "w") as _f:
            json.dump(_plot_meta, _f, indent=2)
        print("Saved data:", save_data_path)

    if pb_corr_save_path is not None and store_rpb_for_langs:
        _save_pb_corr_table_and_plot(
            store_rpb_for_langs, store_rpb_p_for_langs, pb_corr_save_path,
            title=f"Accuracy vs CI (point-biserial, {mode_label})" if title is None else f"{title} — accuracy vs CI",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    if is_cumulative:
        return {
            "test_auc": store_auc_for_langs,
            "baseline_auc": store_baseline_auc_for_langs,
            "pp": store_pp_for_langs,
            "perm_p": store_perm_p_for_langs,
            "selected_layer": store_selected_layer_for_langs,
            "curves": curves_data,
            "rpb": store_rpb_for_langs,
            "rpb_p": store_rpb_p_for_langs,
        }, tt
    else:
        return {
            "r": store_r_for_langs,
            "p": store_p_for_langs,
            "selected_layer": store_selected_layer_for_langs,
            "curves": curves_data,
        }, tt


def _save_pb_corr_table_and_plot(
    rpb_by_lang,
    p_by_lang,
    save_json_path,
    *,
    title="Accuracy vs CI (point-biserial correlation)",
    figsize=(7, 4.5),
):
    """
    Persist per-language point-biserial correlation between accuracy and CI
    (see the sign-flip note in plot_cumulative_all_layers_all_langs: CI =
    const - x_key, so r(accuracy, CI) = -r(accuracy, x_key)) as three
    artifacts next to `save_json_path`:
      * <stem>.json        — {lang: {"rpb": r, "p": p}}
      * <stem>.svg          — bar chart of r per language, significance-starred
      * <stem>_table.svg    — small Lang | r | p table, styled like the
                              cumulative-curve metrics table

    r > 0 means higher CI (i.e. lower x_key) predicts higher accuracy, which
    is the expected direction for an informative feature.
    """
    save_json_path = Path(save_json_path)
    langs = [l for l in rpb_by_lang if np.isfinite(rpb_by_lang[l])]
    if not langs:
        print(f"[skip] no finite point-biserial correlations to save at {save_json_path}")
        return
    langs = sorted(langs, key=lambda l: -abs(rpb_by_lang[l]))

    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json_path, "w") as f:
        json.dump(
            {l: {"rpb": rpb_by_lang[l], "p": p_by_lang.get(l, float("nan"))} for l in langs},
            f, indent=2,
        )
    print("Saved:", save_json_path)

    colors, _ = _pp_get_paper_styles(len(langs))

    # ── bar plot ──
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(langs))
    vals = [rpb_by_lang[l] for l in langs]
    ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for i, l in enumerate(langs):
        p = p_by_lang.get(l, float("nan"))
        star = "**" if np.isfinite(p) and p < 0.01 else ("*" if np.isfinite(p) and p < 0.05 else "")
        if star:
            y = vals[i]
            ax.text(i, y + (0.02 if y >= 0 else -0.02), star, ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Point-biserial r (accuracy, CI)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()

    plot_save_path = save_json_path.with_suffix(".svg")
    fig.savefig(plot_save_path, bbox_inches="tight", format="svg")
    print("Saved:", plot_save_path)
    plt.close(fig)

    # ── table ──
    table_save_path = save_json_path.with_name(save_json_path.stem + "_table.svg")
    fig, ax = plt.subplots(figsize=(3.2, 0.45 * len(langs) + 0.8))
    ax.axis("off")
    table_data = [[l, f"{rpb_by_lang[l]:.3f}", f"{p_by_lang.get(l, float('nan')):.3g}"] for l in langs]
    table = ax.table(cellText=table_data, colLabels=["Lang", "r (acc, CI)", "p"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.3)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("white")
    for j in range(3):
        table[(0, j)].set_text_props(weight="bold")
    for i, (l, color) in enumerate(zip(langs, colors), start=1):
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_alpha(0.22)

    fig.savefig(table_save_path, bbox_inches="tight", format="svg")
    print("Saved table:", table_save_path)
    plt.close(fig)


def _error_recall_auc_perm_p(correct_sorted, observed_auc_above, n_perm=200, seed=0):
    """
    Permutation test for error_recall_auc_above_diagonal: shuffles which examples
    are correct/incorrect (independent of score-based ordering) n_perm times,
    recomputing the AUC-above-diagonal each time as the null distribution;
    perm_p = fraction of null >= observed AUC (higher AUC = errors more
    front-loaded at low scores, so this is a one-sided test). Same pattern as
    compute_failure_pr_auc's perm_p elsewhere in this file.
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


def plot_error_recall_all_layers_all_langs(
    outs_by_layer,
    tt,
    *,
    valid_langs,
    y_mode="accuracy",
    y_acc_by_lang=None,
    y_layer="last",
    title: str | None = None,
    save_path: str | None = None,
    save_data_path: str | None = None,
    show: bool = True,
    fixed_layers_to_plot=None,  # if specified, use these layers directly on TEST
    lang_colors: dict | None = None,
    x_key: str = "x_ortho",
    n_perm: int = 200,
    random_seed: int = 0,
):
    """
    Cumulative error-recall curve, one curve per language, overlaid in a single figure.

    For each language: sort TEST examples by `x_key` ascending (lowest feature
    score first = predicted hardest first, matching the cumulative/non-cumulative
    convention used elsewhere in this file). X = fraction of test examples
    included; Y = fraction of that language's total errors captured among those
    examples.

    Random baseline: Y = X (diagonal) — including x% of examples at random
    captures x% of errors. A curve above the diagonal means errors are
    concentrated at low-score examples, i.e. the feature successfully
    front-loads failures.

    Layer selection: uses `fixed_layers_to_plot[lang]` if given, else falls back
    to `tt[lang]["lowest_AUC_layer"]` (set by a prior cumulative call on the same
    `tt`), else selects the layer with lowest TRAIN cumulative AUC (same
    convention as plot_cumulative_all_layers_all_langs).

    Also writes a `save_data_path` JSON with the per-language sorted
    (x, correct) rows and the error_recall_auc_above_diagonal / error_rate /
    perm_p metrics needed to regenerate this figure. perm_p is a permutation
    test on error_recall_auc_above_diagonal (see _error_recall_auc_perm_p):
    the fraction of times a random correct/incorrect shuffle achieves an
    AUC-above-diagonal at least as high as the observed one.
    """
    if len(outs_by_layer) == 0:
        print("outs_by_layer is empty.")
        return

    layers = sorted(outs_by_layer.keys())
    if len(layers) == 0:
        print("No layers found in outs_by_layer.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0
    curves_data = {}
    store_auc_above_for_langs = {}
    store_error_rate_for_langs = {}
    store_selected_layer_for_langs = {}
    store_perm_p_for_langs = {}

    def _resolve_y_all_for_lang(lang, common_ids):
        y_acc_default = None
        for L_probe in layers:
            if L_probe not in outs_by_layer or lang not in outs_by_layer[L_probe]:
                continue
            pack_probe = outs_by_layer[L_probe][lang]
            if isinstance(pack_probe, dict) and ("y_accuracy" in pack_probe):
                idx_y = np.asarray(pack_probe["indices"], dtype=np.int64)
                pos_y = {int(ex_id): i for i, ex_id in enumerate(idx_y)}
                y_arr_acc = np.asarray(pack_probe["y_accuracy"], dtype=np.float64)
                if all(int(ex_id) in pos_y for ex_id in common_ids):
                    y_acc_default = np.asarray(
                        [float(y_arr_acc[pos_y[int(ex_id)]]) for ex_id in common_ids],
                        dtype=np.float64,
                    )
                    break
        y_all, _ = _resolve_curve_y_from_pack(
            lang=lang,
            y_default=np.full(len(common_ids), np.nan, dtype=np.float64),
            idxs=common_ids, y_acc_by_lang=y_acc_by_lang,
            y_accuracy_default=y_acc_default, require_len=len(common_ids),
        )
        return y_all

    def _x_all_for_layer(lang, layer, common_ids):
        if layer not in outs_by_layer or lang not in outs_by_layer[layer]:
            return None
        pack = outs_by_layer[layer][lang]
        if x_key not in pack:
            return None
        idx_x = np.asarray(pack["indices"], dtype=np.int64)
        pos_x = {int(ex_id): i for i, ex_id in enumerate(idx_x)}
        if not all(int(ex_id) in pos_x for ex_id in common_ids):
            return None
        return np.asarray(
            [float(pack[x_key][pos_x[int(ex_id)]]) for ex_id in common_ids], dtype=np.float64,
        )

    def _train_auc_for_layer(lang, layer, common_ids, train_mask, y_all):
        x_all = _x_all_for_layer(lang, layer, common_ids)
        if x_all is None:
            return np.nan
        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        m = np.isfinite(x_train) & np.isfinite(y_train)
        if int(m.sum()) < 2:
            return np.nan
        order = np.argsort(x_train[m])
        y_sorted_train = y_train[m][order]
        return _cumulative_auc_from_sorted_y(y_sorted_train)

    for lang in valid_langs:
        if lang not in tt:
            continue
        d = tt[lang]
        common_ids = np.asarray(d["common_ids"], dtype=np.int64)
        train_mask = np.asarray(d["train_mask"], dtype=bool)
        test_mask = np.asarray(d["test_mask"], dtype=bool)
        if common_ids.size == 0:
            continue

        y_all = _resolve_y_all_for_lang(lang, common_ids)
        if y_all is None:
            continue

        best_layer = None
        if fixed_layers_to_plot is not None:
            best_layer = fixed_layers_to_plot.get(lang, None)
            if best_layer is None:
                continue
            best_layer = int(best_layer)
            if best_layer not in outs_by_layer or lang not in outs_by_layer[best_layer]:
                continue
        elif "lowest_AUC_layer" in d:
            best_layer = int(d["lowest_AUC_layer"])
        else:
            best_train_auc = np.inf
            for L in layers:
                if L not in outs_by_layer or lang not in outs_by_layer[L]:
                    continue
                auc_tr = _train_auc_for_layer(lang, L, common_ids, train_mask, y_all)
                if np.isfinite(auc_tr) and auc_tr < best_train_auc:
                    best_train_auc = float(auc_tr)
                    best_layer = int(L)

        if best_layer is None:
            continue
        store_selected_layer_for_langs[lang] = int(best_layer)

        x_all = _x_all_for_layer(lang, best_layer, common_ids)
        if x_all is None:
            continue

        x_test = x_all[test_mask]
        y_test = y_all[test_mask]
        m_te = np.isfinite(x_test) & np.isfinite(y_test)
        if int(m_te.sum()) == 0:
            continue
        x_test = x_test[m_te]
        y_test = y_test[m_te]

        order = np.argsort(x_test)
        x_sorted = x_test[order]
        correct_sorted = y_test[order]
        n = int(correct_sorted.size)
        total_errors = int(np.sum(1 - correct_sorted))

        if total_errors == 0:
            store_error_rate_for_langs[lang] = 0.0
            store_auc_above_for_langs[lang] = float("nan")
            store_perm_p_for_langs[lang] = float("nan")
            curves_data[lang] = {
                "x_sorted": x_sorted.tolist(), "correct_sorted": correct_sorted.tolist(),
                "layer": int(best_layer), "error_rate": 0.0,
                "error_recall_auc_above_diagonal": None, "perm_p": None, "n": n, "total_errors": 0,
            }
            continue

        xs = np.arange(1, n + 1) / n
        ys = np.cumsum(1 - correct_sorted) / total_errors
        error_rate = total_errors / n
        auc_above = float(np.trapz(ys - xs, xs))
        perm_p = _error_recall_auc_perm_p(
            correct_sorted, auc_above, n_perm=n_perm, seed=random_seed + 101 * best_layer,
        )
        perm_p_str = f" | p(perm)={perm_p:.3f}" if np.isfinite(perm_p) else ""

        store_error_rate_for_langs[lang] = float(error_rate)
        store_auc_above_for_langs[lang] = auc_above
        store_perm_p_for_langs[lang] = perm_p

        label = f"{lang}(L{best_layer}) | AUC above diag={auc_above:.3f} | error_rate={error_rate:.3f}{perm_p_str}"
        curves_data[lang] = {
            "x_sorted": x_sorted.tolist(), "correct_sorted": correct_sorted.tolist(),
            "xs": xs.tolist(), "ys": ys.tolist(), "label": label, "layer": int(best_layer),
            "error_rate": float(error_rate), "error_recall_auc_above_diagonal": auc_above,
            "perm_p": perm_p, "n": n, "total_errors": total_errors,
        }

        _color = lang_colors.get(lang) if lang_colors else None
        ax.plot(xs, ys, label=label, alpha=0.9, **({} if _color is None else {"color": _color}))
        plotted += 1

    if plotted == 0:
        print("No error-recall curves to plot.")
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5, label="Random baseline (x=y)")
    ax.set_xlabel("Fraction of examples included")
    ax.set_ylabel("Fraction of total errors captured")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)

    if title is None:
        title = "Error Recall Curve"
    ax.set_title(title)

    ax.legend(
        ncol=1, fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="svg")
        print("Saved:", save_path)

    if save_data_path is not None:
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        _plot_meta = {
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "curves": curves_data,
        }
        with open(save_data_path, "w") as _f:
            json.dump(_plot_meta, _f, indent=2)
        print("Saved data:", save_data_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "error_recall_auc_above_diagonal": store_auc_above_for_langs,
        "perm_p": store_perm_p_for_langs,
        "error_rate": store_error_rate_for_langs,
        "selected_layer": store_selected_layer_for_langs,
        "curves": curves_data,
    }


def _resolve_relation(idx, lang, rel_map_targets=None, rel_map_preds=None):
    """
    Resolve relation using (index, lang), preferring targets then preds.
    """
    key = (int(idx), str(lang))

    if rel_map_targets is not None and key in rel_map_targets:
        return rel_map_targets[key]

    if rel_map_preds is not None and key in rel_map_preds:
        return rel_map_preds[key]

    return None


def _pa_cosines(W1, W2):
    """Cosines of principal angles = singular values of W1^T W2, clipped to [0,1]."""
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    if W1.ndim != 2 or W2.ndim != 2 or W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.array([], dtype=np.float64)
    s = np.linalg.svd(W1.T @ W2, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def _pairwise_coherence(W1, W2):
    """
    Max over per-basis-vector mean abs-cosines between two orthonormal bases.
    For each column v_i in W1: mean(|v_i·u_j| for j). Symmetric over both sides.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2)
    return float(np.max(np.concatenate([M.mean(axis=1), M.mean(axis=0)])))


def _pairwise_coherence_weighted(W1, W2, sv1, sv2, weight_mode="sv"):
    """
    Like _pairwise_coherence but |v_i·u_j| is weighted by the target vector's
    singular value: sv2[j] for W1-row means, sv1[i] for W2-col means.
    weight_mode: "sv" normalised by sum(sv); "sv_squared" normalised by sum(sv²);
                 "none": direct sv-weighted sum, no normalisation.
    Pass np.ones(k) when singular values are unavailable.
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
    Like _pairwise_coherence but aggregates per basis vector with sqrt(mean(cosine^4)).
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2) ** 4  # (k1, k2)
    return float(np.max(np.concatenate([np.sqrt(M.mean(axis=1)), np.sqrt(M.mean(axis=0))])))


def _pairwise_coherence_sum(W1, W2):
    """
    Like _pairwise_coherence but uses sum instead of mean — scales with subspace dimensionality.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    M = np.abs(W1.T @ W2)  # (k1, k2)
    return float(np.max(np.concatenate([M.sum(axis=1), M.sum(axis=0)])))


def _pairwise_coherence_full_set_mean(W1, W2):
    """
    Max over per-basis-vector 'full-set mean' abs-cosines.

    For each v_i in W1: sum(|v_i·u_j| for j in W2) / (k1+k2-1).
    Denominator treats the k1-1 within-subspace pairs as 0 (guaranteed by SVD
    orthonormality), giving the true mean over all k1+k2-1 other basis vectors.
    Same denominator holds symmetrically for W2 vectors.
    """
    if W1.shape[1] == 0 or W2.shape[1] == 0:
        return np.nan
    k1, k2 = W1.shape[1], W2.shape[1]
    M = np.abs(W1.T @ W2)  # (k1, k2)
    denom = k1 + k2 - 1
    row_scores = M.sum(axis=1) / denom
    col_scores = M.sum(axis=0) / denom
    return float(np.max(np.concatenate([row_scores, col_scores])))


def _vec_subspace_cos(v, W, eps=1e-12):
    """Cosine = ||proj_W(v)|| / ||v||, alignment of v with subspace W."""
    v = np.asarray(v, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv < eps or W.ndim != 2 or W.shape[1] == 0:
        return np.nan
    proj = W @ (W.T @ v)
    return float(np.linalg.norm(proj) / nv)


def _build_svd_subspace(X, var_prop=0.95, center=False, eps=1e-12):
    """
    Build SVD subspace from X [n, d].
    center=False -> uncentered SVD on raw X (not mean-subtracted).
    Returns (mu [d], W [d, k]) where W has orthonormal columns.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    X_use = X - mu if center else X
    if n <= 1 or np.allclose(X_use, 0.0):
        return mu, np.zeros((d, 0), dtype=np.float64)
    U, S, Vt = np.linalg.svd(X_use, full_matrices=False)
    sv2 = S ** 2
    cum = np.cumsum(sv2) / np.clip(sv2.sum(), eps, None)
    k = int(np.searchsorted(cum, var_prop) + 1)
    W = Vt[:k, :].T.copy()                   # [d, k]
    return mu, _orthonormalize(W), S[:k]      # _orthonormalize available from cell 0


def _affine_vec_to_subspace_ortho(v, mu_L, W_L, eps=1e-12):
    """
    Affine subspace orthogonality:
        ||(I - W W^T)(v - mu_L)|| / ||v - mu_L||
    Returns the fraction of (v - mu_L) that lies outside W_L.
    """
    v = np.asarray(v, dtype=np.float64)
    mu_L = np.asarray(mu_L, dtype=np.float64)
    W_L = np.asarray(W_L, dtype=np.float64)
    delta = v - mu_L
    denom = np.linalg.norm(delta)
    if denom < eps or not np.isfinite(denom):
        return np.nan
    if W_L.ndim != 2 or W_L.shape[1] == 0:
        return np.nan
    proj = W_L @ (W_L.T @ delta)
    resid = delta - proj
    return float(np.linalg.norm(resid) / denom)


def _subspace_metrics(W_dataset, mu_dataset, W_L, mu_L=None, sv_weights=None):
    """All orthogonality metrics between a relation subspace and W_L."""
    _nan_keys = ["ortho_max", "ortho_min", "ortho_mean",
                 "ortho_principal_max", "ortho_principal_min", "ortho_principal_mean",
                 "ortho_weighted_mean", "ortho_fro", "mu_ortho", "affine_ortho"]
    s = _pa_cosines(W_dataset, W_L)
    if s.size == 0:
        return {k: np.nan for k in _nan_keys}
    affine = (
        _affine_vec_to_subspace_ortho(mu_dataset, mu_L, W_L)
        if (mu_L is not None and np.asarray(mu_L).shape == np.asarray(mu_dataset).shape)
        else np.nan
    )
    # Weight by data singular values (variance importance); truncate to len(s) if needed
    if sv_weights is not None and len(sv_weights) > 0:
        w = np.asarray(sv_weights[:len(s)], dtype=np.float64)
    else:
        w = s
    w_sum = w.sum()
    weighted_mean = float(np.dot(w, 1.0 - s) / w_sum) if w_sum > 0 else float(np.mean(1.0 - s))

    # Corrected mean ortho over all SUM*(SUM-1) ordered pairs, where SUM = k1+k2.
    # Within-subspace pairs are orthonormal so similarity = 0 (no explicit contribution).
    # Numerator = sum(|W_dataset^T W_L|) over k1*k2 cross pairs.
    # ortho_mean = 1 - sum(M) / (SUM*(SUM-1))
    k1 = int(W_dataset.shape[1])
    k2 = int(W_L.shape[1])
    SUM = k1 + k2
    M_cross = np.abs(np.asarray(W_dataset, dtype=np.float64).T @ np.asarray(W_L, dtype=np.float64))
    ortho_mean_corrected = float(1.0 - M_cross.sum() / (SUM * (SUM - 1))) if SUM >= 2 else np.nan

    return {
        "ortho_max":           float(1.0 - float(M_cross.max())),
        "ortho_min":           float(1.0 - float(M_cross.min())),
        "ortho_mean":          ortho_mean_corrected,
        "ortho_principal_max": float(1.0 - float(np.max(s))),
        "ortho_principal_min": float(1.0 - float(np.min(s))),
        "ortho_principal_mean": float(1.0 - float(np.mean(s))),
        "ortho_weighted_mean": weighted_mean,
        "ortho_fro":           float(1.0 - np.mean(s ** 2)),
        "mu_ortho":            float(1.0 - _vec_subspace_cos(mu_dataset, W_L)),
        "affine_ortho":        affine,
    }


# -----------------------------------------------------------------------
# NEW HELPERS: Oscar split loading + Exp 2/3 relation-subspace analysis
# -----------------------------------------------------------------------

def load_oscar_W_for_layer(
    layer: int,
    valid_langs: list,
    lang_method: str,
    varp: float,
    oscar_resids_root: str,
    oscar_cache_root: str,
    max_oscar_rows: int,
    oscar_split: str,
    abbr_to_full_map: dict,
    verbose: bool = False,
):
    """
    Load Oscar language subspace W and mu for all valid_langs at a given layer.
    Returns (W_by_lang, mu_by_lang) dicts: lang_abbr -> np.ndarray or None.
    """
    W_by_lang, mu_by_lang, sv_by_lang = {}, {}, {}
    for L in valid_langs:
        full = abbr_to_full_map.get(L, L)
        WL, muL, svL = oscar_W_cached(
            lang=full, layer=layer,
            subspace_method=lang_method, var_prop=varp,
            oscar_resids_root=oscar_resids_root,
            disk_cache_root=oscar_cache_root,
            max_rows=max_oscar_rows,
            verbose=verbose,
            oscar_split=oscar_split,
        )
        W_by_lang[L] = None if WL is None else _orthonormalize(np.asarray(WL, dtype=np.float64))
        mu_by_lang[L] = None if muL is None else np.asarray(muL, dtype=np.float64)
        sv_by_lang[L] = None if svL is None else np.asarray(svL, dtype=np.float64)
    return W_by_lang, mu_by_lang, sv_by_lang


def compute_relation_subspace_metrics_for_layer(
    pack: dict,
    W_L: np.ndarray,
    mu_L: np.ndarray,
    *,
    var_prop_en: float = 0.90,
    center: bool = False,
    min_rel_n: int = 5,
    sv_WL: np.ndarray | None = None,
    sv_weight_mode: str = "sv",
):
    """
    For each relation group in pack, build a SVD subspace from X_base vectors,
    then compute the cumulative-coherence pairwise metric against W_L.

    Returns dict: rel_str -> {'n', 'y_mean', 'subspace_subspace_angle_by_cumulative_coherence'}
    """
    X_base = np.asarray(pack.get("X_base", []), dtype=np.float64)
    y_vals = np.asarray(pack.get("y", []), dtype=np.float64)
    rel_arr = np.asarray(pack.get("relation", []), dtype=object)

    if X_base.ndim != 2 or X_base.shape[0] == 0:
        return {}

    W_L = _orthonormalize(np.asarray(W_L, dtype=np.float64))
    if W_L.shape[1] == 0:
        return {}

    result = {}
    unique_rels = sorted({str(r) for r in rel_arr.tolist() if r is not None and str(r) != "None"})
    for rel in unique_rels:
        mask = (rel_arr == rel)
        X_rel = X_base[mask]
        y_rel = y_vals[mask]
        finite_rows = np.isfinite(X_rel).all(axis=1) & np.isfinite(y_rel)
        X_rel = X_rel[finite_rows]
        y_rel = y_rel[finite_rows]
        if X_rel.shape[0] < min_rel_n:
            continue
        mu_rel, W_rel, S_rel = _build_svd_subspace(X_rel, var_prop=var_prop_en, center=center)
        if W_rel.shape[1] == 0:
            continue

        coh = _pairwise_coherence(W_rel, W_L)

        result[str(rel)] = {
            "n": int(X_rel.shape[0]),
            "y_mean": float(np.mean(y_rel)),
            "subspace_subspace_angle_by_cumulative_coherence": float(1.0 - abs(coh)) if np.isfinite(coh) else np.nan,
        }
    return result


def compute_whole_lang_subspace_metrics(pack, W_L, mu_L, var_prop=0.90, center=True, min_n=5):
    """
    Build one SVD subspace from all X_base in pack (all relations merged), compute
    _subspace_metrics vs W_L.  Returns {'n', 'y_mean', ...metrics...} or None.
    """
    X_base = np.asarray(pack.get("X_base", []), dtype=np.float64)
    y_vals = np.asarray(pack.get("y", []), dtype=np.float64)
    if X_base.ndim != 2 or X_base.shape[0] < min_n:
        return None
    W_L = _orthonormalize(np.asarray(W_L, dtype=np.float64))
    if W_L.shape[1] == 0:
        return None
    fin = np.isfinite(X_base).all(axis=1) & np.isfinite(y_vals)
    X_base = X_base[fin]; y_vals = y_vals[fin]
    if X_base.shape[0] < min_n:
        return None
    mu_enfact, W_enfact, S_enfact = _build_svd_subspace(X_base, var_prop=var_prop, center=center)
    if W_enfact.shape[1] == 0:
        return None
    mets = _subspace_metrics(W_enfact, mu_enfact, W_L, mu_L=mu_L, sv_weights=S_enfact)
    return {"n": int(X_base.shape[0]), "y_mean": float(np.mean(y_vals)), **mets}


def _plot_exp4_cumulative(
    x_test: np.ndarray,
    y_test: np.ndarray,
    best_layer: int,
    title: str,
    save_path: str | None = None,
    save_data_path: str | None = None,
    quantile_step: float = 0.1,
    y_lim: tuple = (0.0, 1.0),
    auc_fmt: str = ".3f",
    show: bool = False,
) -> None:
    """Plot cumulative trend for merged-language pool with permutation test."""
    x_test = np.asarray(x_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    m = np.isfinite(x_test) & np.isfinite(y_test)
    x_test = x_test[m]
    y_test = y_test[m]
    if len(x_test) < 10:
        print(f"[Exp4] not enough valid points ({len(x_test)})")
        return

    order = np.argsort(x_test)
    y_sorted_test = y_test[order]

    steps = np.arange(quantile_step, 1.0 + 1e-9, quantile_step)
    xs, ys = _curve_values_from_sorted_y(y_sorted_test, steps, cumulative_mode="cumulative")
    if len(xs) < 2:
        print("[Exp4] not enough quantile steps")
        return

    summ = _auc_summary_lower_auc_better_from_sorted_y(y_sorted_test)
    test_auc = summ["test_auc"]
    base_auc = summ["baseline_auc"]
    pp = (1.0 - test_auc / base_auc) if (np.isfinite(base_auc) and base_auc > 1e-9) else float("nan")

    rng = np.random.default_rng(42)
    n_perm = 1000
    null_aucs = np.empty(n_perm)
    for _pi in range(n_perm):
        x_perm = rng.permutation(x_test)
        null_aucs[_pi] = _cumulative_auc_from_sorted_y(y_test[np.argsort(x_perm)])
    perm_p = float(np.mean(null_aucs <= test_auc))
    perm_p_str = "p(perm)<0.01" if perm_p < 0.01 else f"p(perm)={perm_p:.3f}"

    label = (
        f"all langs merged (L{best_layer}) | auc={test_auc:{auc_fmt}} | "
        f"avg_acc={base_auc:{auc_fmt}} | PP={pp:{auc_fmt}} | {perm_p_str}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, label=label, alpha=0.9)
    ax.set_xlabel("Prefix/bin fraction (lowest feature first)")
    ax.set_ylabel("Cumulative mean accuracy")
    ax.set_xlim(quantile_step - 1e-9, 1.0)
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", format="svg")
        print("Saved:", save_path)

    if save_data_path is not None:
        os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
        _meta = {
            "title": ax.get_title(), "x_label": ax.get_xlabel(),
            "y_label": "Cumulative mean accuracy",
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "cumulative_mode": "cumulative",
            "curves": {"all_langs": {
                "xs": list(xs), "ys": list(ys), "label": label, "layer": int(best_layer),
                "auc": float(test_auc), "avg_acc": float(base_auc),
                "pp": float(pp), "perm_p": float(perm_p),
            }},
        }
        with open(save_data_path, "w") as _f:
            json.dump(_meta, _f, indent=2)
        print("Saved data:", save_data_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


import types

available_models = ['llama-1b','llama-3b','llama-7b','llama-8b','Qwen2_1.5b','Qwen2-0.5B','qwen_14b']

# parse_known_args (not parse_args) so this still runs unmodified under a
# Jupyter kernel, which injects its own argv (e.g. "-f kernel-xxx.json") that
# would otherwise make argparse error out and kill the kernel.
_arg_parser = argparse.ArgumentParser(description="Geometric feature experiment (merged languages)")
_arg_parser.add_argument(
    "--model", type=str, default="llama-3b",
    help=f"Model to run the experiment for (default: llama-3b). One of {available_models} "
         "(aliases like 'llama_3b'/'qwen-14b' are also accepted).",
)
_args, _unknown_argv = _arg_parser.parse_known_args()
MODEL = _args.model

# Normalize interchangeable spellings to the canonical key used by _model_configs
# and all on-disk paths below (oscar_resids_root, oscar_cache_root, RUN_DIR, ...).
_model_aliases = {
    'llama_3b': 'llama-3b', 'llama-3b': 'llama-3b',
    'llama_7b': 'llama-7b', 'llama-7b': 'llama-7b',
    'qwen2.5_14b': 'qwen_14b', 'qwen_14b': 'qwen_14b', 'qwen-14b': 'qwen_14b',
    'qwen2.5-14b': 'qwen_14b',
}
MODEL = _model_aliases.get(MODEL.lower(), MODEL)

# Directories on disk aren't always saved under the canonical spelling above
# (e.g. an older run saved "oscar_geometry_Qwen2.5-14B" instead of the
# canonical "oscar_geometry_qwen_14b"). Rather than requiring a rename,
# _resolve_existing_model_dir() checks every known equivalent spelling and
# uses whichever one actually exists on disk.
_MODEL_DIR_ALIASES = {
    'llama-3b': ['llama-3b', 'llama_3b'],
    'llama-7b': ['llama-7b', 'llama_7b'],
    'qwen_14b': ['qwen_14b', 'qwen-14b', 'Qwen2.5-14B', 'Qwen2.5_14B', 'qwen2.5-14b', 'qwen2.5_14b'],
}


def _resolve_existing_model_dir(dir_template: str, model_canonical: str) -> str:
    """
    dir_template: a path containing a literal '{MODEL}' placeholder, e.g.
        'oscar_geometry_{MODEL}/all_one_lan_data_resid_pre_oscar2109'
    Tries the canonical spelling first, then every known equivalent spelling
    for model_canonical, and returns the first one that exists as a directory.
    Falls back to the canonical path if none exist (so missing-data errors
    still report the canonical path as before).
    """
    aliases = _MODEL_DIR_ALIASES.get(model_canonical, [model_canonical])
    ordered = [model_canonical] + [a for a in aliases if a != model_canonical]
    for name in ordered:
        candidate = dir_template.format(MODEL=name)
        if os.path.isdir(candidate):
            if name != model_canonical:
                print(f"[info] MODEL='{model_canonical}': using existing dir under equivalent name '{name}' -> {candidate}")
            return candidate
    return dir_template.format(MODEL=model_canonical)

# Configs only — no weights loaded, runs on CPU
_model_configs = {
    'llama-1b':     dict(n_layers=16, d_model=2048, n_heads=32, dtype='float32'),
    'llama-3b':     dict(n_layers=28, d_model=3072, n_heads=24, dtype='float32'),
    'llama-7b':   dict(n_layers=32, d_model=4096, n_heads=32, dtype='float32'),
    'llama-8b':     dict(n_layers=32, d_model=4096, n_heads=32, dtype='float32'),
    'Qwen2_1.5b':   dict(n_layers=28, d_model=1536, n_heads=12, dtype='float32'),
    'Qwen2-0.5B':   dict(n_layers=24, d_model=896,  n_heads=14, dtype='float32'),
    'qwen_14b':  dict(n_layers=48, d_model=5120, n_heads=40, dtype='float32'),
}

# OSCAR cache roots (yours)
oscar_resids_root = _resolve_existing_model_dir(
    'oscar_geometry_{MODEL}/all_one_lan_data_resid_pre_oscar2109', MODEL,
)
oscar_cache_root = _resolve_existing_model_dir('oscar_subspaces_cache_{MODEL}', MODEL)
max_oscar_rows    = 8000

cfg = types.SimpleNamespace(**_model_configs[MODEL])
model = types.SimpleNamespace(cfg=cfg)
tokenizer = None

FIRST_ID_INDEX_LLAMA = 1
device = 'cpu'



#LOADING JENNIFER

auc_tables = defaultdict(dict)
from collections import defaultdict
from multilingual_experiment_auc_plots import load_auc_matrix, plot_auc_heatmap, plot_auc_diagsum_and_rowsum_bars
valid_langs = ["es","fr","hu","ja","ko","nl","ru","uk","vi","zh"]#,"en"]
print(len(valid_langs))


#  lang_mode = ('center_oscar_and_uncentered_language_subspace', 'SVD', 0.99)        
#  lang_mode = ('center_oscar_and_language_subspace', 'SVD', 0.99)           
#  lang_mode = ('center_oscar_and_language_subspace_meanshifted', 'SVD', 0.99)       

#language_subspace_meanshifted
lang_mode =('center_oscar_and_uncentered_language_subspace', 'SVD', 0.99) #('center_oscar_and_language_subspace',#("language_subspace","SVD",0.99)#language_subspace_mean_vector #("language_subspace", "SVD", 0.99)#("language_subspace_meanshifted", "SVD", 0.99) #"translation_vector_from_english_and_into_english"#"translation_vector_into_english" #("language_subspace_meanshifted", "SVD", 0.99)# ["translation_vector_from_english"]#[("language_subspace", "SVD", 0.95)]#,"translation_vector_into_english",]"translation_vector_from_english",
mean_center_by_cluster=True #TODO: should set as TRUE

target_kind,y_transform=[("binary_correct","identity")][0]#,("prob","identity"),("rec_rank","identity"),("rec_rank","log1p")]
shot_tag=possible_shot_tag ='zeroshot'#zeroshot
prompt_num=0#range(5)
feature_list = [
    'en_fact_norm', 'en_fact_density', 'en_fact_max_coordinate_fraction',
    'vec_subspace_orthogonality', 'vec_subspace_angle_radians',
    'squared_energy_projection', 'residual_energy_fraction',
    'projection_coefficient_l1_over_l2', 'projection_max_fraction', 'projection_entropy',
    # 1-abs(cos) convention
    'vec_subspace_mean', 'vec_subspace_min', 'vec_subspace_mean_weighted_by_variance',
    'vec_subspace_angle_by_cumulative_coherence',
    'vec_subspace_angle_by_cumulative_coherence_L2',
    'vec_subspace_angle_by_cumulative_coherence_weighted_by_singular_values',


]
base_or_random_mode = 'en_fact'#'en_fact'#['en_fact'][0]
plot_per_dataset = True
plot_per_layer = True
rep_kind='last'



shot_tag_num = '0shot' if shot_tag == 'zeroshot' else '3shot'
RUN_DIR = f"klar/{MODEL}_eval_save_all_layers_prompt{prompt_num}/{MODEL}_{shot_tag_num}"

path = Path(os.path.join(RUN_DIR,f"predictions_{shot_tag_num}.jsonl"))
TARGETS_JSONL = str(path)  # binary_correct target_kind reads "match" straight from predictions jsonl
if "translation_vector" in lang_mode:
    direction =lang_mode.split("translation_vector_")[-1]
    print(direction)
    feature='vec_vec_orthogonality'
    base_or_random_mode_list = ['en_fact']

if isinstance(lang_mode, (list, tuple)):#subspace
    lang_mode_name = "_".join(str(x) for x in lang_mode)
    valid_langs = ['en'] + valid_langs
    if "mean_vector" in lang_mode[0]:
        feature ='vec_vec_orthogonality'
    else:
        feature='vec_subspace_angle_by_cumulative_coherence'
    base_or_random_mode_list = ['en_fact']
else:
    lang_mode_name = str(lang_mode)
    



if "translation_vector" in lang_mode:
    translation_data_dirs = {"translation_vector_from_english":f"translation_common_words/llama_3b_eval_save_all_layers_resid_and_logits_from_english_cap500/llama-3b_{shot_tag_num}",
                             "translation_vector_into_english":f"translation_common_words/llama_3b_eval_save_all_layers_resid_and_logits_into_english_cap500/llama-3b_{shot_tag_num}"}

else:
    translation_data_dirs=None

# Single source of truth for this run's experiment root dir — every plot_root
# below is a subdirectory of this. Includes MODEL so different models never
# overwrite each other's plots.
#EXP_ROOT_DIR = os.path.join(
#    "weekly_meetings", f"geometric_features_experiments_{MODEL}",
#    f"prompt{prompt_num}_{shot_tag}_{rep_kind}_{lang_mode_name}_{target_kind}_{y_transform}_centerenfacts{mean_center_by_cluster}",
#)

EXP_ROOT_DIR = os.path.join(
    f"multilingual_experiments_{MODEL}",
    f"prompt{prompt_num}_{shot_tag}_{rep_kind}_{lang_mode_name}_{target_kind}_{y_transform}_centerenfacts{mean_center_by_cluster}",
)


def _auc_out_to_pp(auc_out: dict) -> dict:
    """Convert plot_cumulative_all_layers_all_langs output to PP = 1 - test_auc/baseline_auc."""
    test_aucs = auc_out.get("test_auc", {})
    base_aucs = auc_out.get("baseline_auc", {})
    pp = {}
    for lang, ta in test_aucs.items():
        ba = base_aucs.get(lang, float("nan"))
        if np.isfinite(ta) and np.isfinite(ba) and ba > 1e-9:
            pp[lang] = 1.0 - ta / ba
        else:
            pp[lang] = float("nan")
    return pp


# ── Failure PR-AUC helpers ────────────────────────────────────────────────────
from sklearn.metrics import average_precision_score as _avg_prec_score
from sklearn.metrics import precision_recall_curve as _prec_recall_curve


def compute_failure_pr_auc(x, y_correct, n_perm=200, n_boot=0, ci=0.95, seed=0):
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
        print("Saved:", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _compute_and_save_pr_auc_all_langs(
    outs_by_layer, tt, valid_langs, best_layers, y_acc_by_lang, save_dir, label, n_perm=200,
    n_boot=0, ci=0.95, seed=0,
    x_key: str = "x_ortho", plot_curves: bool = True, summary_filename: str = "pr_auc_summary.json",
):
    """Compute failure PR-AUC per language at best_layers[lang] and save summary JSON
    (+ a per-language precision-recall curve svg, unless plot_curves=False)."""
    if y_acc_by_lang is None:
        print(f"[PR-AUC {label}] skipping: y_acc_by_lang is None")
        return {}

    os.makedirs(save_dir, exist_ok=True)
    summary = {}

    for lang in valid_langs:
        best_layer = best_layers.get(lang)
        if best_layer is None or lang not in tt:
            continue
        common_ids = tt[lang].get("common_ids")
        test_mask = tt[lang].get("test_mask")
        if common_ids is None or test_mask is None:
            continue
        if best_layer not in outs_by_layer or lang not in outs_by_layer[best_layer]:
            continue
        pack = outs_by_layer[best_layer][lang]
        if x_key not in pack:
            continue
        idx_x = np.asarray(pack["indices"], dtype=np.int64)
        pos_x = {int(eid): i for i, eid in enumerate(idx_x)}
        if not all(int(eid) in pos_x for eid in common_ids):
            continue
        x = np.asarray([pack[x_key][pos_x[int(eid)]] for eid in common_ids], dtype=np.float64)
        if lang not in y_acc_by_lang:
            continue
        y_correct = np.asarray(y_acc_by_lang[lang], dtype=np.float64)
        if len(y_correct) != len(x):
            continue
        # Held-out only: exclude the examples used to pick best_layer, so PR-AUC
        # is not evaluated on layer-selection data (mirrors plot_cumulative_all_layers_all_langs).
        x, y_correct = x[test_mask], y_correct[test_mask]

        pr_stats = compute_failure_pr_auc(x, y_correct, n_perm=n_perm, n_boot=n_boot, ci=ci, seed=seed)
        summary[lang] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in pr_stats.items()
        }
        summary[lang]["layer"] = int(best_layer)
        if plot_curves:
            plot_failure_pr_curve(
                pr_stats,
                save_path=os.path.join(save_dir, f"pr_curve_{lang}.svg"),
                show=False,
                title=f"[{label}] {lang} failure PR — layer {best_layer}",
            )

    summary_path = os.path.join(save_dir, summary_filename)
    with open(summary_path, "w") as _f:
        json.dump(summary, _f, indent=2)
    print(f"[PR-AUC {label}] saved summary: {summary_path}")
    return summary


_SRC_TGT_CSV_FIELDS = [
    "target", "source", "kind", "layer_select", "layer_used",
    "source_own_best_layer", "auc", "pr_auc", "normalized_pr_auc",
    "failure_rate", "n_examples",
]


def _src_tgt_permutation_test_report(
    rows, save_dir, tag, x_key, n_perm=100_000, seed=0,
    metrics=(("auc", False), ("pr_auc", True)),
):
    """Self-rank permutation test on the SRC x TGT matrix, mirroring
    random_seeds_imported_rankings.py / self_rank_permutation_from_csvs.py's
    diagonal_rank_permutation_test: build the k_s x k_t cross-eval matrix per
    metric (M[source, target]), rank each column (sign-flipped first for
    lower-is-better metrics), then test whether the diagonal (self) entries'
    mean within-column rank beats the null expectation under a random
    source<->target assignment, via a one-sided permutation test.

    The source set and target set need not be equal: some sources may be
    "extra" (present only as a row, e.g. an always-imported language that is
    never itself scored as a target) and therefore have no "self" cell. Every
    target must also appear as a source (so its self cell exists), but the
    reverse need not hold. When source and target sets are identical this
    reduces exactly to the original square k x k test.

    Writes a plain-text report (not just the CSV) so the statistical result
    is available without a separate script run. Returns the report path.
    """
    from scipy.stats import rankdata

    target_labels = sorted({r["target"] for r in rows})
    source_labels = sorted({r["source"] for r in rows})
    k_t = len(target_labels)
    k_s = len(source_labels)
    tidx = {label: i for i, label in enumerate(target_labels)}
    sidx = {label: i for i, label in enumerate(source_labels)}
    extra_sources = [s for s in source_labels if s not in tidx]
    missing_self = [t for t in target_labels if t not in sidx]
    if missing_self:
        raise ValueError(
            f"target(s) {missing_self} have no corresponding source row; "
            f"cannot define a self cell for them."
        )
    self_row_for_target = np.array([sidx[t] for t in target_labels])
    layer_select_label = rows[0].get("layer_select", "?") if rows else "?"

    lines = [
        "Exp1 SRC x TGT imported-subspace negative control — self-rank permutation test",
        "=" * 78,
        f"Ranking metric (x_key): {x_key}",
        f"Layer selection: {layer_select_label}"
        + (
            " (source's own dev-best layer, reused across all targets — "
            "confounds subspace identity with layer choice off-diagonal)"
            if layer_select_label == "best_layer_fit_src_dev"
            else " (per (source, target) pair: layer swept and picked to best "
            "fit the TARGET's own dev split using the source's subspace — "
            "layer co-varies with which subspace is imported, same as "
            "best_layer_fit_src_dev but fit against the target's dev data "
            "instead of the source's)"
            if layer_select_label == "best_layer_fit_tgt_dev"
            else ""
        ),
        f"Targets (k_t={k_t}): {', '.join(target_labels)}",
        f"Sources (k_s={k_s}): {', '.join(source_labels)}"
        + (f"  [extra, no self: {', '.join(extra_sources)}]" if extra_sources else ""),
        f"Permutations: {n_perm}, seed={seed}",
        "",
        "Method: for each metric below, build the k_s x k_t cross-eval matrix M where "
        "M[source, target] is that metric for the source language's subspace "
        "scored against the target language's facts (diagonal = self, where "
        "source==target). Rank each column over all k_s sources (lower-is-better "
        "metrics are sign-flipped first), then test whether the mean rank of the "
        "diagonal beats the null expectation of (k_s+1)/2 under a random "
        "injective assignment of targets to sources (a bijection when k_s==k_t, "
        "generalized to an injection when there are extra sources with no self "
        "cell), via a one-sided permutation test "
        "(p = fraction of permuted null stats >= observed).",
        "",
        "Caveat: this is an imported-subspace NEGATIVE CONTROL, not a clean "
        "ablation like the random-seed experiment. Swapping the language "
        "subspace also swaps the linguistic condition, and different language "
        "subspaces may share geometry, so a significant own-language effect "
        "shows CI captures some target-language-specific interference, not "
        "full isolation from training-data statistics.",
        "",
    ]

    for metric, higher_is_better in metrics:
        M = np.full((k_s, k_t), np.nan)
        for row in rows:
            if row["source"] not in sidx or row["target"] not in tidx:
                continue
            M[sidx[row["source"]], tidx[row["target"]]] = row.get(metric, float("nan"))

        direction = "higher is better" if higher_is_better else "lower is better (sign-flipped for ranking)"
        lines.append(f"--- metric = {metric} ({direction}) ---")
        n_missing = int(np.isnan(M).sum())
        if n_missing:
            lines.append(f"  [skip] {n_missing}/{M.size} (source, target) entries missing; cannot run test.")
            lines.append("")
            continue

        R = rankdata(M if higher_is_better else -M, axis=0)
        diag_ranks = R[self_row_for_target, np.arange(k_t)]
        obs = float(diag_ranks.mean())
        null_mean = (k_s + 1) / 2.0

        rng = np.random.default_rng(seed)
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm = rng.permutation(k_s)[:k_t]
            null[i] = R[perm, np.arange(k_t)].mean()
        p_value = float((np.sum(null >= obs) + 1) / (len(null) + 1))

        lines.append(f"  mean self-rank = {obs:.3f} / {k_s}  (null mean = {null_mean:.3f})")
        lines.append(f"  p-value = {p_value:.4g}  (n_perm={n_perm})")
        lines.append(f"  per-language self rank ({k_s} = best in its column, 1 = worst; "
                     f"rankdata ranks ascending and only the values are sign-flipped "
                     f"for lower-is-better metrics, not which end is 'good'):")
        for i, lang in enumerate(target_labels):
            lines.append(f"    {lang:>6s}: {diag_ranks[i]:.1f} / {k_s}")
        lines.append("")

    report_path = os.path.join(save_dir, f"exp1_src_tgt_subspace_control_{tag}_permutation_test.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Exp1 SRC-TGT] saved permutation test report -> {report_path}")
    return report_path


def _src_tgt_row_from_pack(
    outs_src, target_lang, source_lang, layer_used, source_own_best_layer,
    tt_dev, y_acc_by_lang_test, x_key, min_n, layer_select_label,
):
    """Build one (source, target) row from a compute_geometric_feature output
    dict (outs_src), or return None if any prerequisite is missing/degenerate.
    Shared by both layer_select modes of compute_exp1_src_tgt_subspace_control.
    """
    if target_lang not in tt_dev or target_lang not in outs_src:
        return None
    common_ids = tt_dev[target_lang].get("common_ids")
    test_mask = tt_dev[target_lang].get("test_mask")
    if common_ids is None or test_mask is None or target_lang not in y_acc_by_lang_test:
        return None
    pack = outs_src[target_lang]
    if x_key not in pack:
        return None

    idx_x = np.asarray(pack["indices"], dtype=np.int64)
    pos_x = {int(eid): i for i, eid in enumerate(idx_x)}
    if not all(int(eid) in pos_x for eid in common_ids):
        return None
    x = np.asarray([pack[x_key][pos_x[int(eid)]] for eid in common_ids], dtype=np.float64)
    y_correct = np.asarray(y_acc_by_lang_test[target_lang], dtype=np.float64)
    if len(y_correct) != len(x):
        return None
    # Held-out only: common_ids/test_mask come from tt_dev (the target's own
    # dev-phase layer-selection split, reused as-is for src_dev and tgt_dev —
    # see compute_exp1_src_tgt_subspace_control), so this excludes exactly the
    # examples used to pick the target's best layer, matching Exp1's own
    # cumulative-curve AUC convention and the PR-AUC fix in
    # _compute_and_save_pr_auc_all_langs.
    x, y_correct = x[test_mask], y_correct[test_mask]

    m = np.isfinite(x) & np.isfinite(y_correct)
    if int(m.sum()) < min_n:
        return None
    x_m, y_m = x[m], y_correct[m]

    auc = _cumulative_auc_from_sorted_y(y_m[np.argsort(x_m)])
    pr_stats = compute_failure_pr_auc(x_m, y_m, n_perm=0, n_boot=0)

    kind = "self" if target_lang == source_lang else f"imported_from_{source_lang}"
    return {
        "target": target_lang, "source": source_lang, "kind": kind,
        "layer_select": layer_select_label,
        "layer_used": int(layer_used), "source_own_best_layer": int(source_own_best_layer),
        "auc": auc, "pr_auc": pr_stats["pr_auc"],
        "normalized_pr_auc": pr_stats["normalized_pr_auc"],
        "failure_rate": pr_stats["failure_rate"], "n_examples": int(m.sum()),
    }


def compute_exp1_src_tgt_subspace_control(
    *,
    model, run_dir, shot_tag, valid_langs, rep_kind, lang_mode,
    targets_jsonl_path, target_kind, y_transform, translation_data_dirs,
    feature, mean_center_by_cluster, rel_map_targets, rel_map_preds,
    sv_weight_mode, min_n, tt_dev, y_acc_by_lang_test, dev_best_layers,
    save_dir, tag, x_key="x_ortho", n_perm=100_000, perm_seed=0,
    layer_select="src_dev", source_langs=None, extra_dev_best_layers=None,
    dev_layers=None, dev_frac=0.1, x_bar="percentile", acc_map=None,
):
    """Imported-subspace negative control for Exp1: for every SOURCE language
    ell', score every TARGET language ell's en-facts with CI(a_q, B_ell') —
    i.e. reuse `fixed_subspace_{ell'}` mode (which already overrides every
    target's subspace with ell''s).

    `source_langs` (default: valid_langs) is the set looped over as SOURCE;
    TARGETS are always exactly `valid_langs`. Pass a superset (e.g.
    valid_langs + ["en"]) to always import an extra language's subspace into
    every target without that language ever being scored as a target itself
    (no self cell for it — see _src_tgt_permutation_test_report, which
    handles source_labels being a strict superset of target_labels). Best
    layers for any source in source_langs but not in valid_langs must be
    supplied via `extra_dev_best_layers` (e.g. {"en": 14}); dev_best_layers
    itself is left untouched and still drives `distinct_layers` for the
    "tgt_dev" mode (targets only, unaffected by extra sources).

    `layer_select` controls which layer that CI is evaluated at, since the two
    choices confound different things:
      - "src_dev" (default): use ell''s OWN dev-selected best layer, reused
        across every target — the same "best_layer_fit_src_dev" convention as
        random_seeds_imported_rankings.py. Self (ell'==ell) always lands on
        the target's own best layer since it's the same layer either way, but
        off-diagonal (imported) cells vary BOTH subspace identity and layer
        simultaneously as you move down a target's column — it is not a pure
        subspace-only ablation. Cheap: one compute_geometric_feature call per
        source (k calls total for the full k x k matrix).
      - "tgt_dev": mirror image of "src_dev" — sweep `dev_layers` with the
        SOURCE's subspace (fixed_subspace_{source}) and pick, independently
        for each target, whichever layer best fits THAT target's own dev
        split (same tt_select_best_layer_with_split PP-selection used
        upstream for dev_best_layers, just re-run per source against
        fixed_subspace_{source} features instead of each language's base
        features). So the layer is chosen per (source, target) pair using
        the target's dev data, not reused from the target's self-subspace
        best layer. On the diagonal (source==target) this necessarily
        recovers dev_best_layers[target], since fixed_subspace_{target} IS
        target's base subspace. Costs one compute_geometric_feature call per
        (source, dev layer) pair for the sweep (k * len(dev_layers) total)
        plus one call per (source, distinct selected target-layer) for
        scoring — substantially more expensive than "src_dev".

    `x_key` selects which CI feature drives the ranking — "x_ortho" is the
    cumulative-coherence default (any key computed by compute_geometric_feature
    works, but the pipeline only ever computes cumulative coherence, e.g.
    "vec_subspace_angle_by_cumulative_coherence"). Layer selection
    (dev_best_layers) is always the one chosen via x_ortho on the dev split —
    same convention the coherence plots elsewhere in Exp1 already use
    (fixed_layers_to_plot=dev_best_layers regardless of x_key).

    Writes (filenames include layer_select so both modes coexist):
      - a flat CSV (schema matches self_rank_permutation_from_csvs.py: target,
        source, kind, layer_used, source_own_best_layer, auc, pr_auc,
        normalized_pr_auc, failure_rate, n_examples) with one row per (source,
        target) pair; the diagonal (source==target) is the "self" row.
      - a .txt report running the same self-rank permutation test as
        random_seeds_imported_rankings.py / self_rank_permutation_from_csvs.py
        directly on this matrix (see _src_tgt_permutation_test_report), so the
        statistical result is available without a separate script run.

    This is a negative control, not a clean ablation: swapping the language
    subspace also swaps the linguistic condition, and different language
    subspaces may share geometry, so a significant own-language effect shows
    CI captures some target-language-specific interference, not that geometry
    is fully isolated from training-data statistics.
    """
    if layer_select not in ("src_dev", "tgt_dev"):
        raise ValueError("layer_select must be 'src_dev' or 'tgt_dev'")
    layer_select_label = (
        "best_layer_fit_src_dev" if layer_select == "src_dev" else "best_layer_fit_tgt_dev"
    )
    source_langs = list(source_langs) if source_langs is not None else list(valid_langs)
    src_dev_best_layers = dict(dev_best_layers)
    if extra_dev_best_layers:
        src_dev_best_layers.update(extra_dev_best_layers)

    rows = []

    if layer_select == "src_dev":
        for source_lang in source_langs:
            src_layer = src_dev_best_layers.get(source_lang)
            if src_layer is None:
                print(f"[Exp1 SRC-TGT] no dev-selected best layer for source={source_lang}; skipping")
                continue

            outs_src = compute_geometric_feature(
                model=model, run_dir=run_dir, shot_tag=shot_tag,
                valid_langs=valid_langs, layer=src_layer, rep_kind=rep_kind,
                lang_mode=lang_mode, targets_jsonl_path=targets_jsonl_path,
                target_kind=target_kind, y_transform=y_transform,
                translation_data_dirs=translation_data_dirs,
                min_n=min_n, separate_correct_incorrect_examples=False,
                feature=feature, base_or_random_mode=f"fixed_subspace_{source_lang}",
                random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
                rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
                oscar_split="test", store_X_base=False,
                sv_weight_mode=sv_weight_mode,
            )

            for target_lang in valid_langs:
                row = _src_tgt_row_from_pack(
                    outs_src, target_lang, source_lang, src_layer, src_layer,
                    tt_dev, y_acc_by_lang_test, x_key, min_n, layer_select_label,
                )
                if row is not None:
                    rows.append(row)

            del outs_src

    else:  # "tgt_dev"
        if dev_layers is None:
            # Fallback only: without an explicit sweep range we can't do a
            # real per-pair dev fit, so just reuse the targets' self-best
            # layers as candidates (old, degenerate behavior).
            dev_layers = sorted({dev_best_layers.get(t) for t in valid_langs} - {None})
            print("[Exp1 SRC-TGT] tgt_dev: no dev_layers given, falling back to "
                  f"distinct dev_best_layers as the sweep range: {dev_layers}")
        else:
            dev_layers = list(dev_layers)

        for source_lang in source_langs:
            src_own_layer = src_dev_best_layers.get(source_lang)
            if src_own_layer is None:
                print(f"[Exp1 SRC-TGT] no dev-selected best layer for source={source_lang}; skipping")
                continue

            # Sweep dev_layers with THIS source's subspace and, independently
            # per target, pick whichever layer best fits that target's own
            # dev split — the mirror image of "src_dev", which does the same
            # sweep/select but against the source's own dev split instead.
            outs_src_dev_by_layer = {}
            for L in dev_layers:
                outs_src_dev_by_layer[L] = compute_geometric_feature(
                    model=model, run_dir=run_dir, shot_tag=shot_tag,
                    valid_langs=valid_langs, layer=L, rep_kind=rep_kind,
                    lang_mode=lang_mode, targets_jsonl_path=targets_jsonl_path,
                    target_kind=target_kind, y_transform=y_transform,
                    translation_data_dirs=translation_data_dirs,
                    min_n=min_n, separate_correct_incorrect_examples=False,
                    feature=feature, base_or_random_mode=f"fixed_subspace_{source_lang}",
                    random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
                    rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
                    oscar_split="dev", store_X_base=False,
                    sv_weight_mode=sv_weight_mode,
                )

            tt_pair = tt_select_best_layer_with_split(
                outs_src_dev_by_layer, valid_langs=valid_langs,
                layers=dev_layers, seed=0, y_layer="last",
                min_n=min_n, train_frac=dev_frac, x_bar=x_bar, acc_map=acc_map,
            )
            del outs_src_dev_by_layer

            pair_layer = {
                t: int(info["best_layer"])
                for t, info in tt_pair.items()
                if info.get("best_layer") is not None
            }
            missing = [t for t in valid_langs if t not in pair_layer]
            if missing:
                print(f"[Exp1 SRC-TGT] source={source_lang}: no dev-fit layer for "
                      f"targets {missing}; skipping those cells")

            targets_by_layer = defaultdict(list)
            for target_lang, L in pair_layer.items():
                targets_by_layer[L].append(target_lang)

            for L, targets_at_L in targets_by_layer.items():
                outs_src = compute_geometric_feature(
                    model=model, run_dir=run_dir, shot_tag=shot_tag,
                    valid_langs=valid_langs, layer=L, rep_kind=rep_kind,
                    lang_mode=lang_mode, targets_jsonl_path=targets_jsonl_path,
                    target_kind=target_kind, y_transform=y_transform,
                    translation_data_dirs=translation_data_dirs,
                    min_n=min_n, separate_correct_incorrect_examples=False,
                    feature=feature, base_or_random_mode=f"fixed_subspace_{source_lang}",
                    random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
                    rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
                    oscar_split="test", store_X_base=False,
                    sv_weight_mode=sv_weight_mode,
                )

                for target_lang in targets_at_L:
                    row = _src_tgt_row_from_pack(
                        outs_src, target_lang, source_lang, L, src_own_layer,
                        tt_dev, y_acc_by_lang_test, x_key, min_n, layer_select_label,
                    )
                    if row is not None:
                        rows.append(row)

                del outs_src

    tag_full = f"{tag}_{layer_select}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"exp1_src_tgt_subspace_control_{tag_full}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SRC_TGT_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Exp1 SRC-TGT] saved {len(rows)} rows ({len(source_langs)} sources x {len(valid_langs)} targets, "
          f"layer_select={layer_select}, x_key={x_key}) -> {csv_path}")

    if rows:
        _src_tgt_permutation_test_report(
            rows, save_dir=save_dir, tag=tag_full, x_key=x_key, n_perm=n_perm, seed=perm_seed,
        )
    else:
        print("[Exp1 SRC-TGT] no rows produced; skipping permutation test report")

    return rows


def plot_and_save_example_table(
    outs_by_layer,
    best_layers,
    valid_langs,
    acc_map,
    n_per_relation=10,
    top_k_per_lang=300,
    subset_langs=("zh", "vi", "uk", "ru", "nl"),
    save_dir=None,
    title="Orthogonality Example Table",
    x_key="x_ortho",
):
    """
    Plot a (n_langs x n_cols) table showing per-language-normalised orthogonality
    as a red-white-green gradient with ground truth T/F in each cell.

    Selection: for each relation (topic/dataset), find the top-n_per_relation examples
    with consistently high x_ortho across languages, then concatenate across relations.
    Runs twice — once using all languages, once using subset_langs — saving a separate
    SVG and JSON for each.

    Colormap: 0.65 = white, below → red, above → green (TwoSlopeNorm).
    """
    from collections import defaultdict
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    import matplotlib.colors as mcolors

    # ── 1. Build per-lang data ────────────────────────────────────────────────
    lang_to_idx_ortho = {}   # lang -> {idx: ortho}
    lang_to_idx_acc   = {}   # lang -> {idx: 0/1}
    idx_to_relation   = {}   # idx -> relation string (from any lang)
    used_langs = []

    for lang in sorted(valid_langs):
        layer = best_layers.get(lang)
        if layer is None:
            continue
        if layer not in outs_by_layer or lang not in outs_by_layer[layer]:
            continue
        pack = outs_by_layer[layer][lang]
        idxs   = np.asarray(pack["indices"], dtype=np.int64)
        orthos = np.asarray(pack.get(x_key, pack["x_ortho"]), dtype=np.float64)
        rels   = pack.get("relation", None)
        valid_m = np.isfinite(orthos) & np.isfinite(idxs.astype(float))
        idxs   = idxs[valid_m]
        orthos = orthos[valid_m]
        if len(idxs) == 0:
            continue
        idx_to_ortho = {int(i): float(o) for i, o in zip(idxs, orthos)}
        if rels is not None:
            rels_arr = np.asarray(rels, dtype=object)[valid_m]
            for i, r in zip(idxs, rels_arr):
                if int(i) not in idx_to_relation and r is not None:
                    idx_to_relation[int(i)] = str(r)
        idx_to_acc = {}
        if acc_map is not None:
            for idx in idx_to_ortho:
                key = (int(idx), str(lang))
                if key in acc_map:
                    idx_to_acc[idx] = int(acc_map[key])
        lang_to_idx_ortho[lang] = idx_to_ortho
        lang_to_idx_acc[lang]   = idx_to_acc
        used_langs.append(lang)

    if not used_langs:
        print("[example_table] No data; skipping.")
        return

    # ── 2. Per-language normalisation stats (all examples) ───────────────────
    eps = 1e-12
    lang_mean_ortho = {}
    lang_max_dev    = {}
    for lang in used_langs:
        all_vals = np.array(list(lang_to_idx_ortho[lang].values()), dtype=np.float64)
        all_vals = all_vals[np.isfinite(all_vals)]
        m = float(np.mean(all_vals)) if len(all_vals) else 0.5
        lang_mean_ortho[lang] = m
        lang_max_dev[lang]    = float(np.max(np.abs(all_vals - m))) if len(all_vals) else 1.0

    # ── 3. Group indices by relation ─────────────────────────────────────────
    relation_to_idxs = defaultdict(set)
    for idx in (k for l in used_langs for k in lang_to_idx_ortho[l]):
        relation_to_idxs[idx_to_relation.get(idx, "unknown")].add(idx)

    # ── 4. Selection helper ───────────────────────────────────────────────────
    def select_top_per_relation(langs_to_use):
        """
        Returns list of (relation, idx) ordered by relation then mean ortho desc.
        For each relation picks up to n_per_relation examples with the highest
        x_ortho consistently across langs_to_use.
        """
        active = [l for l in langs_to_use if l in lang_to_idx_ortho]
        if not active:
            return []
        result = []
        for rel in sorted(relation_to_idxs):
            rel_pool = relation_to_idxs[rel]
            k = top_k_per_lang
            common = set()
            while k <= 10_000:
                top_sets = []
                for lang in active:
                    rel_orthos = {i: lang_to_idx_ortho[lang][i]
                                  for i in rel_pool if i in lang_to_idx_ortho[lang]}
                    if not rel_orthos:
                        top_sets.append(set())
                        continue
                    ranked = sorted(rel_orthos, key=rel_orthos.__getitem__, reverse=True)
                    top_sets.append(set(ranked[:k]))
                common = set.intersection(*top_sets) if top_sets else set()
                if len(common) >= n_per_relation:
                    break
                new_k = int(k * 1.5)
                if new_k == k:
                    break
                k = new_k
            if not common:
                continue
            mean_o = {
                idx: float(np.mean([lang_to_idx_ortho[l][idx]
                                    for l in active if idx in lang_to_idx_ortho[l]]))
                for idx in common
            }
            top = sorted(mean_o, key=mean_o.__getitem__, reverse=True)[:n_per_relation]
            for idx in top:
                result.append((rel, idx))
        return result

    # ── 5. Custom colormap: red → white at 0.65 → green ──────────────────────
    cmap = LinearSegmentedColormap.from_list("rwg", ["#d73027", "white", "#1a9850"])

    # ── 6. Build matrices and plot for one selection ─────────────────────────
    def _plot_and_save(selection_name, langs_for_sel, selected):
        if not selected:
            print(f"[example_table] No examples for '{selection_name}'; skipping.")
            return
        relations_list = [r for r, _ in selected]
        sel_idxs       = [idx for _, idx in selected]
        n_ex   = len(sel_idxs)
        n_langs = len(used_langs)

        ortho_matrix = np.full((n_langs, n_ex), np.nan)
        norm_matrix  = np.full((n_langs, n_ex), np.nan)
        for i, lang in enumerate(used_langs):
            mean_L = lang_mean_ortho[lang]
            dev_L  = lang_max_dev[lang]
            for j, idx in enumerate(sel_idxs):
                v = lang_to_idx_ortho[lang].get(idx, np.nan)
                ortho_matrix[i, j] = v
                if np.isfinite(v):
                    norm_matrix[i, j] = float(
                        np.clip(0.5 + (v - mean_L) / (2.0 * dev_L + eps), 0.0, 1.0)
                    )

        cell_w, cell_h = 1.1, 0.65
        fig_w = max(14, cell_w * n_ex + 3.5)
        fig_h = max(4, cell_h * n_langs + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # White at normalised 0.65
        col_norm = TwoSlopeNorm(vmin=0.0, vcenter=0.65, vmax=1.0)

        for i, lang in enumerate(used_langs):
            row = n_langs - 1 - i
            for j, idx in enumerate(sel_idxs):
                nv = norm_matrix[i, j]
                acc_val = lang_to_idx_acc.get(lang, {}).get(idx, None)
                color = cmap(col_norm(nv)) if np.isfinite(nv) else (0.85, 0.85, 0.85, 1.0)
                ax.add_patch(plt.Rectangle([j, row], 1, 1, color=color,
                                           linewidth=0.5, edgecolor="white"))
                if acc_val is not None:
                    ax.text(j + 0.5, row + 0.5, "T" if acc_val == 1 else "F",
                            ha="center", va="center", fontsize=9,
                            color="black", fontweight="bold")
                else:
                    ax.text(j + 0.5, row + 0.5, "?",
                            ha="center", va="center", fontsize=9, color="gray")

        # Vertical separators + relation labels between groups
        prev_rel = None
        rel_group_start = 0
        for j, rel in enumerate(relations_list):
            if rel != prev_rel:
                if prev_rel is not None:
                    ax.axvline(x=j, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
                    mid = (rel_group_start + j) / 2
                    ax.text(mid, n_langs + 0.08, prev_rel, ha="center", va="bottom",
                            fontsize=7, rotation=25, clip_on=False)
                rel_group_start = j
                prev_rel = rel
        # Last group label
        if prev_rel is not None:
            mid = (rel_group_start + n_ex) / 2
            ax.text(mid, n_langs + 0.08, prev_rel, ha="center", va="bottom",
                    fontsize=7, rotation=25, clip_on=False)

        ax.set_xlim(0, n_ex)
        ax.set_ylim(0, n_langs)
        ax.set_xticks(np.arange(n_ex) + 0.5)
        ax.set_xticklabels([f"#{idx}" for idx in sel_idxs], rotation=60, ha="right", fontsize=6)
        ax.set_yticks(np.arange(n_langs) + 0.5)
        ax.set_yticklabels(used_langs[::-1], fontsize=9)
        ax.set_title(f"{title} [{selection_name}]", fontsize=11, pad=18)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=col_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax,
                     label="Norm ortho (0.65=white/threshold; green→T, red→F)",
                     shrink=0.8)
        fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            svg_path  = os.path.join(save_dir, f"example_table_{selection_name}.svg")
            json_path = os.path.join(save_dir, f"example_table_data_{selection_name}.json")
            fig.savefig(svg_path, bbox_inches="tight", format="svg")
            print(f"[example_table] Saved: {svg_path}")
            data = {
                "selection_name": selection_name,
                "langs_used_for_selection": list(langs_for_sel),
                "selected_indices": [int(i) for i in sel_idxs],
                "relations": relations_list,
                "languages": used_langs,
                "lang_mean_ortho": {l: lang_mean_ortho[l] for l in used_langs},
                "lang_max_dev":    {l: lang_max_dev[l]    for l in used_langs},
                "best_layers": {l: int(best_layers[l]) for l in used_langs if l in best_layers},
                "ortho_matrix": {
                    lang: {str(idx): (ortho_matrix[i, j] if np.isfinite(ortho_matrix[i, j]) else None)
                           for j, idx in enumerate(sel_idxs)}
                    for i, lang in enumerate(used_langs)
                },
                "norm_matrix": {
                    lang: {str(idx): (norm_matrix[i, j] if np.isfinite(norm_matrix[i, j]) else None)
                           for j, idx in enumerate(sel_idxs)}
                    for i, lang in enumerate(used_langs)
                },
                "acc_data": {
                    lang: {str(idx): lang_to_idx_acc.get(lang, {}).get(idx, None)
                           for idx in sel_idxs}
                    for lang in used_langs
                },
            }
            with open(json_path, "w") as _f:
                json.dump(data, _f, indent=2)
            print(f"[example_table] Saved data: {json_path}")

        plt.show()
        plt.close(fig)

    # ── 7. Run both selections ────────────────────────────────────────────────
    _plot_and_save("all_langs", used_langs, select_top_per_relation(used_langs))
    subset_valid = [l for l in subset_langs if l in lang_to_idx_ortho]
    if subset_valid:
        _plot_and_save("subset_langs", subset_valid, select_top_per_relation(subset_valid))


rel_map_targets = load_relation_by_index_lang_from_jsonl(TARGETS_JSONL)
print("relations found in TARGETS_JSONL:", len(rel_map_targets))

rel_map_preds = load_relation_by_index_lang_from_jsonl(path)   # your predictions JSONL path
print("relations found in predictions JSONL:", len(rel_map_preds))

# ── Experiment on/off switches ───────────────────────────────────────────────
DO_EXP0  = False   # train-selected + fixed-layer cumulative AUC (all modes)
DO_EXP1  = True   # dev/test Oscar split, en-fact orthogonality
DO_EXP1_SWEEP = False  # exp1 second part: sweep W_L var_prop + layer, pick best by dev AUC
EXP1_SWEEP_VAR_PROPS = [0.85, 0.90, 0.95, 0.99]
EXP1_SRC_TGT_sub_control_exp = True  # imported-subspace negative control: every source
                                     # language's subspace vs every target language's facts
                                     # (see compute_exp1_src_tgt_subspace_control below)
# Single metric used to rank facts for the SRC x TGT control matrix — the
# same cumulative-coherence metric used throughout Exp1.
EXP1_SRC_TGT_X_KEY = "vec_subspace_angle_by_cumulative_coherence"
EXP1_SRC_TGT_METRIC_TAG = "coherence"
# Which layer_select mode to run for the SRC x TGT control — "src_dev"
# (original: source's own best layer, confounds subspace with layer choice
# off-diagonal) or "tgt_dev" (pure subspace-only ablation: target's own best
# layer held fixed for every imported source). See
# compute_exp1_src_tgt_subspace_control's docstring for the trade-off.
EXP1_SRC_TGT_LAYER_SELECT_MODE = "tgt_dev"  # or "src_dev"
# If True, "en" is always included as an extra SOURCE in the SRC x TGT
# control (its subspace gets imported into every target's column, competing
# for rank there) but is NEVER itself a target — no en column, no self cell,
# no en row in Exp1's other valid_langs-driven outputs. Requires a one-off
# extra dev-layer-selection pass scoped to "en" alone (see call site) since
# "en" isn't in valid_langs.
EXP1_SRC_TGT_ALWAYS_INCLUDE_EN_AS_SOURCE = True
DO_EXP2  = True   # per-relation en-fact subspace vs Oscar W_L
DO_EXP2_SWEEP = False  # exp2 second part: sweep W_L var × W_rel var × layer, pick best by dev r
DO_EXP3  = False   # en-fact vector vs mu_L (mean of language subspace)
DO_EXP3B = False   # en-fact vector vs W_L subspace (ortho_min per example)
DO_EXP4  = False   # merged all-language cumulative experiment
DO_EXP5  = False   # mean_translation_vector angle experiment
EXP0_PLOT_EXAMPLE_TABLE = False
EXP1_PLOT_EXAMPLE_TABLE   =True
EXP1_TABLE_X_KEY = "vec_subspace_angle_by_cumulative_coherence"  # metric used to sort/colour example table


# ── Experiment constants ────────────────────────────────────────────────────
dev_frac         = 0.1   # fraction of en-facts used as dev (train) split
LAYERS_TO_LOOP   = list(range(1, model.cfg.n_layers))
EXCLUDE_LANGS    = ()
MIN_EXAMPLES     = 5
DATASET_VAR_PROP =0.99 #0.99
SUBSPACE_CENTERED = False
SV_WEIGHT_MODE = "none"  # "sv": normalised by sum(sv); "sv_squared": normalised by sum(sv²); "none": direct sv-weighted sum
ALL_METRICS = ["subspace_subspace_angle_by_cumulative_coherence"]
# Subset of ALL_METRICS to actually run in exp2 — just the one CI metric.
ACTIVE_METRICS = ["subspace_subspace_angle_by_cumulative_coherence"]
_COHERENCE_METRICS = {
    "subspace_subspace_angle_by_cumulative_coherence",
}
# No weighted-coherence variant remains, so the sv_weight_mode sweep over
# ["sv", "sv_squared", "none"] below is a no-op (kept only because the single
# active metric doesn't need special-casing out of that loop).
_EXP2_WEIGHTED_METRICS = set()
METRIC_LABELS = {
    "subspace_subspace_angle_by_cumulative_coherence":
        "1 - |pairwise-coherence(W_rel, W_L)|",
}

en_fact_best_layer = None
pr_auc_tables_exp0 = {}

y_mode          = "accuracy"
cumulative_mode = "cumulative"
quantile_step   = 0.1
sign_of_feature = "lowest_feature_first"
x_bar           = "percentile"  # options: "percentile", "fixedrange", "pct_scaled"

_acc_map = {}
try:
    with open(path, "r") as _f:
        for _line in _f:
            _ex = json.loads(_line)
            _acc_map[(int(_ex["index"]), str(_ex["lang"]))] = int(_ex["match"])
except Exception as _e:
    print(f"[warn] could not load acc_map from {path}: {_e}")
    _acc_map = None

for base_or_random_mode in (base_or_random_mode_list if DO_EXP0 else []):
    print(lang_mode)
    print(base_or_random_mode)
    # --------------- run across layers ---------------
    outs_by_layer = {}

    for layer in range(1, model.cfg.n_layers):
        outs_by_layer[layer] = compute_geometric_feature(
            model=model,
            run_dir=RUN_DIR,
            shot_tag=shot_tag,
            valid_langs=valid_langs,
            layer=layer,
            rep_kind=rep_kind,
            lang_mode=lang_mode,
            targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind,
            y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100,
            separate_correct_incorrect_examples=False,
            feature=feature,
            base_or_random_mode=base_or_random_mode,
            random_seed=0,
            mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets,
            rel_map_preds=rel_map_preds,
            store_X_base=False,
        )
        print(f"[done] layer={layer:02d} langs={len(outs_by_layer[layer])}")

    # ------------------------------------------------------------
    # AUC bookkeeping
    # ------------------------------------------------------------
    layers_run = sorted(outs_by_layer.keys())
    if len(layers_run) == 0:
        print(f"[warn] no layers computed for mode={base_or_random_mode}; skipping AUC table fill")
        continue


    tt = tt_select_best_layer_with_split(
        outs_by_layer,
        valid_langs=valid_langs,
        layers=layers_run,
        seed=0,
        y_layer="last",
        min_n=100,
        train_frac=dev_frac,
        x_bar=x_bar,
        acc_map=_acc_map,
    )

    y_acc_by_lang = None
    if y_mode == "accuracy":
        try:
            y_acc_by_lang = build_y_acc_by_lang_from_predictions_jsonl(path, tt)
        except Exception as e:
            print(f"[warn] could not build y_acc_by_lang from predictions JSONL; using pack y_accuracy if available: {e}")
            y_acc_by_lang = None

    plot_root = os.path.join(EXP_ROOT_DIR, f"{feature}_{sign_of_feature}", "auc_plots")
    os.makedirs(plot_root, exist_ok=True)

    exp_key_trainselected = (
        prompt_num, shot_tag, rep_kind, lang_mode_name, target_kind, y_transform,
        feature, sign_of_feature, y_mode, "trainselected_layer",
    )
    trainselected_auc_out, tt = plot_cumulative_all_layers_all_langs(
        outs_by_layer, tt,
        valid_langs=valid_langs,
        y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang,
        y_layer="last",
        quantile_step=quantile_step,
        cumulative_mode=cumulative_mode,
        title=f"{base_or_random_mode}: train-selected layer cumulative AUC",
        save_path=os.path.join(plot_root, f"{base_or_random_mode}__trainselected_layer__{y_mode}_{cumulative_mode}.svg"),
        save_data_path=os.path.join(plot_root, f"{base_or_random_mode}__trainselected_layer__{y_mode}_{cumulative_mode}.json"),
        calculate_auc=True,
        show=False,
        fixed_layers_to_plot=None,
        y_lim=(0.0, 1.0),
        pb_corr_subset="all",
        pb_corr_save_path=os.path.join(plot_root, f"{base_or_random_mode}__trainselected_layer__pb_corr.json"),
    )
    auc_tables[exp_key_trainselected][base_or_random_mode] = _auc_out_to_pp(trainselected_auc_out)

    exp_key_fixed = (
        prompt_num, shot_tag, rep_kind, lang_mode_name, target_kind, y_transform,
        feature, sign_of_feature, y_mode, "fixed_layer",
    )
    if base_or_random_mode == "en_fact":
        en_fact_best_layer = {
            lang: int(info["best_layer"])
            for lang, info in tt.items()
            if info.get("best_layer", None) is not None
        }
    fixed_auc_out, tt = plot_cumulative_all_layers_all_langs(
        outs_by_layer, tt,
        valid_langs=valid_langs,
        y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang,
        y_layer="last",
        quantile_step=quantile_step,
        cumulative_mode=cumulative_mode,
        title=f"{base_or_random_mode}: fixed PP-selected layer cumulative AUC",
        save_path=os.path.join(plot_root, f"{base_or_random_mode}__fixed_layer__{y_mode}_{cumulative_mode}.svg"),
        save_data_path=os.path.join(plot_root, f"{base_or_random_mode}__fixed_layer__{y_mode}_{cumulative_mode}.json"),
        calculate_auc=True,
        show=False,
        fixed_layers_to_plot=en_fact_best_layer,
        y_lim=(0.0, 1.0),
        pb_corr_subset="all",
        pb_corr_save_path=os.path.join(plot_root, f"{base_or_random_mode}__fixed_layer__pb_corr.json"),
    )
    auc_tables[exp_key_fixed][base_or_random_mode] = _auc_out_to_pp(fixed_auc_out)

    # ── PR-AUC (failure prediction) at train-selected best layer ─────────────
    if y_acc_by_lang is not None:
        _exp0_best_layers = {
            lang: int(info["best_layer"])
            for lang, info in tt.items()
            if info.get("best_layer") is not None
        }
        _exp0_pr_dir = os.path.join(plot_root, f"pr_auc_{base_or_random_mode}_trainselected")
        _exp0_pr_summary = _compute_and_save_pr_auc_all_langs(
            outs_by_layer, tt, valid_langs,
            best_layers=_exp0_best_layers,
            y_acc_by_lang=y_acc_by_lang,
            save_dir=_exp0_pr_dir,
            label=f"Exp0 {base_or_random_mode} train-sel",
        )
        pr_auc_tables_exp0[base_or_random_mode] = _exp0_pr_summary

    if EXP0_PLOT_EXAMPLE_TABLE and base_or_random_mode == "en_fact":
        _exp0_table_best_layers = {
            lang: int(info["best_layer"])
            for lang, info in tt.items()
            if info.get("best_layer") is not None
        }
        plot_and_save_example_table(
            outs_by_layer=outs_by_layer,
            best_layers=_exp0_table_best_layers,
            valid_langs=valid_langs,
            acc_map=_acc_map,
            n_per_relation=10,
            top_k_per_lang=300,
            save_dir=os.path.join(plot_root, "..", "example_table_exp0"),
            title="[Exp0 en_fact] Orthogonality predictor — top-10 per relation",
        )

    del outs_by_layer


for exp_key, mode_to_auc in auc_tables.items():
    (
        prompt_num,
        shot_tag,
        rep_kind,
        lang_mode_name,
        target_kind,
        y_transform,
        feature,
        sign_of_feature,
        y_mode,
        cumulative_mode,
    ) = exp_key

    # build dataframe: rows=langs, cols=base_or_random_mode
    df = pd.DataFrame(index=sorted(valid_langs))

    for mode_name, lang_to_auc in mode_to_auc.items():
        df[mode_name] = pd.Series(lang_to_auc)

    out_dir = os.path.join(EXP_ROOT_DIR, f"{feature}_{sign_of_feature}", "auc_tables")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"pp_compare_y_{y_mode}_{cumulative_mode}_fixedlayer_bestlangspecific.csv",
    )
    df.to_csv(csv_path, index_label="language")
    print("Saved AUC table:", csv_path)

    try:
        raw_df, en_fact_series = load_auc_matrix(csv_path, valid_langs=valid_langs)
        base = csv_path.replace(".csv", "")
        plot_auc_heatmap(
            raw_df, en_fact_series,
            add_en_fact_row=True,
            save_path=base + "_heatmap.svg",
            title=os.path.basename(csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            raw_df, en_fact_series,
            add_en_fact_row=True,
            save_path=base + "_barplot.svg",
            title=os.path.basename(csv_path),
        )
    except Exception as e:
        print(f"[warn] could not plot AUC for {csv_path}: {e}")

    # ── ranked bar plot: each mode sorted by mean PP ──────────────────────────
    _ranked_entries = []
    for _mode, _lang_to_pp in mode_to_auc.items():
        _vals = [v for v in _lang_to_pp.values() if np.isfinite(v)]
        _mean = float(np.nanmean(_vals)) if _vals else float("nan")
        _ranked_entries.append((_mode, _mean))
    _ranked_entries.sort(key=lambda t: t[1] if np.isfinite(t[1]) else -np.inf, reverse=True)
    _rlabels = [t[0] for t in _ranked_entries]
    _rvals   = [t[1] for t in _ranked_entries]

    fig, ax = plt.subplots(figsize=(8, 6))
    _x = np.arange(len(_ranked_entries))
    _bars = ax.bar(_x, _rvals, color="#5A9BD5")
    ax.set_xticks(_x)
    ax.set_xticklabels(_rlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean PP across target languages")
    ax.set_title(f"Modes ranked by mean PP — {y_mode} {cumulative_mode}")
    ax.grid(axis="y", alpha=0.3)
    _fv = [v for v in _rvals if np.isfinite(v)]
    if _fv:
        ax.set_ylim(min(0, min(_fv)) * 1.05, max(_fv) * 1.1)
    for _b, _v in zip(_bars, _rvals):
        if np.isfinite(_v):
            ax.text(_b.get_x() + _b.get_width() / 2, _v, f"{_v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _ranked_path = os.path.join(out_dir, f"pp_ranked_modes_{y_mode}_{cumulative_mode}.svg")
    fig.savefig(_ranked_path, bbox_inches="tight", format="svg")
    print("Saved ranked bar plot:", _ranked_path)
    plt.show()
    plt.close(fig)

# ── Exp 0 PR-AUC table ───────────────────────────────────────────────────────
if pr_auc_tables_exp0:
    _pr0_dir = os.path.join(EXP_ROOT_DIR, f"{feature}_{sign_of_feature}", "pr_auc_tables")
    os.makedirs(_pr0_dir, exist_ok=True)

    # Full JSON — keeps precision/recall arrays for replotting
    _pr0_json_path = os.path.join(_pr0_dir, "pr_auc_compare_trainselected.json")
    with open(_pr0_json_path, "w") as _f:
        json.dump(pr_auc_tables_exp0, _f, indent=2)
    print("Saved Exp0 PR-AUC table:", _pr0_json_path)

    # CSV — rows=langs, cols=modes, for heatmap
    _pr0_df = pd.DataFrame(index=sorted(valid_langs))
    for _mode, _lang_stats in pr_auc_tables_exp0.items():
        _pr0_df[_mode] = pd.Series({lang: v["pr_auc"] for lang, v in _lang_stats.items()})
    _pr0_csv_path = os.path.join(_pr0_dir, "pr_auc_compare_trainselected.csv")
    _pr0_df.to_csv(_pr0_csv_path, index_label="language")
    print("Saved Exp0 PR-AUC CSV:", _pr0_csv_path)

    try:
        _pr0_raw_df, _pr0_en_fact_series = load_auc_matrix(_pr0_csv_path, valid_langs=valid_langs)
        _pr0_base = _pr0_csv_path.replace(".csv", "")
        plot_auc_heatmap(
            _pr0_raw_df, _pr0_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr0_base + "_heatmap.svg",
            title=os.path.basename(_pr0_csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            _pr0_raw_df, _pr0_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr0_base + "_barplot.svg",
            title=os.path.basename(_pr0_csv_path),
        )
    except Exception as _pr0_err:
        print(f"[warn] could not plot Exp0 PR-AUC heatmap: {_pr0_err}")

# ═══════════════════════════════════════════════════════════════════════════
# PAPER-STYLE REPLOTS — ported from replot_multilingual_facts.ipynb
#   (same fonts/colors/layout as the paper figures; scoped via plt.rc_context
#   so it doesn't change the styling of any other plot in this script)
# ═══════════════════════════════════════════════════════════════════════════
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PAPER_RCPARAMS = {
    "figure.figsize": (8, 6),
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 13,
    "lines.linewidth": 2,
}
PAPER_GLOBAL_LINEWIDTH = 3.2
PAPER_COLORS = [
    "#2B5C9A",  # deep blue
    "#2CA6C8",  # cyan
    "#1AA6A6",  # teal
    "#7FD3DB",  # stronger light cyan
    "#6C4DB8",  # purple
    "#7A5CCF",  # violet
    "#A98EF0",  # stronger lavender
    "#C83E3E",  # red
    "#F26C4F",  # coral
    "#A05434",  # stronger amber/orange
]
PAPER_MARKERS = ["h", "p", ">", "D", "*", "P", "v", "o", "s", "^"]

# Same 10 swatches as before, reordered into a blue (low) -> brown (high)
# sequence. Colors are assigned by RANK of mean accuracy, not by fixed
# language identity: the lowest-mean-accuracy language always gets the
# bluest color, the highest-mean-accuracy language always gets the
# brownest, with the rest interpolated in between. Compute the mapping
# once per run (see _pp_lang_styles_by_mean_accuracy) and thread it through
# the cumulative, non-cumulative, and PR-AUC plots so a given language is
# the same color (and marker) everywhere for that run.
PAPER_COLORS_BLUE_TO_BROWN = [
    "#2D5D97",  # dark blue (lowest accuracy)
    "#57B1C1",  # cyan
    "#89D0DB",  # light cyan
    "#8CCAC7",  # teal
    "#6D7BDA",  # blue-purple
    "#6D63D6",  # purple-blue
    "#A564D6",  # purple
    "#C85A4C",  # brick red
    "#CF7856",  # salmon
    "#B46A3A",  # brown (highest accuracy)
]
LANG_DEFAULT_COLOR = "#7F7F7F"


def _pp_get_paper_styles(n):
    colors = [PAPER_COLORS[i % len(PAPER_COLORS)] for i in range(n)]
    markers = [PAPER_MARKERS[i % len(PAPER_MARKERS)] for i in range(n)]
    return colors, markers


def _pp_colors_by_rank(n):
    """n colors sampled evenly along the blue(low)->brown(high) gradient."""
    if n <= 0:
        return []
    if n == 1:
        return [PAPER_COLORS_BLUE_TO_BROWN[len(PAPER_COLORS_BLUE_TO_BROWN) // 2]]
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_to_brown", PAPER_COLORS_BLUE_TO_BROWN)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _pp_lang_styles_by_mean_accuracy(mean_acc_by_lang: dict):
    """
    Rank languages ascending by mean accuracy (lowest=blue ... highest=brown).

    Returns (lang_order, lang_colors, lang_markers) — compute once per run and
    reuse across the cumulative, non-cumulative, and PR-AUC plots so color,
    marker, and left-to-right/legend order all stay identical for that run.
    """
    lang_order = sorted(mean_acc_by_lang.keys(), key=lambda l: mean_acc_by_lang[l])
    colors = _pp_colors_by_rank(len(lang_order))
    markers = [PAPER_MARKERS[i % len(PAPER_MARKERS)] for i in range(len(lang_order))]
    lang_colors = dict(zip(lang_order, colors))
    lang_markers = dict(zip(lang_order, markers))
    return lang_order, lang_colors, lang_markers


def _pp_as_float_array(x):
    return np.asarray(x, dtype=float)


def _pp_clean_label(label):
    return str(label).split("|")[0].split("(")[0].strip()


def _pp_curve_auc(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return np.nan
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.trapz(ys, xs))


def _pp_baseline_from_curve(ys):
    ys = np.asarray(ys, dtype=float)
    ys = ys[np.isfinite(ys)]
    if len(ys) == 0:
        return np.nan
    return float(np.mean(ys))


def _pp_predictive_power(auc, baseline):
    """Approximate PP. Lower AUC is better; baseline AUC ~ flat mean-accuracy curve."""
    if not np.isfinite(auc) or not np.isfinite(baseline):
        return np.nan
    auc_baseline = baseline
    auc_perfect = 0.0
    denom = auc_baseline - auc_perfect
    if denom <= 1e-8:
        return np.nan
    return float((auc_baseline - auc) / denom)


def _pp_save_metrics_table_svg(
    sorted_items, colors, markers, table_save_path, *, figsize=(3.2, 3.2),
    value_cols=(("AUC", "auc"), ("PP", "pp")),
):
    """
    value_cols: (column_label, curve_dict_key) pairs shown after Lang, e.g.
    the cumulative table shows AUC/PP; the non-cumulative table (see
    replot_exp1_noncumulative_from_json) shows R/p instead — same styling.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    style_map = {
        lang: (color, marker)
        for (lang, _), color, marker in zip(sorted_items, colors, markers)
    }

    baseline_sorted = sorted(sorted_items, key=lambda kv: kv[1]["baseline"], reverse=True)

    table_data = []
    for lang, v in baseline_sorted:
        clean = _pp_clean_label(v["label"])
        table_data.append(["", clean] + [f"{v[key]:.2f}" for _, key in value_cols])

    table = ax.table(
        cellText=table_data,
        colLabels=["", "Lang"] + [label for label, _ in value_cols],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.3)

    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("white")

    for j in range(2 + len(value_cols)):
        table[(0, j)].set_text_props(weight="bold")

    fig.canvas.draw()

    for i, (lang, _) in enumerate(baseline_sorted, start=1):
        color, marker = style_map[lang]
        cell = table[(i, 0)]
        bbox = cell.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transAxes.inverted()
        (x0, y0) = inv.transform((bbox.x0, bbox.y0))
        (x1, y1) = inv.transform((bbox.x1, bbox.y1))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.scatter([cx], [cy], marker=marker, s=120, color=color,
                   transform=ax.transAxes, clip_on=False, zorder=5)

    for i, (lang, _) in enumerate(baseline_sorted, start=1):
        color, _ = style_map[lang]
        cell = table[(i, 1)]
        cell.set_facecolor(color)
        cell.set_alpha(0.22)

    table_save_path = Path(table_save_path)
    table_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(table_save_path, bbox_inches="tight", format="svg")
    print(f"Saved table: {table_save_path}")
    plt.close(fig)


def replot_exp1_cumulative_from_json(
    json_path,
    save_path=None,
    *,
    table_save_path=None,
    remove_langs=("en", "english"),
    title="Cumulative Accuracy by Cosine Distance",
    figsize=(8, 6),
    dot_size=120,
    show=False,
    lang_order: list | None = None,
    lang_colors: dict | None = None,
    lang_markers: dict | None = None,
):
    """
    lang_order/lang_colors/lang_markers: pass the shared mapping built by
    _pp_lang_styles_by_mean_accuracy (once per run, from the main cumulative
    pass) so color/marker/order stay identical across the cumulative,
    non-cumulative, and PR-AUC plots for that run. If omitted, falls back to
    ranking languages by this json's own mean accuracy (curve baseline).
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    curves = data.get("curves", {})
    if not curves:
        print(f"[skip] no curves in {json_path}")
        return

    usable = {}
    for lang, curve in curves.items():
        if lang.lower() in remove_langs:
            continue
        xs = _pp_as_float_array(curve["xs"])
        ys = _pp_as_float_array(curve["ys"])
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue
        auc = _pp_curve_auc(xs, ys)
        baseline = _pp_baseline_from_curve(ys)
        pp = _pp_predictive_power(auc, baseline)
        usable[lang] = {"xs": xs, "ys": ys, "label": curve.get("label", lang),
                         "auc": auc, "baseline": baseline, "pp": pp}

    if not usable:
        print(f"[skip] no usable non-English curves in {json_path}")
        return

    if lang_order is not None:
        # Shared order from the canonical run-level ranking; any usable
        # language missing from it (shouldn't normally happen) is appended,
        # ranked by this json's own mean accuracy.
        ordered = [l for l in lang_order if l in usable]
        leftover = sorted((l for l in usable if l not in ordered), key=lambda l: usable[l]["baseline"])
        sorted_items = [(l, usable[l]) for l in ordered + leftover]
    else:
        sorted_items = sorted(usable.items(), key=lambda kv: kv[1]["baseline"])

    if lang_colors is not None and lang_markers is not None:
        colors = [lang_colors.get(lang, LANG_DEFAULT_COLOR) for lang, _ in sorted_items]
        markers = [lang_markers.get(lang, PAPER_MARKERS[i % len(PAPER_MARKERS)])
                   for i, (lang, _) in enumerate(sorted_items)]
    else:
        _, _lc, _lm = _pp_lang_styles_by_mean_accuracy({l: v["baseline"] for l, v in sorted_items})
        colors = [_lc[lang] for lang, _ in sorted_items]
        markers = [_lm[lang] for lang, _ in sorted_items]

    # y range spans the min/max actually observed across all languages (with a
    # small pad), instead of a fixed range that can crop languages whose
    # accuracy falls outside it.
    all_ys = np.concatenate([v["ys"] for _, v in sorted_items])
    all_ys = all_ys[np.isfinite(all_ys)]
    if all_ys.size > 0:
        y_lo, y_hi = float(np.min(all_ys)), float(np.max(all_ys))
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        y_lo, y_hi = max(0.0, y_lo - pad), min(1.0, y_hi + pad)
    else:
        y_lo, y_hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []
    for (lang, v), color, marker in zip(sorted_items, colors, markers):
        ax.plot(v["xs"], v["ys"], color=color, lw=PAPER_GLOBAL_LINEWIDTH, alpha=0.95)
        ax.scatter(v["xs"], v["ys"], color=color, marker=marker, s=dot_size,
                   alpha=0.9, linewidths=0)
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle="-",
                   markersize=9, label=_pp_clean_label(v["label"]))
        )

    ax.set_xticks([0.1, 0.5, 1.0])
    ax.set_yticks(np.linspace(y_lo, y_hi, 3))
    ax.set_xlabel("Minimum CI cutoff percentile (high to low)")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(title)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(min(0,y_lo), y_hi+0.1)
    ax.grid(alpha=0.2)
    ax.legend(
        handles=legend_handles, ncol=1, fontsize=10,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False,
    )
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.tight_layout()

    save_path = Path("paper_plots") / "cumulative_2d.svg" if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", format="svg")
    print(f"Saved plot: {save_path}")

    if table_save_path is None:
        table_save_path = save_path.with_name(save_path.stem + "_table.svg")
    _pp_save_metrics_table_svg(sorted_items, colors, markers, table_save_path)

    if show:
        plt.show()
    plt.close(fig)


def replot_exp1_noncumulative_from_json(
    json_path,
    save_path=None,
    *,
    table_save_path=None,
    remove_langs=("en", "english"),
    title="Non-cumulative Accuracy by Cosine Distance",
    figsize=(8, 6),
    dot_size=120,
    show=False,
    lang_order: list | None = None,
    lang_colors: dict | None = None,
    lang_markers: dict | None = None,
):
    """
    Paper-style replot of a non-cumulative curve JSON (per-bin mean accuracy,
    curves carry "r"/"p" instead of "auc"/"pp"). Same visual style as
    replot_exp1_cumulative_from_json, and takes the same lang_order/
    lang_colors/lang_markers so color/marker/order stay identical to the
    cumulative and PR-AUC plots for this run.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    curves = data.get("curves", {})
    if not curves:
        print(f"[skip] no curves in {json_path}")
        return

    usable = {}
    for lang, curve in curves.items():
        if lang.lower() in remove_langs:
            continue
        xs = _pp_as_float_array(curve["xs"])
        ys = _pp_as_float_array(curve["ys"])
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue
        baseline = _pp_baseline_from_curve(ys)
        usable[lang] = {
            "xs": xs, "ys": ys, "label": curve.get("label", lang), "baseline": baseline,
            "r": float(curve.get("r", float("nan"))), "p": float(curve.get("p", float("nan"))),
        }

    if not usable:
        print(f"[skip] no usable non-English curves in {json_path}")
        return

    if lang_order is not None:
        ordered = [l for l in lang_order if l in usable]
        leftover = sorted((l for l in usable if l not in ordered), key=lambda l: usable[l]["baseline"])
        sorted_items = [(l, usable[l]) for l in ordered + leftover]
    else:
        sorted_items = sorted(usable.items(), key=lambda kv: kv[1]["baseline"])

    if lang_colors is not None and lang_markers is not None:
        colors = [lang_colors.get(lang, LANG_DEFAULT_COLOR) for lang, _ in sorted_items]
        markers = [lang_markers.get(lang, PAPER_MARKERS[i % len(PAPER_MARKERS)])
                   for i, (lang, _) in enumerate(sorted_items)]
    else:
        _, _lc, _lm = _pp_lang_styles_by_mean_accuracy({l: v["baseline"] for l, v in sorted_items})
        colors = [_lc[lang] for lang, _ in sorted_items]
        markers = [_lm[lang] for lang, _ in sorted_items]

    all_ys = np.concatenate([v["ys"] for _, v in sorted_items])
    all_ys = all_ys[np.isfinite(all_ys)]
    if all_ys.size > 0:
        y_lo, y_hi = float(np.min(all_ys)), float(np.max(all_ys))
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        y_lo, y_hi = max(0.0, y_lo - pad), min(1.0, y_hi + pad)
    else:
        y_lo, y_hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []
    for (lang, v), color, marker in zip(sorted_items, colors, markers):
        ax.plot(v["xs"], v["ys"], color=color, lw=PAPER_GLOBAL_LINEWIDTH, alpha=0.95)
        ax.scatter(v["xs"], v["ys"], color=color, marker=marker, s=dot_size,
                   alpha=0.9, linewidths=0)
        r_str = f"{v['r']:.3f}" if np.isfinite(v["r"]) else "nan"
        p_str = f"{v['p']:.3g}" if np.isfinite(v["p"]) else "nan"
        legend_label = f"{_pp_clean_label(v['label'])} (R={r_str}, p={p_str})"
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle="-",
                   markersize=9, label=legend_label)
        )

    ax.set_xticks([0.1, 0.5, 1.0])
    ax.set_yticks(np.linspace(y_lo, y_hi, 3))
    ax.set_xlabel("Bin (lowest feature first)")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(title)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(min(0, y_lo), y_hi + 0.1)
    ax.grid(alpha=0.2)
    ax.legend(
        handles=legend_handles, ncol=1, fontsize=10,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False,
    )
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.tight_layout()

    save_path = Path("paper_plots") / "noncumulative_2d.svg" if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", format="svg")
    print(f"Saved plot: {save_path}")

    if table_save_path is None:
        table_save_path = save_path.with_name(save_path.stem + "_table.svg")
    _pp_save_metrics_table_svg(sorted_items, colors, markers, table_save_path,
                                value_cols=(("R", "r"), ("p", "p")))

    if show:
        plt.show()
    plt.close(fig)


def replot_exp1_error_recall_from_json(
    json_path,
    save_path=None,
    *,
    table_save_path=None,
    remove_langs=("en", "english"),
    title="Error Recall Curve",
    figsize=(8, 6),
    dot_size=3,
    linewidth=0.05,
    show=False,
    lang_order: list | None = None,
    lang_colors: dict | None = None,
    lang_markers: dict | None = None,
):
    """
    Paper-style replot of an error-recall curve JSON (curves carry
    "error_rate"/"error_recall_auc_above_diagonal" instead of "auc"/"pp").
    Same visual style + shared lang_order/lang_colors/lang_markers as the
    cumulative/non-cumulative/PR-AUC plots for this run. Languages with zero
    errors (no "xs"/"ys" saved) are skipped, same as the raw plot.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    curves = data.get("curves", {})
    if not curves:
        print(f"[skip] no curves in {json_path}")
        return

    usable = {}
    for lang, curve in curves.items():
        if lang.lower() in remove_langs:
            continue
        if "xs" not in curve or "ys" not in curve:
            continue
        xs = _pp_as_float_array(curve["xs"])
        ys = _pp_as_float_array(curve["ys"])
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue
        error_rate = float(curve.get("error_rate", float("nan")))
        auc_above = curve.get("error_recall_auc_above_diagonal", None)
        auc_above = float(auc_above) if auc_above is not None else float("nan")
        usable[lang] = {
            "xs": xs, "ys": ys, "label": curve.get("label", lang),
            "baseline": (1.0 - error_rate) if np.isfinite(error_rate) else float("nan"),
            "error_rate": error_rate, "auc_above": auc_above,
        }

    if not usable:
        print(f"[skip] no usable non-English curves in {json_path}")
        return

    def _rank_key(l):
        b = usable[l]["baseline"]
        return b if np.isfinite(b) else -np.inf

    if lang_order is not None:
        ordered = [l for l in lang_order if l in usable]
        leftover = sorted((l for l in usable if l not in ordered), key=_rank_key)
        sorted_items = [(l, usable[l]) for l in ordered + leftover]
    else:
        sorted_items = sorted(usable.items(), key=lambda kv: _rank_key(kv[0]))

    if lang_colors is not None and lang_markers is not None:
        colors = [lang_colors.get(lang, LANG_DEFAULT_COLOR) for lang, _ in sorted_items]
        markers = [lang_markers.get(lang, PAPER_MARKERS[i % len(PAPER_MARKERS)])
                   for i, (lang, _) in enumerate(sorted_items)]
    else:
        _valid_baseline = {l: v["baseline"] for l, v in sorted_items if np.isfinite(v["baseline"])}
        _lc, _lm = {}, {}
        if _valid_baseline:
            _, _lc, _lm = _pp_lang_styles_by_mean_accuracy(_valid_baseline)
        colors = [_lc.get(lang, LANG_DEFAULT_COLOR) for lang, _ in sorted_items]
        markers = [_lm.get(lang, PAPER_MARKERS[i % len(PAPER_MARKERS)])
                   for i, (lang, _) in enumerate(sorted_items)]

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []
    for (lang, v), color, marker in zip(sorted_items, colors, markers):
        ax.plot(v["xs"], v["ys"], color=color, lw=linewidth, alpha=0.95)
        ax.scatter(v["xs"], v["ys"], color=color, marker=marker, s=dot_size,
                   alpha=0.9, linewidths=0)
        # auc_above is stored baseline-subtracted (raw_auc - 0.5, see
        # error_recall_auc_above_diagonal = trapz(ys - xs, xs)); add 0.5 back for
        # display so the legend shows raw AUC and random baseline reads as 0.5.
        _auc_str = f"{v['auc_above'] + 0.5:.3f}" if np.isfinite(v["auc_above"]) else "n/a"
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle="-",
                   markersize=9, label=f"{_pp_clean_label(v['label'])} (AUC={_auc_str})")
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5)
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Random baseline (x=y)")
    )

    ax.set_xlabel("Fraction of examples included")
    ax.set_ylabel("Fraction of total errors captured")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(
        handles=legend_handles, ncol=1, fontsize=10,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False,
    )

    fig.tight_layout()

    save_path = Path("paper_plots") / "error_recall_2d.svg" if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", format="svg")
    print(f"Saved plot: {save_path}")

    if table_save_path is None:
        table_save_path = save_path.with_name(save_path.stem + "_table.svg")
    _pp_save_metrics_table_svg(sorted_items, colors, markers, table_save_path,
                                value_cols=(("AUCad", "auc_above"), ("ErrRate", "error_rate")))

    if show:
        plt.show()
    plt.close(fig)


def plot_3d_cumulative_surface_by_language(
    json_path,
    save_path="paper_plots/exp1_cumulative_accuracy_3d_by_language.svg",
    *,
    remove_langs=("en", "english"),
    title="Cumulative Accuracy by Cosine Distance",
    figsize=(6.0, 7.2),
    dot_size=3.2,
    linewidth=2.0,
    show=False,
):
    """3D cumulative plot: x=percentile, y=language, z=cumulative accuracy."""
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    curves = data.get("curves", {})
    if not curves:
        print(f"[skip] expected curves JSON with data['curves']; got keys: {data.keys()}")
        return

    usable = {}
    for lang, curve in curves.items():
        if lang.lower() in remove_langs:
            continue
        xs = _pp_as_float_array(curve["xs"])
        ys = _pp_as_float_array(curve["ys"])
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            continue
        auc = _pp_curve_auc(xs, ys)
        usable[lang] = {"xs": xs, "ys": ys, "label": curve.get("label", lang), "auc": auc}

    if not usable:
        print(f"[skip] no usable non-English curves in {json_path}")
        return

    sorted_items = sorted(
        usable.items(),
        key=lambda kv: np.inf if not np.isfinite(kv[1]["auc"]) else kv[1]["auc"],
    )
    n = len(sorted_items)
    y_positions = np.arange(n)

    tab20 = plt.get_cmap("tab20")
    tab20_rainbow_order = [0, 1, 18, 19, 4, 5, 16, 17, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    colors = [tab20(tab20_rainbow_order[i % len(tab20_rainbow_order)]) for i in range(n)]

    all_z = np.concatenate([v["ys"] for _, v in sorted_items])
    all_z = all_z[np.isfinite(all_z)]
    zmin, zmax = float(np.min(all_z)), float(np.max(all_z))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for y_idx, ((lang, v), color) in enumerate(zip(sorted_items, colors)):
        xs, zs = v["xs"], v["ys"]
        X = np.vstack([xs, xs])
        Y = np.vstack([np.full_like(xs, y_idx), np.full_like(xs, y_idx)])
        Z = np.vstack([np.full_like(zs, zmin), zs])
        ax.plot_surface(X, Y, Z, color=color, alpha=0.1, linewidth=0, antialiased=True, shade=True)
        ax.plot(xs, np.full_like(xs, y_idx), zs, color=color, lw=linewidth,
                marker="o", markersize=dot_size, alpha=0.98)
        for x, z in zip(xs, zs):
            ax.plot([x, x], [y_idx, y_idx], [zmin, z], color=color, lw=0.6, alpha=0.22)

    ax.set_title(title, pad=14)
    ax.set_xlabel("Cosine distance percentile", labelpad=10)
    ax.set_ylabel("Language", labelpad=10)
    ax.set_zlabel("Cumulative Accuracy", labelpad=10)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_zlim(zmin, zmax)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lang for lang, _ in sorted_items])
    ax.view_init(elev=20, azim=-70)
    ax.set_box_aspect((1.25, 1.25, 0.75))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(1.05, 1.0), frameon=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", format="svg")
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def load_pr_auc_summary_for_plot(
    input_path,
    *,
    feature="vec_subspace_angle_by_cumulative_coherence",
    base_or_random_mode="en_fact",
    coherence_tag="coherence",
    remove_langs=("en", "english"),
    preferred_lang_order=("nl", "ru", "fr", "es", "zh", "hu", "uk", "vi", "ja", "ko"),
):
    """
    Load multilingual PR-AUC and random baseline values from:
        input_path/{feature}_lowest_feature_first/exp1_dev_test_split/
            exp1_{base_or_random_mode}_pr_auc_summary_{coherence_tag}.json

    coherence_tag=None (or "") loads the main (non-coherence) summary:
        exp1_{base_or_random_mode}_pr_auc_summary.json
    Lives next to the cumulative-curve JSON/svg — no separate per-mode folder.
    """
    input_path = Path(input_path)
    summary_filename = (
        f"exp1_{base_or_random_mode}_pr_auc_summary.json" if not coherence_tag
        else f"exp1_{base_or_random_mode}_pr_auc_summary_{coherence_tag}.json"
    )
    summary_path = (
        input_path
        / f"{feature}_lowest_feature_first"
        / "exp1_dev_test_split"
        / summary_filename
    )

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing PR-AUC summary file: {summary_path}")

    with open(summary_path, "r") as f:
        data = json.load(f)

    if "summary" in data and isinstance(data["summary"], dict):
        data = data["summary"]

    rows = {}
    for lang, vals in data.items():
        clean_lang = str(lang).lower()
        if clean_lang in remove_langs:
            continue
        if not isinstance(vals, dict):
            continue
        pr = vals.get("pr_auc") or vals.get("PR-AUC") or vals.get("auc") or vals.get("AUC")
        base = (vals.get("random_baseline") or vals.get("baseline")
                or vals.get("failure_rate") or vals.get("random"))
        if pr is None or base is None:
            print(f"[skip] {lang}: could not find pr_auc/random_baseline keys. keys={list(vals.keys())}")
            continue
        rows[clean_lang] = {"pr_auc": float(pr), "random_baseline": float(base)}

    langs = [l for l in preferred_lang_order if l in rows]
    langs += sorted([l for l in rows if l not in langs])

    pr_auc = np.array([rows[l]["pr_auc"] for l in langs], dtype=float)
    random_baseline = np.array([rows[l]["random_baseline"] for l in langs], dtype=float)

    print(f"Loaded PR-AUC summary from: {summary_path}")
    for l, pr, base in zip(langs, pr_auc, random_baseline):
        print(f"  {l}: PR-AUC={pr:.3f}, random_baseline={base:.3f}")

    return langs, pr_auc, random_baseline


def plot_multilingual_pr_auc_baseline_vertical(
    save_plot_path="paper_plots/multilingual_pr_auc_baseline_plot.svg",
    save_table_path="paper_plots/multilingual_pr_auc_baseline_table.svg",
    *,
    langs=None,
    pr_auc=None,
    failure_rate=None,
    lang_order: list | None = None,
    lang_colors: dict | None = None,
    reverse_order: bool = False,
    show=False,
    dpi=600,
):
    """
    lang_order/lang_colors: pass the shared mapping built by
    _pp_lang_styles_by_mean_accuracy (once per run, from the main cumulative
    pass) so color matches the cumulative/non-cumulative plots for that run.
    lang_order also sets left-to-right position (ascending mean accuracy,
    i.e. blue on the left); pass reverse_order=True to lay out bars from
    highest accuracy (brown) on the left to lowest (blue) on the right —
    colors are unaffected, only the x position changes. If lang_order is
    omitted, falls back to ranking languages by this call's own mean accuracy
    (1 - failure_rate).
    """
    def _lighten_color(color, amount=0.55):
        rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple(rgb + (white - rgb) * amount)

    if langs is None:
        langs = ["nl", "ru", "fr", "es", "zh", "hu", "uk", "vi", "ja", "ko"]
    if pr_auc is None:
        raise ValueError("pr_auc must be provided.")
    if failure_rate is None:
        raise ValueError("failure_rate/random_baseline must be provided.")

    langs = list(langs)
    pr_auc = np.asarray(pr_auc, dtype=float)
    failure_rate = np.asarray(failure_rate, dtype=float)

    if len(langs) != len(pr_auc) or len(langs) != len(failure_rate):
        raise ValueError(
            f"Length mismatch: len(langs)={len(langs)}, "
            f"len(pr_auc)={len(pr_auc)}, len(failure_rate)={len(failure_rate)}"
        )

    by_lang = dict(zip(langs, zip(pr_auc, failure_rate)))
    if lang_order is not None:
        ordered = [l for l in lang_order if l in by_lang]
        # Any language missing from lang_order (shouldn't normally happen):
        # append ranked ascending by mean accuracy (1 - failure_rate), same
        # convention as _pp_lang_styles_by_mean_accuracy.
        leftover = sorted((l for l in by_lang if l not in ordered),
                          key=lambda l: 1.0 - by_lang[l][1])
        langs = ordered + leftover
    if reverse_order:
        langs = list(reversed(langs))
    pr_auc = np.array([by_lang[l][0] for l in langs], dtype=float)
    failure_rate = np.array([by_lang[l][1] for l in langs], dtype=float)

    if lang_colors is None:
        mean_acc_local = {l: 1.0 - fr for l, fr in zip(langs, failure_rate)}
        _, lang_colors, _ = _pp_lang_styles_by_mean_accuracy(mean_acc_local)

    baseline_color = "#D9D9D9"
    default_color = LANG_DEFAULT_COLOR

    # ── 1) bar plot ──
    x = np.arange(len(langs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for i, lang in enumerate(langs):
        color = lang_colors.get(lang, default_color)
        ax.bar(x[i] - width / 2, pr_auc[i], width=width, color=color,
               edgecolor="white", linewidth=0.8, zorder=3)

    ax.bar(x + width / 2, failure_rate, width=width, color=baseline_color,
           edgecolor="white", linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.grid(axis="y", alpha=0.22, linestyle="--", zorder=1)

    legend_handles = [
        Patch(facecolor="#7F7F7F", edgecolor="white", label="PR-AUC"),
        Patch(facecolor=baseline_color, edgecolor="white", label="Random baseline"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=14)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color("black")

    fig.tight_layout()
    save_plot_path = Path(save_plot_path)
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_plot_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved plot to: {save_plot_path}")
    if show:
        plt.show()
    plt.close(fig)

    # ── 2) table ──
    table_rows = [
        [lang, f"{auc:.2f}", f"{base:.2f}"]
        for lang, auc, base in zip(langs, pr_auc, failure_rate)
    ]
    fig_tab, ax_tab = plt.subplots(figsize=(3.8, 4.8))
    ax_tab.axis("off")

    table = ax_tab.table(
        cellText=table_rows,
        colLabels=["Lang", "PR-AUC", "Baseline"],
        cellLoc="center", colLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.15, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("white")
        else:
            lang = langs[row - 1]
            color = lang_colors.get(lang, default_color)
            if col == 0:
                cell.set_facecolor(_lighten_color(color, amount=0.55))
                cell.set_text_props(weight="bold", color="black")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(color="black")

    fig_tab.tight_layout()
    save_table_path = Path(save_table_path)
    save_table_path.parent.mkdir(parents=True, exist_ok=True)
    fig_tab.savefig(save_table_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved table to: {save_table_path}")
    if show:
        plt.show()
    plt.close(fig_tab)


def _pp_make_minimal_distinct_formatter(ticks, max_decimals=6):
    ticks = np.asarray(ticks, dtype=float)
    ticks = ticks[np.isfinite(ticks)]
    if len(ticks) == 0:
        return FuncFormatter(lambda x, pos: f"{x:.2f}")
    for nd in range(0, max_decimals + 1):
        labels = [f"{t:.{nd}f}" for t in ticks]
        if len(set(labels)) == len(labels):
            return FuncFormatter(lambda x, pos: f"{x:.{nd}f}")
    return FuncFormatter(lambda x, pos: f"{x:.{max_decimals}f}")


def _pp_padded_limits(vals, pad_frac=0.10):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    vmin, vmax = float(vals.min()), float(vals.max())
    span = vmax - vmin
    if span <= 1e-12:
        pad = 0.05 if abs(vmax) <= 1e-12 else 0.08 * abs(vmax)
    else:
        pad = pad_frac * span
    return vmin - pad, vmax + pad


def _pp_clipped_padded_limits(vals, pad_frac=0.10, clip=(0.0, 1.0)):
    vals = np.asarray(vals, dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) == 0:
        return clip if clip is not None else (0.0, 1.0)
    lo, hi = _pp_padded_limits(finite_vals, pad_frac=pad_frac)
    if clip is not None:
        lo = max(clip[0], lo)
        hi = min(clip[1], hi)
    if hi <= lo:
        mid = float(np.nanmean(finite_vals))
        if clip is None:
            lo, hi = mid - 0.02, mid + 0.02
        else:
            lo = max(clip[0], mid - 0.02)
            hi = min(clip[1], mid + 0.02)
            if hi <= lo:
                lo, hi = clip
    return lo, hi


def _pp_apply_three_ticks_clipped_axes(ax, xs, ys, *, x_clip=(0.0, 1.0), y_clip=(0.0, 1.0),
                                        pad_frac=0.10, tick_labelsize=24):
    x_lo, x_hi = _pp_clipped_padded_limits(xs, pad_frac=pad_frac, clip=x_clip)
    y_lo, y_hi = _pp_clipped_padded_limits(ys, pad_frac=pad_frac, clip=y_clip)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    x_ticks = np.linspace(x_lo, x_hi, 3)
    y_ticks = np.linspace(y_lo, y_hi, 3)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.xaxis.set_major_formatter(_pp_make_minimal_distinct_formatter(x_ticks))
    ax.yaxis.set_major_formatter(_pp_make_minimal_distinct_formatter(y_ticks))
    ax.tick_params(axis="both", labelsize=tick_labelsize, width=2.0, length=6)


def replot_exp2_scatter_from_json(json_path, save_dir=None, figsize=(8, 6), show=False):
    """One scatter plot per language: x=CI=1-orthogonality, y=mean accuracy, per relation."""
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    langs = data.get("langs", {})
    if not langs:
        print(f"[skip] no lang data in {json_path}")
        return

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    results = {}

    for lang, ldata in langs.items():
        relations = ldata.get("relations", {})
        if not relations:
            continue

        raw_xs, ys, tasks = [], [], []
        for rel, pt in relations.items():
            x = pt.get("x", np.nan)
            y = pt.get("y", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                raw_xs.append(float(x))
                ys.append(float(y))
                tasks.append(rel)

        if len(raw_xs) == 0:
            continue

        raw_xs = np.asarray(raw_xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        xs = 1.0 - raw_xs

        if len(xs) >= 2 and np.std(xs) > 1e-12 and np.std(ys) > 1e-12:
            r_val, p_val = pearsonr(xs, ys)
            r_val, p_val = float(r_val), float(p_val)
        else:
            r_val, p_val = np.nan, np.nan

        results[lang] = {"x": xs, "y": ys, "tasks": tasks, "pearson_r": r_val, "pearson_p": p_val}

        unique_tasks = sorted(set(tasks))
        cmap = plt.get_cmap("tab20", len(unique_tasks))
        task_to_color = {t: cmap(i) for i, t in enumerate(unique_tasks)}
        task_to_marker = {t: markers[i % len(markers)] for i, t in enumerate(unique_tasks)}

        fig, ax = plt.subplots(figsize=figsize)
        for t in unique_tasks:
            idx = [i for i, tt in enumerate(tasks) if tt == t]
            ax.scatter(xs[idx], ys[idx], s=180, color=task_to_color[t], marker=task_to_marker[t],
                       edgecolor="black", linewidth=0.4, alpha=0.9, label=t)

        if len(xs) > 1 and np.std(xs) > 1e-12:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, m * x_line + b, color="black", linestyle="--", linewidth=2.0,
                    alpha=0.8, label="fit")

        ax.set_xlabel("CI")
        ax.set_ylabel("Mean Accuracy")
        if np.isfinite(r_val):
            ps = "p<1e-3" if p_val < 1e-3 else f"p={p_val:.2g}"
            ax.set_title(f"{lang}: Topic Accuracy and CI\n$r={r_val:.2f}$, {ps}")
        else:
            ax.set_title(f"{lang}: Topic Accuracy and CI")

        ax.grid(alpha=0.25, linestyle="--")
        _pp_apply_three_ticks_clipped_axes(ax, xs, ys, x_clip=(0.0, 1.0), y_clip=(0.0, 1.0),
                                            pad_frac=0.10, tick_labelsize=24)
        ax.legend(title="Dataset", fontsize=7, title_fontsize=9,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"{json_path.stem}_{lang}.svg"
            fig.savefig(save_path, bbox_inches="tight", format="svg")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXP 1 — Dev/test Oscar split, en-fact orthogonality (same feature as above
#          but W_L built from dev Oscar rows for layer selection and test Oscar
#          rows for evaluation)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 1] Dev/test Oscar split orthogonality experiment")
exp1_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature}_lowest_feature_first", "exp1_dev_test_split")

en_fact_dev_best_layers_exp1 = None  # populated below when base_or_random_mode == 'en_fact'
exp1_auc_tables = {}
pr_auc_tables_exp1 = {}

for base_or_random_mode in (base_or_random_mode_list if DO_EXP1 else []):
    print(f"[Exp 1] mode={base_or_random_mode}")

    # DEV phase: compute features using dev Oscar W_L
    outs_dev_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_dev_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', store_X_base=False,
            sv_weight_mode=SV_WEIGHT_MODE,
        )

    layers_run_dev = sorted(outs_dev_by_layer.keys())
    if not layers_run_dev:
        print(f"[Exp 1] no dev layers for mode={base_or_random_mode}")
        continue

    # Select best layer using dev Oscar features on dev (train) en-facts
    tt_dev = tt_select_best_layer_with_split(
        outs_dev_by_layer, valid_langs=valid_langs,
        layers=layers_run_dev, seed=0, y_layer="last",
        min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
    )
    dev_best_layers = {
        lang: int(info["best_layer"])
        for lang, info in tt_dev.items()
        if info.get("best_layer") is not None
    }
    if base_or_random_mode == 'en_fact':
        en_fact_dev_best_layers_exp1 = dict(dev_best_layers)

    # TEST phase: compute features using test Oscar W_L
    outs_test_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_test_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', store_X_base=False,
            sv_weight_mode=SV_WEIGHT_MODE,
        )

    # Build y_acc for test set
    y_acc_by_lang_test = None
    if y_mode == "accuracy":
        try:
            y_acc_by_lang_test = build_y_acc_by_lang_from_predictions_jsonl(path, tt_dev)
        except Exception as e:
            print(f"[Exp 1] warn: could not build y_acc: {e}")

    os.makedirs(exp1_plot_root, exist_ok=True)

    # Cumulative curve: test Oscar features at dev-selected layer, test en-facts.
    # save_path=None here — the raw render is unconditionally replaced a few
    # lines below by replot_exp1_cumulative_from_json at the very same path,
    # so drawing+encoding an SVG for it here would be pure waste. Only the
    # data JSON (consumed by the replot) is written.
    _exp1_auc_out, _ = plot_cumulative_all_layers_all_langs(
        outs_test_by_layer, tt_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_test, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="cumulative",
        title=f"Cumulative Curve: {base_or_random_mode}",
        save_path=None,
        save_data_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_cumulative.json"),
        calculate_auc=True, show=False,
        fixed_layers_to_plot=dev_best_layers, y_lim=(0.0, 1.0),
        pb_corr_subset="test",
        pb_corr_save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_pb_corr.json"),
    )
    exp1_auc_tables[base_or_random_mode] = _auc_out_to_pp(_exp1_auc_out)

    # Rank languages once by mean accuracy (blue=lowest ... brown=highest) and
    # reuse this exact color/marker/order everywhere for this run: the
    # cumulative curve (incl. every coherence-tag variant below), the
    # non-cumulative curve, and the PR-AUC bar plots.
    exp1_lang_order, exp1_lang_colors, exp1_lang_markers = _pp_lang_styles_by_mean_accuracy(
        _exp1_auc_out.get("baseline_auc", {})
    )

    # Replace the cumulative curve directly with the paper style (ported from
    # replot_multilingual_facts.ipynb), reading back the JSON just written.
    try:
        with plt.rc_context(PAPER_RCPARAMS):
            replot_exp1_cumulative_from_json(
                os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_cumulative.json"),
                save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_cumulative.svg"),
                table_save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_cumulative_table.svg"),
                lang_order=exp1_lang_order, lang_colors=exp1_lang_colors, lang_markers=exp1_lang_markers,
            )
    except Exception as _pp_exp1_err:
        print(f"[warn] could not render paper-style Exp1 cumulative plot: {_pp_exp1_err}")
    # Non-cumulative curve — same paper styling + language colors as the
    # cumulative and PR-AUC plots for this run. save_path=None: replaced below
    # by replot_exp1_noncumulative_from_json at the same path (see comment on
    # the cumulative call above).
    with plt.rc_context(PAPER_RCPARAMS):
        plot_cumulative_all_layers_all_langs(
            outs_test_by_layer, tt_dev,
            valid_langs=valid_langs, y_mode=y_mode,
            y_acc_by_lang=y_acc_by_lang_test, y_layer="last",
            quantile_step=quantile_step, cumulative_mode="non_cumulative",
            title=f"Noncumulative Curve: {base_or_random_mode}",
            save_path=None,
            save_data_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_noncumulative.json"),
            calculate_auc=False, show=False,
            fixed_layers_to_plot=dev_best_layers, y_lim=(0.0, 1.0),
            lang_colors=exp1_lang_colors,
        )
    # Replace it in place with the paper style, same colors/order as the
    # cumulative curve and PR-AUC bars for this run.
    try:
        with plt.rc_context(PAPER_RCPARAMS):
            replot_exp1_noncumulative_from_json(
                os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_noncumulative.json"),
                save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_noncumulative.svg"),
                table_save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_noncumulative_table.svg"),
                lang_order=exp1_lang_order, lang_colors=exp1_lang_colors, lang_markers=exp1_lang_markers,
            )
    except Exception as _pp_exp1_nc_err:
        print(f"[warn] could not render paper-style Exp1 non-cumulative plot: {_pp_exp1_nc_err}")

    # Error-recall curve (one plot with all languages overlaid) — same paper
    # styling + language colors as the cumulative/non-cumulative/PR-AUC plots
    # for this run. save_path=None: replaced below by
    # replot_exp1_error_recall_from_json at the same path.
    with plt.rc_context(PAPER_RCPARAMS):
        plot_error_recall_all_layers_all_langs(
            outs_test_by_layer, tt_dev,
            valid_langs=valid_langs, y_mode=y_mode,
            y_acc_by_lang=y_acc_by_lang_test, y_layer="last",
            title=f"Error Recall Curve: {base_or_random_mode}",
            save_path=None,
            save_data_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_error_recall.json"),
            show=False,
            fixed_layers_to_plot=dev_best_layers,
            lang_colors=exp1_lang_colors,
        )
    try:
        with plt.rc_context(PAPER_RCPARAMS):
            replot_exp1_error_recall_from_json(
                os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_error_recall.json"),
                save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_error_recall.svg"),
                table_save_path=os.path.join(exp1_plot_root, f"exp1_{base_or_random_mode}_error_recall_table.svg"),
                lang_order=exp1_lang_order, lang_colors=exp1_lang_colors, lang_markers=exp1_lang_markers,
            )
    except Exception as _pp_exp1_er_err:
        print(f"[warn] could not render paper-style Exp1 error-recall plot: {_pp_exp1_er_err}")

    # ── PR-AUC (failure prediction) at dev-selected layer, test Oscar ────────
    # Summary JSON + the vertical bar plot live alongside the cumulative curve
    # (exp1_plot_root) rather than a separate per-mode folder; no per-language
    # precision-recall curve is plotted.
    if y_acc_by_lang_test is not None:
        _exp1_pr_summary = _compute_and_save_pr_auc_all_langs(
            outs_test_by_layer, tt_dev, valid_langs,
            best_layers=dev_best_layers,
            y_acc_by_lang=y_acc_by_lang_test,
            save_dir=exp1_plot_root,
            label=f"PR-AUC: {base_or_random_mode}",
            plot_curves=False,
            summary_filename=f"exp1_{base_or_random_mode}_pr_auc_summary.json",
            n_boot=200,
        )
        pr_auc_tables_exp1[base_or_random_mode] = _exp1_pr_summary

    # ── Imported-subspace negative control (SRC x TGT matrix) ────────────────
    # Only meaningful once per run, off of the real per-language subspaces
    # (base_or_random_mode == 'en_fact'); random/translation-vector modes have
    # no per-language B_ell to import across languages.
    if EXP1_SRC_TGT_sub_control_exp and base_or_random_mode == "en_fact" and y_acc_by_lang_test is not None:
        _src_tgt_source_langs = None
        _src_tgt_extra_dev_best_layers = None
        if EXP1_SRC_TGT_ALWAYS_INCLUDE_EN_AS_SOURCE and "en" not in valid_langs:
            # One-off extra dev pass scoped to "en" alone, so it gets its own
            # dev-selected best layer via the same x_ortho method as every
            # other language, without adding "en" to valid_langs anywhere else.
            _en_outs_dev_by_layer = {}
            for _layer in LAYERS_TO_LOOP:
                _en_outs_dev_by_layer[_layer] = compute_geometric_feature(
                    model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
                    valid_langs=["en"], layer=_layer, rep_kind=rep_kind,
                    lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
                    target_kind=target_kind, y_transform=y_transform,
                    translation_data_dirs=translation_data_dirs,
                    min_n=100, separate_correct_incorrect_examples=False,
                    feature=feature, base_or_random_mode=base_or_random_mode,
                    random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
                    rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
                    oscar_split='dev', store_X_base=False,
                    sv_weight_mode=SV_WEIGHT_MODE,
                )
            _en_layers_run_dev = sorted(_en_outs_dev_by_layer.keys())
            _en_tt_dev = tt_select_best_layer_with_split(
                _en_outs_dev_by_layer, valid_langs=["en"],
                layers=_en_layers_run_dev, seed=0, y_layer="last",
                min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
            )
            _en_best_layer = _en_tt_dev.get("en", {}).get("best_layer")
            if _en_best_layer is None:
                print("[Exp1 SRC-TGT] warn: could not select a dev-best layer for "
                      "'en'; it will be excluded as a source for this mode.")
            else:
                _src_tgt_source_langs = list(valid_langs) + ["en"]
                _src_tgt_extra_dev_best_layers = {"en": int(_en_best_layer)}
            del _en_outs_dev_by_layer, _en_tt_dev

        compute_exp1_src_tgt_subspace_control(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag, valid_langs=valid_langs,
            rep_kind=rep_kind, lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs, feature=feature,
            mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            source_langs=_src_tgt_source_langs,
            extra_dev_best_layers=_src_tgt_extra_dev_best_layers,
            sv_weight_mode=SV_WEIGHT_MODE, min_n=100,
            tt_dev=tt_dev, y_acc_by_lang_test=y_acc_by_lang_test,
            dev_best_layers=dev_best_layers,
            save_dir=exp1_plot_root, tag=f"{lang_mode_name}_{EXP1_SRC_TGT_METRIC_TAG}",
            x_key=EXP1_SRC_TGT_X_KEY, layer_select=EXP1_SRC_TGT_LAYER_SELECT_MODE,
            dev_layers=layers_run_dev, dev_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
        )

    if EXP1_PLOT_EXAMPLE_TABLE and base_or_random_mode == "en_fact":
        plot_and_save_example_table(
            outs_by_layer=outs_test_by_layer,
            best_layers=dev_best_layers,
            valid_langs=valid_langs,
            acc_map=_acc_map,
            n_per_relation=10,
            top_k_per_lang=300,
            save_dir=os.path.join(exp1_plot_root, "example_table_exp1"),
            title=f"{base_or_random_mode}: Orthogonality predictor — top-10 per relation",
            x_key=EXP1_TABLE_X_KEY,
        )

    del outs_dev_by_layer, outs_test_by_layer
    print(f"[Exp 1] done mode={base_or_random_mode}")

# ── Exp 1: 3D surface + PR-AUC bar/table, paper style (ported from
#    replot_multilingual_facts.ipynb; the 2D cumulative curve itself is
#    already replaced in place above) ──
if DO_EXP1 and en_fact_dev_best_layers_exp1 is not None:
    try:
        with plt.rc_context(PAPER_RCPARAMS):
            plot_3d_cumulative_surface_by_language(
                os.path.join(exp1_plot_root, "exp1_en_fact_cumulative.json"),
                save_path=os.path.join(exp1_plot_root, "exp1_en_fact_cumulative_3d.svg"),
            )
    except Exception as _pp1_err:
        print(f"[warn] could not generate Exp1 3D surface plot: {_pp1_err}")

    # PR-AUC vertical bar/table, paper style. `feature` is the cumulative-coherence
    # metric everywhere in Exp1, so there is only the one (main) PR-AUC summary to
    # plot here — this just plots what was already saved to disk above.
    _pr_auc_variants = [(None, "main")]
    for _pp_coh_tag, _pp_variant_label in _pr_auc_variants:
        try:
            _pp_langs, _pp_pr_auc, _pp_baseline = load_pr_auc_summary_for_plot(
                EXP_ROOT_DIR, feature=feature, base_or_random_mode="en_fact",
                coherence_tag=_pp_coh_tag,
            )
            _pp_pr_suffix = "" if _pp_coh_tag is None else f"_{_pp_coh_tag}"
            with plt.rc_context(PAPER_RCPARAMS):
                plot_multilingual_pr_auc_baseline_vertical(
                    save_plot_path=os.path.join(exp1_plot_root, f"exp1_en_fact_pr_auc{_pp_pr_suffix}.svg"),
                    save_table_path=os.path.join(exp1_plot_root, f"exp1_en_fact_pr_auc{_pp_pr_suffix}_table.svg"),
                    langs=_pp_langs, pr_auc=_pp_pr_auc, failure_rate=_pp_baseline,
                    lang_order=exp1_lang_order, lang_colors=exp1_lang_colors, reverse_order=True,
                    show=False,
                )
        except Exception as _pp2_err:
            print(f"[warn] could not generate paper-style PR-AUC bar/table ({_pp_variant_label}): {_pp2_err}")

# ── Exp 1 AUC table ──────────────────────────────────────────────────────────
if exp1_auc_tables:
    exp1_auc_dir = os.path.join(exp1_plot_root, "auc_tables")
    os.makedirs(exp1_auc_dir, exist_ok=True)

    exp1_df = pd.DataFrame(index=sorted(valid_langs))
    for _mode_name, _lang_to_pp in exp1_auc_tables.items():
        exp1_df[_mode_name] = pd.Series(_lang_to_pp)

    exp1_csv_path = os.path.join(exp1_auc_dir, f"pp_compare_y_{y_mode}_cumulative_devselected.csv")
    exp1_df.to_csv(exp1_csv_path, index_label="language")
    print("Saved Exp1 AUC table:", exp1_csv_path)

    try:
        _e1_raw_df, _e1_en_fact_series = load_auc_matrix(exp1_csv_path, valid_langs=valid_langs)
        _e1_base = exp1_csv_path.replace(".csv", "")
        plot_auc_heatmap(
            _e1_raw_df, _e1_en_fact_series,
            add_en_fact_row=True,
            save_path=_e1_base + "_heatmap.svg",
            title=os.path.basename(exp1_csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            _e1_raw_df, _e1_en_fact_series,
            add_en_fact_row=True,
            save_path=_e1_base + "_barplot.svg",
            title=os.path.basename(exp1_csv_path),
        )
    except Exception as _e1_err:
        print(f"[warn] could not plot Exp1 AUC for {exp1_csv_path}: {_e1_err}")

    _e1_ranked = []
    for _mode, _ltp in exp1_auc_tables.items():
        _vals = [v for v in _ltp.values() if np.isfinite(v)]
        _mean = float(np.nanmean(_vals)) if _vals else float("nan")
        _e1_ranked.append((_mode, _mean))
    _e1_ranked.sort(key=lambda t: t[1] if np.isfinite(t[1]) else -np.inf, reverse=True)
    _e1_rlabels = [t[0] for t in _e1_ranked]
    _e1_rvals = [t[1] for t in _e1_ranked]

    fig, ax = plt.subplots(figsize=(8, 6))
    _e1_x = np.arange(len(_e1_ranked))
    _e1_bars = ax.bar(_e1_x, _e1_rvals, color="#5A9BD5")
    ax.set_xticks(_e1_x)
    ax.set_xticklabels(_e1_rlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean PP across target languages")
    ax.set_title(f"[Exp1] Modes ranked by mean PP — {y_mode} cumulative")
    ax.grid(axis="y", alpha=0.3)
    _e1_fv = [v for v in _e1_rvals if np.isfinite(v)]
    if _e1_fv:
        ax.set_ylim(min(0, min(_e1_fv)) * 1.05, max(_e1_fv) * 1.1)
    for _b, _v in zip(_e1_bars, _e1_rvals):
        if np.isfinite(_v):
            ax.text(_b.get_x() + _b.get_width() / 2, _v, f"{_v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _e1_ranked_path = os.path.join(exp1_auc_dir, f"pp_ranked_modes_{y_mode}_cumulative.svg")
    fig.savefig(_e1_ranked_path, bbox_inches="tight", format="svg")
    print("Saved Exp1 ranked bar plot:", _e1_ranked_path)
    plt.show()
    plt.close(fig)

# ── Exp 1 PR-AUC table ───────────────────────────────────────────────────────
if pr_auc_tables_exp1:
    _pr1_dir = os.path.join(exp1_plot_root, "pr_auc_tables")
    os.makedirs(_pr1_dir, exist_ok=True)

    # Full JSON — keeps precision/recall arrays for replotting
    _pr1_json_path = os.path.join(_pr1_dir, "pr_auc_compare_devselected.json")
    with open(_pr1_json_path, "w") as _f:
        json.dump(pr_auc_tables_exp1, _f, indent=2)
    print("Saved Exp1 PR-AUC table:", _pr1_json_path)

    # CSV — rows=langs, cols=modes, for heatmap
    _pr1_df = pd.DataFrame(index=sorted(valid_langs))
    for _mode, _lang_stats in pr_auc_tables_exp1.items():
        _pr1_df[_mode] = pd.Series({lang: v["pr_auc"] for lang, v in _lang_stats.items()})
    _pr1_csv_path = os.path.join(_pr1_dir, "pr_auc_compare_devselected.csv")
    _pr1_df.to_csv(_pr1_csv_path, index_label="language")
    print("Saved Exp1 PR-AUC CSV:", _pr1_csv_path)

    try:
        _pr1_raw_df, _pr1_en_fact_series = load_auc_matrix(_pr1_csv_path, valid_langs=valid_langs)
        _pr1_base = _pr1_csv_path.replace(".csv", "")
        plot_auc_heatmap(
            _pr1_raw_df, _pr1_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr1_base + "_heatmap.svg",
            title=os.path.basename(_pr1_csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            _pr1_raw_df, _pr1_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr1_base + "_barplot.svg",
            title=os.path.basename(_pr1_csv_path),
        )
    except Exception as _pr1_err:
        print(f"[warn] could not plot Exp1 PR-AUC heatmap: {_pr1_err}")

# ═══════════════════════════════════════════════════════════════════════════
# EXP 1 — Second part: sweep W_L variance and layer, select best by dev AUC
# For each language, the (var_prop, layer) combination that gives the lowest
# dev AUC (= highest PP) is selected, then evaluated on test Oscar W_L.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 1 sweep_var_and_layer] Sweeping W_L variance and layer")
exp1_sweep_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature}_lowest_feature_first", "exp1_sweep_var_and_layer")

_exp1_sweep_ok = (
    DO_EXP1_SWEEP
    and DO_EXP1
    and isinstance(lang_mode, (tuple, list))
    and len(lang_mode) >= 3
)
if not _exp1_sweep_ok:
    print("[Exp 1 sweep_var_and_layer] Skipping: DO_EXP1_SWEEP=False or lang_mode is not a subspace tuple")
else:
    _sw_mode_base = str(lang_mode[0])
    _sw_method    = str(lang_mode[1])

    # ── Step 1: Dev phase — for each var_prop run all layers ────────────────
    _sw_dev_outs_by_varp = {}   # varp -> {layer: {lang: pack}}
    _sw_dev_tt_by_varp   = {}   # varp -> tt dict

    for _sw_vp in EXP1_SWEEP_VAR_PROPS:
        _sw_lm = (_sw_mode_base, _sw_method, _sw_vp)
        print(f"  [sweep dev] var_prop={_sw_vp}")
        _sw_dev_outs = {}
        for layer in LAYERS_TO_LOOP:
            _sw_dev_outs[layer] = compute_geometric_feature(
                model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
                valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
                lang_mode=_sw_lm, targets_jsonl_path=TARGETS_JSONL,
                target_kind=target_kind, y_transform=y_transform,
                translation_data_dirs=translation_data_dirs,
                min_n=100, separate_correct_incorrect_examples=False,
                feature=feature, base_or_random_mode='en_fact',
                random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
                rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
                oscar_split='dev', store_X_base=False,
                sv_weight_mode=SV_WEIGHT_MODE,
            )
        _sw_layers_run = sorted(_sw_dev_outs.keys())
        _sw_tt = tt_select_best_layer_with_split(
            _sw_dev_outs, valid_langs=valid_langs,
            layers=_sw_layers_run, seed=0, y_layer="last",
            min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
        )
        _sw_dev_outs_by_varp[_sw_vp] = _sw_dev_outs
        _sw_dev_tt_by_varp[_sw_vp]   = _sw_tt
        print(f"    var_prop={_sw_vp}: {len(_sw_tt)} langs have best-layer info")

    # ── Step 2: Select best global variance by win count ─────────────────────
    # For each language, whichever var_prop gives the lowest AUC (highest
    # best_train_pp, since PP = 1 - AUC/baseline) at its own best layer "wins"
    # that language.  The variance with the most wins is chosen globally.
    # Tiebreak: sum of best_train_pp across languages.
    _sw_lang_pp_per_var = {}   # lang -> {vp: best_train_pp}
    for lang in valid_langs:
        _pp_map = {}
        for _sw_vp in EXP1_SWEEP_VAR_PROPS:
            _tt_vp = _sw_dev_tt_by_varp.get(_sw_vp, {})
            if lang not in _tt_vp:
                continue
            _pp = _tt_vp[lang].get("best_train_pp", float("nan"))
            if np.isfinite(_pp):
                _pp_map[_sw_vp] = float(_pp)
        if _pp_map:
            _sw_lang_pp_per_var[lang] = _pp_map

    _sw_var_wins    = {vp: 0   for vp in EXP1_SWEEP_VAR_PROPS}
    _sw_var_pp_sum  = {vp: 0.0 for vp in EXP1_SWEEP_VAR_PROPS}
    for lang, _pp_map in _sw_lang_pp_per_var.items():
        _winner = max(_pp_map, key=_pp_map.get)
        _sw_var_wins[_winner] += 1
        for vp, pp in _pp_map.items():
            _sw_var_pp_sum[vp] += pp

    _sw_best_global_vp = max(
        EXP1_SWEEP_VAR_PROPS,
        key=lambda vp: (_sw_var_wins[vp], _sw_var_pp_sum[vp]),
    )
    print(f"  [sweep] Win counts per variance: { {vp: _sw_var_wins[vp] for vp in EXP1_SWEEP_VAR_PROPS} }")
    print(f"  [sweep] PP sums per variance:    { {vp: round(_sw_var_pp_sum[vp], 4) for vp in EXP1_SWEEP_VAR_PROPS} }")
    print(f"  [sweep] Selected global variance: {_sw_best_global_vp} ({_sw_var_wins[_sw_best_global_vp]} wins)")

    # Per-language best layer under the globally selected variance
    _sw_best_varp_layer = {}   # lang -> (best_global_vp, layer)
    _sw_global_tt = _sw_dev_tt_by_varp.get(_sw_best_global_vp, {})
    for lang in valid_langs:
        if lang not in _sw_global_tt:
            continue
        _bl = _sw_global_tt[lang].get("best_layer", None)
        if _bl is not None:
            _sw_best_varp_layer[lang] = (_sw_best_global_vp, int(_bl))

    print(f"  [sweep] best (var_prop={_sw_best_global_vp}, layer) selected for {len(_sw_best_varp_layer)} languages")
    for _l, (_v, _bl) in sorted(_sw_best_varp_layer.items()):
        print(f"    {_l}: var_prop={_v}, layer={_bl}")

    # ── Step 3: Test phase — one variance, unique best layers only ────────────
    _sw_lm             = (_sw_mode_base, _sw_method, _sw_best_global_vp)
    _sw_unique_layers  = sorted(set(bl for _, bl in _sw_best_varp_layer.values()))
    _sw_layer_to_id    = {layer: i for i, layer in enumerate(_sw_unique_layers)}
    _sw_combo_to_id    = {(_sw_best_global_vp, layer): _sw_layer_to_id[layer]
                          for layer in _sw_unique_layers}

    _sw_test_outs_by_layer = {}   # layer -> {lang: pack}
    for _sw_bl in _sw_unique_layers:
        print(f"  [sweep test] var_prop={_sw_best_global_vp}, layer={_sw_bl}")
        _sw_test_outs_by_layer[_sw_bl] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=_sw_bl, rep_kind=rep_kind,
            lang_mode=_sw_lm, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode='en_fact',
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', store_X_base=False,
            sv_weight_mode=SV_WEIGHT_MODE,
        )

    # Reindex by int id so plot_cumulative_all_layers_all_langs can use it
    _sw_outs_test = {
        _sw_layer_to_id[layer]: packs
        for layer, packs in _sw_test_outs_by_layer.items()
    }
    # Per-lang pointer: lang -> layer_id (its "layer" key in _sw_outs_test)
    _sw_fixed_layers = {
        lang: _sw_layer_to_id[bl]
        for lang, (_, bl) in _sw_best_varp_layer.items()
    }
    # tt: all languages use the globally selected variance's dev tt
    _sw_tt_combined = {
        lang: _sw_global_tt[lang]
        for lang in _sw_best_varp_layer
        if lang in _sw_global_tt
    }

    # ── Step 4: y_acc for test split ─────────────────────────────────────────
    _sw_y_acc_test = None
    if y_mode == "accuracy" and _sw_tt_combined:
        try:
            _sw_y_acc_test = build_y_acc_by_lang_from_predictions_jsonl(path, _sw_tt_combined)
        except Exception as _sw_e:
            print(f"  [sweep] warn: could not build y_acc: {_sw_e}")

    os.makedirs(exp1_sweep_plot_root, exist_ok=True)

    # ── Step 5: Cumulative and non-cumulative plots ───────────────────────────
    _sw_auc_out, _ = plot_cumulative_all_layers_all_langs(
        _sw_outs_test, _sw_tt_combined,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=_sw_y_acc_test, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="cumulative",
        title="[Exp1 sweep_var_and_layer] best (var_prop, layer) per lang, test Oscar W_L",
        save_path=os.path.join(exp1_sweep_plot_root, "exp1_sweep_cumulative.svg"),
        save_data_path=os.path.join(exp1_sweep_plot_root, "exp1_sweep_cumulative.json"),
        calculate_auc=True, show=False,
        fixed_layers_to_plot=_sw_fixed_layers, y_lim=(0.0, 1.0),
        pb_corr_subset="test",
        pb_corr_save_path=os.path.join(exp1_sweep_plot_root, "exp1_sweep_pb_corr.json"),
    )

    plot_cumulative_all_layers_all_langs(
        _sw_outs_test, _sw_tt_combined,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=_sw_y_acc_test, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="non_cumulative",
        title="[Exp1 sweep_var_and_layer] best (var_prop, layer) per lang, test Oscar W_L (non-cumul)",
        save_path=os.path.join(exp1_sweep_plot_root, "exp1_sweep_noncumulative.svg"),
        save_data_path=os.path.join(exp1_sweep_plot_root, "exp1_sweep_noncumulative.json"),
        calculate_auc=False, show=False,
        fixed_layers_to_plot=_sw_fixed_layers, y_lim=(0.0, 1.0),
    )

    # ── Step 6: Summary plot — selected var_prop and layer per language ───────
    if _sw_best_varp_layer:
        from matplotlib.patches import Patch as _SwPatch
        _sw_langs_sorted  = sorted(_sw_best_varp_layer.keys())
        _sw_sel_varp      = [_sw_best_varp_layer[l][0] for l in _sw_langs_sorted]
        _sw_sel_layer     = [_sw_best_varp_layer[l][1] for l in _sw_langs_sorted]
        _sw_vp_colors     = {0.85: "#4C72B0", 0.90: "#DD8452", 0.95: "#55A868", 0.99: "#C44E52"}
        _sw_bar_colors    = [_sw_vp_colors.get(v, "#888888") for v in _sw_sel_varp]
        _sw_x             = np.arange(len(_sw_langs_sorted))

        fig_sw, (ax_sw1, ax_sw2) = plt.subplots(2, 1, figsize=(max(8, 0.6 * len(_sw_langs_sorted)), 8))

        ax_sw1.bar(_sw_x, _sw_sel_varp, color=_sw_bar_colors)
        ax_sw1.set_xticks(_sw_x)
        ax_sw1.set_xticklabels(_sw_langs_sorted, rotation=45, ha="right", fontsize=8)
        ax_sw1.set_ylabel("Selected var_prop")
        ax_sw1.set_title("[Exp1 sweep] Selected W_L variance per language")
        ax_sw1.grid(axis="y", alpha=0.3)
        ax_sw1.set_ylim(0.80, 1.03)
        for _xi, _v in zip(_sw_x, _sw_sel_varp):
            ax_sw1.text(_xi, _v + 0.001, f"{_v}", ha="center", va="bottom", fontsize=8)
        _sw_legend_els = [_SwPatch(facecolor=c, label=f"var={v}") for v, c in sorted(_sw_vp_colors.items())]
        ax_sw1.legend(handles=_sw_legend_els, loc="upper right", fontsize=8)

        ax_sw2.bar(_sw_x, _sw_sel_layer, color=_sw_bar_colors)
        ax_sw2.set_xticks(_sw_x)
        ax_sw2.set_xticklabels(_sw_langs_sorted, rotation=45, ha="right", fontsize=8)
        ax_sw2.set_ylabel("Selected layer")
        ax_sw2.set_title("[Exp1 sweep] Selected layer per language")
        ax_sw2.grid(axis="y", alpha=0.3)
        for _xi, _v in zip(_sw_x, _sw_sel_layer):
            ax_sw2.text(_xi, _v + 0.1, str(_v), ha="center", va="bottom", fontsize=8)

        fig_sw.tight_layout()
        _sw_summary_path = os.path.join(exp1_sweep_plot_root, "exp1_sweep_selected_varp_layer.svg")
        fig_sw.savefig(_sw_summary_path, bbox_inches="tight", format="svg")
        print("[Exp1 sweep] Saved selection summary:", _sw_summary_path)
        plt.show()
        plt.close(fig_sw)

    # ── Step 7: Save best (var_prop, layer) selection as JSON and CSV ─────────
    _sw_sel_dict = {
        lang: {"var_prop": float(vp), "layer": int(bl)}
        for lang, (vp, bl) in _sw_best_varp_layer.items()
    }
    _sw_sel_json = os.path.join(exp1_sweep_plot_root, "exp1_sweep_best_varp_layer.json")
    with open(_sw_sel_json, "w") as _f:
        json.dump(_sw_sel_dict, _f, indent=2)
    print("[Exp1 sweep] Saved best (var_prop, layer) table:", _sw_sel_json)

    # ── Step 8: PP summary table ─────────────────────────────────────────────
    if _sw_auc_out:
        _sw_pp_by_lang = _auc_out_to_pp(_sw_auc_out)
        _sw_vals       = [v for v in _sw_pp_by_lang.values() if np.isfinite(v)]
        _sw_mean_pp    = float(np.nanmean(_sw_vals)) if _sw_vals else float("nan")
        print(f"[Exp1 sweep] Mean PP across languages: {_sw_mean_pp:.4f}")

        import csv as _csv_sw
        _sw_pp_csv = os.path.join(exp1_sweep_plot_root, "exp1_sweep_pp_by_lang.csv")
        with open(_sw_pp_csv, "w", newline="") as _f:
            _wr = _csv_sw.writer(_f)
            _wr.writerow(["language", "pp", "var_prop", "layer"])
            for _l in sorted(valid_langs):
                _vbl = _sw_best_varp_layer.get(_l, (None, None))
                _wr.writerow([_l, _sw_pp_by_lang.get(_l, ""), _vbl[0], _vbl[1]])
        print("[Exp1 sweep] Saved PP table:", _sw_pp_csv)

    del _sw_dev_outs_by_varp, _sw_test_outs_by_layer
    print("[Exp1 sweep_var_and_layer] done")

# ── Gate exp2+ on en_fact mode ────────────────────────────────────────────────
base_or_random_mode_list_exp2plus = base_or_random_mode_list if base_or_random_mode == 'en_fact' else []
if not base_or_random_mode_list_exp2plus:
    print("[Exp 2+] Skipping: base_or_random_mode != 'en_fact'")


# ═══════════════════════════════════════════════════════════════════════════
# EXP 2 — Per-relation en-fact subspace vs Oscar W_L (simplified)
#          Uses PP-selected best layer per language from Exp 1 (en_fact mode).
#          Plots scatter: x=subspace angle metric, y=y_mean per relation.
#          Two versions: dev PP-selected layers AND dev-corr-selected layers.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 2] Per-relation en-fact subspace vs Oscar W_L (simplified, Exp 1 best layer)")

_exp2_ok = (
    DO_EXP2
    #and en_fact_dev_best_layers_exp1 is not None
    and base_or_random_mode == 'en_fact'
    and isinstance(lang_mode, (tuple, list)) and len(lang_mode) >= 3
)
if not _exp2_ok:
    print("[Exp 2] Skipping: DO_EXP2=False or requires en_fact_dev_best_layers_exp1 from Exp 1, base_or_random_mode='en_fact', and subspace lang_mode.")
else:
    from scipy.stats import pearsonr as _pr_e2

    _, _sm_L2, _vp_L2 = lang_mode[0], lang_mode[1], float(lang_mode[2])
    _subspace_mode_name_exp2 = str(lang_mode[0]).lower()
    _exp2_use_center_oscar = _subspace_mode_name_exp2 in (
        "center_oscar_and_uncentered_language_subspace",
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )
    _exp2_use_center_oscar_per_lang = _subspace_mode_name_exp2 in (
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )
    _all_full_langs_exp2 = list(abbr_to_full_LANGUAGE_CODE_MAP.values())

    exp2_plot_root = os.path.join(
        EXP_ROOT_DIR, f"exp2_relation_subspace_en_fact_best_layer_relation_var_{DATASET_VAR_PROP}",
    )
    exp2_plot_root_v2 = os.path.join(
        EXP_ROOT_DIR, f"exp2_relation_subspace_dev_corr_selected_relation_var_{DATASET_VAR_PROP}",
    )
    exp2_plot_root_v2_dev = os.path.join(
        EXP_ROOT_DIR, f"exp2_relation_subspace_dev_corr_selected_dev_set_relation_var_{DATASET_VAR_PROP}",
    )

    print(f"[Exp 2] Precomputing test en-fact features at all layers: {LAYERS_TO_LOOP}")
    _exp2_test_outs = {}
    for _layer2 in LAYERS_TO_LOOP:
        _exp2_test_outs[_layer2] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=_layer2, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=MIN_EXAMPLES, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode='en_fact',
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', sv_weight_mode="sv",
        )
    print("[Exp 2 v2] Precomputing dev en-fact features at all layers for corr-based layer selection...")
    _exp2_dev_outs = {}
    for _layer2 in LAYERS_TO_LOOP:
        _exp2_dev_outs[_layer2] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=_layer2, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=MIN_EXAMPLES, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode='en_fact',
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', sv_weight_mode="sv",
        )

    def _exp2_build_rel_metrics(layer_source_dict, sv_weight_mode="sv"):
        _grand_mean_cache_exp2 = {}
        # Phase 1: load all language subspaces, grouped by layer, and cache packs.
        # oscar_W_cached is disk-cached so repeated calls are cheap.
        _loaded = {}        # lang -> (pack, W_L_ortho, mu_L_arr, sv_WL, best_layer)
        for _lang2 in valid_langs:
            _best_L2 = layer_source_dict.get(_lang2)
            if _best_L2 is None or _best_L2 not in _exp2_test_outs:
                continue
            _pack2 = _exp2_test_outs[_best_L2].get(_lang2)
            if _pack2 is None:
                continue
            _full2 = abbr_to_full_LANGUAGE_CODE_MAP.get(_lang2, _lang2)
            if _exp2_use_center_oscar:
                if _best_L2 not in _grand_mean_cache_exp2:
                    _grand_mean_cache_exp2[_best_L2] = _oscar_global_mean_cached(
                        _best_L2, _all_full_langs_exp2,
                        oscar_resids_root, oscar_cache_root, max_oscar_rows, verbose=False,
                    )
                _WL2, _muL2, _svWL2 = oscar_W_global_centered_cached(
                    lang=_full2, layer=_best_L2,
                    subspace_method=_sm_L2, var_prop=_vp_L2,
                    oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                    grand_mean=_grand_mean_cache_exp2[_best_L2],
                    per_lang_center=_exp2_use_center_oscar_per_lang,
                    max_rows=max_oscar_rows, verbose=False,
                )
            else:
                _WL2, _muL2, _svWL2 = oscar_W_cached(
                    lang=_full2, layer=_best_L2,
                    subspace_method=_sm_L2, var_prop=_vp_L2,
                    oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                    max_rows=max_oscar_rows, verbose=False, oscar_split='test',
                )
            if _WL2 is None:
                continue
            _WL2_ortho = _orthonormalize(np.asarray(_WL2, dtype=np.float64))
            _muL2_arr = None if _muL2 is None else np.asarray(_muL2, dtype=np.float64)
            _svWL2_arr = None if _svWL2 is None else np.asarray(_svWL2, dtype=np.float64)
            _loaded[_lang2] = (_pack2, _WL2_ortho, _muL2_arr, _svWL2_arr, _best_L2)
    
        rel_metrics = {}
        for _lang2, (_pack2, _WL2_ortho, _muL2_arr, _svWL2_arr, _best_L2) in _loaded.items():
            rel_metrics[_lang2] = compute_relation_subspace_metrics_for_layer(
                _pack2, _WL2_ortho, _muL2_arr,
                var_prop_en=DATASET_VAR_PROP, center=SUBSPACE_CENTERED,
                min_rel_n=MIN_EXAMPLES,
                sv_WL=_svWL2_arr, sv_weight_mode=sv_weight_mode,
            )
        return rel_metrics

    def _exp2_plot(rel_metrics, layer_source_dict, plot_root, version_label, only_metric=None, only_metrics_set=None, file_suffix=""):
        os.makedirs(plot_root, exist_ok=True)

        if only_metric is not None:
            _metrics_to_plot = [only_metric]
        elif only_metrics_set is not None:
            _metrics_to_plot = [m for m in ACTIVE_METRICS if m in only_metrics_set]
        else:
            _metrics_to_plot = ACTIVE_METRICS
        for _metric2 in _metrics_to_plot:
            _x_label2 = METRIC_LABELS.get(_metric2, _metric2)
            _all_mv2, _all_yv2 = [], []
            _json_langs = {}  # lang -> JSON record for this metric

            for _lang2 in valid_langs:
                if _lang2 not in rel_metrics:
                    continue
                _best_L2 = layer_source_dict.get(_lang2)
                # Compute within-language Pearson r for legend label
                _pairs2 = [
                    (rd.get(_metric2, np.nan), rd.get("y_mean", np.nan))
                    for rd in rel_metrics[_lang2].values()
                ]
                _lx2 = np.array([p[0] for p in _pairs2], dtype=np.float64)
                _ly2 = np.array([p[1] for p in _pairs2], dtype=np.float64)
                _lfin2 = np.isfinite(_lx2) & np.isfinite(_ly2)
                _within_r = None
                if _lfin2.sum() >= 3 and np.std(_lx2[_lfin2]) > 0 and np.std(_ly2[_lfin2]) > 0:
                    _lr2, _ = _pr_e2(_lx2[_lfin2], _ly2[_lfin2])
                    _within_r = float(_lr2)
                    _lbl2 = f"{_lang2}(L{_best_L2}) r={_lr2:.2f}"
                else:
                    _lbl2 = f"{_lang2}(L{_best_L2})"
                _lang_pts = {}
                for _rel2, _rd2 in rel_metrics[_lang2].items():
                    _v2 = _rd2.get(_metric2, np.nan)
                    _ym2 = _rd2.get("y_mean", np.nan)
                    if np.isfinite(_v2) and np.isfinite(_ym2):
                        _all_mv2.append(_v2)
                        _all_yv2.append(_ym2)
                        _lang_pts[_rel2] = {"x": float(_v2), "y": float(_ym2)}
                _json_langs[_lang2] = {
                    "layer": int(_best_L2) if _best_L2 is not None else None,
                    "label": _lbl2,
                    "within_r": _within_r,
                    "color_index": list(valid_langs).index(_lang2) if _lang2 in valid_langs else 0,
                    "relations": _lang_pts,
                }

            _mv2_a = np.array(_all_mv2, dtype=np.float64)
            _yv2_a = np.array(_all_yv2, dtype=np.float64)
            _fin2 = np.isfinite(_mv2_a) & np.isfinite(_yv2_a)
            _overall_r, _overall_p, _overall_n = None, None, int(_fin2.sum())
            if _fin2.sum() >= 3 and np.std(_mv2_a[_fin2]) > 0 and np.std(_yv2_a[_fin2]) > 0:
                _r2, _p2 = _pr_e2(_mv2_a[_fin2], _yv2_a[_fin2])
                _overall_r, _overall_p = float(_r2), float(_p2)

            _title2 = (
                f"[Exp 2] relation subspace vs W_L — {_metric2}\n"
                f"({version_label}, test Oscar)"
            )

            # Save JSON, then render directly in the paper style (ported from
            # replot_multilingual_facts.ipynb) — one scatter plot per language.
            _jd2 = {
                "type": "exp2_scatter",
                "title": _title2,
                "x_label": _x_label2,
                "y_label": "Mean y per relation (test Oscar W_L)",
                "metric": _metric2,
                "version_label": version_label,
                "overall_r": _overall_r,
                "overall_p": _overall_p,
                "overall_n": _overall_n,
                "langs": _json_langs,
            }
            _jp2 = os.path.join(plot_root, f"exp2_scatter_{_metric2}{file_suffix}.json")
            with open(_jp2, "w") as _jf2:
                json.dump(_jd2, _jf2, indent=2)
            print("Saved:", _jp2)

            try:
                with plt.rc_context(PAPER_RCPARAMS):
                    replot_exp2_scatter_from_json(_jp2, save_dir=plot_root, show=False)
            except Exception as _pp_exp2_err:
                print(f"[warn] could not render paper-style Exp2 scatter for {_metric2}: {_pp_exp2_err}")


    for _sv_wm in ["sv", "sv_squared", "none"]:
        _sv_wm_is_first = (_sv_wm == "sv")

        # Version 1: dev PP-selected layers from Exp 1 (only available when DO_EXP1 ran)
        if en_fact_dev_best_layers_exp1 is not None:
            _exp2_rel_metrics = _exp2_build_rel_metrics(en_fact_dev_best_layers_exp1, sv_weight_mode=_sv_wm)
            if _sv_wm_is_first:
                _exp2_plot(_exp2_rel_metrics, en_fact_dev_best_layers_exp1, exp2_plot_root,
                           "dev PP-selected layer per lang",
                           only_metrics_set={m for m in ACTIVE_METRICS if m not in _EXP2_WEIGHTED_METRICS})
            _exp2_plot(_exp2_rel_metrics, en_fact_dev_best_layers_exp1, exp2_plot_root,
                       "dev PP-selected layer per lang",
                       only_metrics_set=_EXP2_WEIGHTED_METRICS,
                       file_suffix=f"_svwt_{_sv_wm}")

        # Compute dev subspace metrics at every layer for every language
        _exp2_dev_rel_metrics = {}  # layer -> lang -> {rel: metrics}
        for _layer2 in LAYERS_TO_LOOP:
            _exp2_dev_rel_metrics[_layer2] = {}
            if _exp2_use_center_oscar:
                _grand_mean_dev_L2 = _oscar_global_mean_cached(
                    _layer2, _all_full_langs_exp2,
                    oscar_resids_root, oscar_cache_root, max_oscar_rows, verbose=False,
                )
            _dev_loaded = {}  # lang -> (pack, W_L_ortho, mu_L_arr, sv_WL)
            for _lang2 in valid_langs:
                _pack2 = _exp2_dev_outs[_layer2].get(_lang2)
                if _pack2 is None:
                    continue
                _full2 = abbr_to_full_LANGUAGE_CODE_MAP.get(_lang2, _lang2)
                if _exp2_use_center_oscar:
                    _WL2_d, _muL2_d, _svWL2_d = oscar_W_global_centered_cached(
                        lang=_full2, layer=_layer2,
                        subspace_method=_sm_L2, var_prop=_vp_L2,
                        oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                        grand_mean=_grand_mean_dev_L2,
                        per_lang_center=_exp2_use_center_oscar_per_lang,
                        max_rows=max_oscar_rows, verbose=False,
                    )
                else:
                    _WL2_d, _muL2_d, _svWL2_d = oscar_W_cached(
                        lang=_full2, layer=_layer2,
                        subspace_method=_sm_L2, var_prop=_vp_L2,
                        oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                        max_rows=max_oscar_rows, verbose=False, oscar_split='dev',
                    )
                if _WL2_d is None:
                    continue
                _WL2_d_ortho = _orthonormalize(np.asarray(_WL2_d, dtype=np.float64))
                _muL2_d_arr = None if _muL2_d is None else np.asarray(_muL2_d, dtype=np.float64)
                _svWL2_d_arr = None if _svWL2_d is None else np.asarray(_svWL2_d, dtype=np.float64)
                _dev_loaded[_lang2] = (_pack2, _WL2_d_ortho, _muL2_d_arr, _svWL2_d_arr)
            for _lang2, (_pack2, _WL2_d_ortho, _muL2_d_arr, _svWL2_d_arr) in _dev_loaded.items():
                _exp2_dev_rel_metrics[_layer2][_lang2] = compute_relation_subspace_metrics_for_layer(
                    _pack2, _WL2_d_ortho, _muL2_d_arr,
                    var_prop_en=DATASET_VAR_PROP, center=SUBSPACE_CENTERED,
                    min_rel_n=MIN_EXAMPLES,
                    sv_WL=_svWL2_d_arr, sv_weight_mode=_sv_wm,
                )


        # For each metric × language, select the layer:
        #   coherence metrics: highest mean score (values already stored as 1-|coh|)
        #   other metrics: highest dev Pearson r with y_mean
        _exp2_corr_best_layers = {}  # metric -> {lang -> best_layer}
        for _metric2 in ACTIVE_METRICS:
            _exp2_corr_best_layers[_metric2] = {}
            for _lang2 in valid_langs:
                _best_score2, _best_layer2 = -np.inf, None
                for _layer2 in LAYERS_TO_LOOP:
                    _reld = _exp2_dev_rel_metrics.get(_layer2, {}).get(_lang2)
                    if _reld is None:
                        continue
                    _px = np.array([rd.get(_metric2, np.nan) for rd in _reld.values()], dtype=np.float64)
                    if _metric2 in _COHERENCE_METRICS:
                        _pxf = np.isfinite(_px)
                        if _pxf.sum() < 1:
                            continue
                        _layer_score = float(np.mean(_px[_pxf]))
                    else:
                        _py = np.array([rd.get("y_mean", np.nan) for rd in _reld.values()], dtype=np.float64)
                        _pf = np.isfinite(_px) & np.isfinite(_py)
                        if _pf.sum() < 3 or np.std(_px[_pf]) < 1e-12 or np.std(_py[_pf]) < 1e-12:
                            continue
                        _layer_score, _ = _pr_e2(_px[_pf], _py[_pf])
                    if _layer_score > _best_score2:
                        _best_score2, _best_layer2 = _layer_score, _layer2
                if _best_layer2 is not None:
                    _exp2_corr_best_layers[_metric2][_lang2] = _best_layer2
    
        # Plot: per metric, use the layer that maximised dev correlation for that metric
        for _metric2 in ACTIVE_METRICS:
            _is_weighted_m2 = _metric2 in _EXP2_WEIGHTED_METRICS
            if not _is_weighted_m2 and not _sv_wm_is_first:
                continue
            _layer_src_v2 = _exp2_corr_best_layers.get(_metric2, {})
            _file_suffix_v2 = f"_svwt_{_sv_wm}" if _is_weighted_m2 else ""

            # Dev set plot at dev-selected layer (before test set plot)
            _dev_rel_metrics_v2 = {
                _lang2_d: _exp2_dev_rel_metrics[_best_l_d][_lang2_d]
                for _lang2_d in valid_langs
                if (_best_l_d := _layer_src_v2.get(_lang2_d)) is not None
                and _best_l_d in _exp2_dev_rel_metrics
                and _lang2_d in _exp2_dev_rel_metrics[_best_l_d]
            }
            _exp2_plot(_dev_rel_metrics_v2, _layer_src_v2, exp2_plot_root_v2_dev,
                       "dev corr-selected layer per lang (dev set)", only_metric=_metric2,
                       file_suffix=_file_suffix_v2)

            _rel_metrics_v2 = _exp2_build_rel_metrics(_layer_src_v2, sv_weight_mode=_sv_wm)
            _exp2_plot(_rel_metrics_v2, _layer_src_v2, exp2_plot_root_v2,
                       "dev corr-selected layer per lang", only_metric=_metric2,
                       file_suffix=_file_suffix_v2)

        del _exp2_dev_rel_metrics

        print("[Exp 2] done")

    del _exp2_test_outs, _exp2_dev_outs


# ═══════════════════════════════════════════════════════════════════════════
# EXP 2 — Second part: sweep W_L var_prop × W_rel var_prop × layer
#          For each (metric, language), the (lang_vp, rel_vp, layer) combo with
#          the highest dev Pearson r (or mean score for coherence metrics) is
#          selected, then evaluated on test Oscar W_L.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 2 sweep_var_and_layer] Sweeping W_L var × W_rel var × layer")

_exp2sw_ok = (
    DO_EXP2_SWEEP
    and base_or_random_mode == 'en_fact'
    and isinstance(lang_mode, (tuple, list)) and len(lang_mode) >= 3
)
if not _exp2sw_ok:
    print("[Exp 2 sweep_var_and_layer] Skipping: DO_EXP2_SWEEP=False or lang_mode not a subspace tuple")
else:
    from scipy.stats import pearsonr as _pr_e2sw

    _e2sw_mode_base        = str(lang_mode[0]).lower()
    _e2sw_sm               = str(lang_mode[1])
    _e2sw_use_center       = _e2sw_mode_base in (
        "center_oscar_and_uncentered_language_subspace",
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )
    _e2sw_use_center_per_lang = _e2sw_mode_base in (
        "center_oscar_and_language_subspace",
        "center_oscar_and_language_subspace_meanshifted",
    )
    _e2sw_all_full_langs = list(abbr_to_full_LANGUAGE_CODE_MAP.values())

    exp2_sweep_plot_root = os.path.join(EXP_ROOT_DIR, "exp2_sweep_var_and_layer")
    os.makedirs(exp2_sweep_plot_root, exist_ok=True)

    # ── Step 1: Compute dev packs once per layer ──────────────────────────────
    # pack["X_base"] = raw activations; independent of lang_vp / rel_vp,
    # so we compute packs once and reuse them across the full (lv × rv) sweep.
    print("  [exp2 sweep] computing dev packs (X_base is var_prop-independent)...")
    _e2sw_dev_outs = {}
    for _layer_sw in LAYERS_TO_LOOP:
        _e2sw_dev_outs[_layer_sw] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=_layer_sw, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=MIN_EXAMPLES, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode='en_fact',
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', sv_weight_mode="sv",
        )

    # ── Step 2: Dev sweep — track best (rel_vp, layer) per (metric, lang, lang_vp) ──
    # After the full sweep we pick a single global lang_vp by win count across
    # languages (aggregated over metrics), then re-derive _e2sw_best from that vp.
    _e2sw_best_score_per_lv = {}  # (metric, lang, lang_vp) -> float
    _e2sw_best_per_lv       = {}  # (metric, lang, lang_vp) -> (rel_vp, layer)

    _e2sw_gm_dev = {}       # layer -> grand_mean (center_oscar mode)

    for _layer_sw in LAYERS_TO_LOOP:
        if _e2sw_use_center and _layer_sw not in _e2sw_gm_dev:
            _e2sw_gm_dev[_layer_sw] = _oscar_global_mean_cached(
                _layer_sw, _e2sw_all_full_langs,
                oscar_resids_root, oscar_cache_root, max_oscar_rows, verbose=False,
            )

        for _e2sw_lv in EXP1_SWEEP_VAR_PROPS:
            # Load W_L (dev) for every lang at this (layer, lang_vp)
            _e2sw_WL_dev = {}   # lang -> (WL_ortho, muL_arr, svWL_arr)
            for _lang_sw in valid_langs:
                if _e2sw_dev_outs[_layer_sw].get(_lang_sw) is None:
                    continue
                _full_sw = abbr_to_full_LANGUAGE_CODE_MAP.get(_lang_sw, _lang_sw)
                if _e2sw_use_center:
                    _WL_d, _muL_d, _svWL_d = oscar_W_global_centered_cached(
                        lang=_full_sw, layer=_layer_sw,
                        subspace_method=_e2sw_sm, var_prop=_e2sw_lv,
                        oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                        grand_mean=_e2sw_gm_dev[_layer_sw],
                        per_lang_center=_e2sw_use_center_per_lang,
                        max_rows=max_oscar_rows, verbose=False,
                    )
                else:
                    _WL_d, _muL_d, _svWL_d = oscar_W_cached(
                        lang=_full_sw, layer=_layer_sw,
                        subspace_method=_e2sw_sm, var_prop=_e2sw_lv,
                        oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                        max_rows=max_oscar_rows, verbose=False, oscar_split='dev',
                    )
                if _WL_d is None:
                    continue
                _e2sw_WL_dev[_lang_sw] = (
                    _orthonormalize(np.asarray(_WL_d, dtype=np.float64)),
                    None if _muL_d  is None else np.asarray(_muL_d,  dtype=np.float64),
                    None if _svWL_d is None else np.asarray(_svWL_d, dtype=np.float64),
                )

            for _e2sw_rv in EXP1_SWEEP_VAR_PROPS:
                for _lang_sw, (_WLo, _muLa, _svWLa) in _e2sw_WL_dev.items():
                    _pack_sw = _e2sw_dev_outs[_layer_sw].get(_lang_sw)
                    if _pack_sw is None:
                        continue
                    _reld = compute_relation_subspace_metrics_for_layer(
                        _pack_sw, _WLo, _muLa,
                        var_prop_en=_e2sw_rv, center=SUBSPACE_CENTERED,
                        min_rel_n=MIN_EXAMPLES,
                        sv_WL=_svWLa, sv_weight_mode="sv",
                    )
                    if not _reld:
                        continue
                    for _metric_sw in ACTIVE_METRICS:
                        _px = np.array([rd.get(_metric_sw, np.nan) for rd in _reld.values()], dtype=np.float64)
                        if _metric_sw in _COHERENCE_METRICS:
                            _pxf = np.isfinite(_px)
                            if _pxf.sum() < 1:
                                continue
                            _score = float(np.mean(_px[_pxf]))
                        else:
                            _py = np.array([rd.get("y_mean", np.nan) for rd in _reld.values()], dtype=np.float64)
                            _pf = np.isfinite(_px) & np.isfinite(_py)
                            if _pf.sum() < 3 or np.std(_px[_pf]) < 1e-12 or np.std(_py[_pf]) < 1e-12:
                                continue
                            _score, _ = _pr_e2sw(_px[_pf], _py[_pf])
                        _key_lv = (_metric_sw, _lang_sw, _e2sw_lv)
                        if _score > _e2sw_best_score_per_lv.get(_key_lv, -np.inf):
                            _e2sw_best_score_per_lv[_key_lv] = float(_score)
                            _e2sw_best_per_lv[_key_lv] = (_e2sw_rv, _layer_sw)

        print(f"  [exp2 sweep dev] layer {_layer_sw} done")

    del _e2sw_dev_outs

    # ── Step 2b: Win count — pick single global lang_vp ───────────────────────
    # For each (metric, lang), find its best lang_vp; count wins per lang_vp
    # aggregated across all metrics; tiebreak by sum of best scores.
    _e2sw_lv_wins    = {lv: 0   for lv in EXP1_SWEEP_VAR_PROPS}
    _e2sw_lv_score_sum = {lv: 0.0 for lv in EXP1_SWEEP_VAR_PROPS}
    for _lang_sw in valid_langs:
        for _metric_sw in ACTIVE_METRICS:
            _scores_for_lv = {
                lv: _e2sw_best_score_per_lv[(_metric_sw, _lang_sw, lv)]
                for lv in EXP1_SWEEP_VAR_PROPS
                if (_metric_sw, _lang_sw, lv) in _e2sw_best_score_per_lv
            }
            if not _scores_for_lv:
                continue
            _winner_lv = max(_scores_for_lv, key=_scores_for_lv.get)
            _e2sw_lv_wins[_winner_lv] += 1
            for lv, sc in _scores_for_lv.items():
                _e2sw_lv_score_sum[lv] += sc

    _e2sw_best_global_lv = max(
        EXP1_SWEEP_VAR_PROPS,
        key=lambda lv: (_e2sw_lv_wins[lv], _e2sw_lv_score_sum[lv]),
    )
    print(f"  [exp2 sweep] Win counts per lang_vp: { {lv: _e2sw_lv_wins[lv] for lv in EXP1_SWEEP_VAR_PROPS} }")
    print(f"  [exp2 sweep] Score sums per lang_vp: { {lv: round(_e2sw_lv_score_sum[lv], 4) for lv in EXP1_SWEEP_VAR_PROPS} }")
    print(f"  [exp2 sweep] Selected global lang_vp: {_e2sw_best_global_lv} ({_e2sw_lv_wins[_e2sw_best_global_lv]} wins)")

    # Re-derive _e2sw_best fixing lang_vp = _e2sw_best_global_lv
    _e2sw_best = {}  # (metric, lang) -> (lang_vp, rel_vp, layer)
    for _lang_sw in valid_langs:
        for _metric_sw in ACTIVE_METRICS:
            _key_lv = (_metric_sw, _lang_sw, _e2sw_best_global_lv)
            if _key_lv in _e2sw_best_per_lv:
                _rv_sel, _ly_sel = _e2sw_best_per_lv[_key_lv]
                _e2sw_best[(_metric_sw, _lang_sw)] = (_e2sw_best_global_lv, _rv_sel, _ly_sel)

    print(f"  [exp2 sweep] selections for {len(_e2sw_best)} (metric, lang) pairs")

    # ── Step 3: Test phase — compute packs + W_L for unique (lang_vp, layer) ─
    _e2sw_unique_lv_layer = sorted(set(
        (_lv, _ly) for (_lv, _rv, _ly) in _e2sw_best.values()
    ))
    print(f"  [exp2 sweep] {len(_e2sw_unique_lv_layer)} unique (lang_vp, layer) for test")

    _e2sw_test_outs = {}    # (lang_vp, layer) -> {lang: pack}
    _e2sw_gm_test   = {}    # layer -> grand_mean

    for _e2sw_lv, _layer_sw in _e2sw_unique_lv_layer:
        _sw_lm_t = (lang_mode[0], _e2sw_sm, _e2sw_lv)
        print(f"  [exp2 sweep test packs] lang_vp={_e2sw_lv}, layer={_layer_sw}")
        _e2sw_test_outs[(_e2sw_lv, _layer_sw)] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=_layer_sw, rep_kind=rep_kind,
            lang_mode=_sw_lm_t, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=MIN_EXAMPLES, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode='en_fact',
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', sv_weight_mode="sv",
        )
        if _e2sw_use_center and _layer_sw not in _e2sw_gm_test:
            _e2sw_gm_test[_layer_sw] = _oscar_global_mean_cached(
                _layer_sw, _e2sw_all_full_langs,
                oscar_resids_root, oscar_cache_root, max_oscar_rows, verbose=False,
            )

    # ── Step 4: Compute test relation metrics for selected (lv, rv, layer) ───
    _e2sw_unique_triples = sorted(set(_e2sw_best.values()))   # (lv, rv, layer)
    _e2sw_test_rel = {}   # (lv, rv, layer) -> lang -> {rel: metrics}

    for _e2sw_lv, _e2sw_rv, _layer_sw in _e2sw_unique_triples:
        _key_t = (_e2sw_lv, _e2sw_rv, _layer_sw)
        _e2sw_test_rel[_key_t] = {}
        _packs_t = _e2sw_test_outs.get((_e2sw_lv, _layer_sw), {})
        for _lang_sw in valid_langs:
            _pack_t = _packs_t.get(_lang_sw)
            if _pack_t is None:
                continue
            _full_sw = abbr_to_full_LANGUAGE_CODE_MAP.get(_lang_sw, _lang_sw)
            if _e2sw_use_center:
                _WLt, _muLt, _svWLt = oscar_W_global_centered_cached(
                    lang=_full_sw, layer=_layer_sw,
                    subspace_method=_e2sw_sm, var_prop=_e2sw_lv,
                    oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                    grand_mean=_e2sw_gm_test[_layer_sw],
                    per_lang_center=_e2sw_use_center_per_lang,
                    max_rows=max_oscar_rows, verbose=False,
                )
            else:
                _WLt, _muLt, _svWLt = oscar_W_cached(
                    lang=_full_sw, layer=_layer_sw,
                    subspace_method=_e2sw_sm, var_prop=_e2sw_lv,
                    oscar_resids_root=oscar_resids_root, disk_cache_root=oscar_cache_root,
                    max_rows=max_oscar_rows, verbose=False, oscar_split='test',
                )
            if _WLt is None:
                continue
            _e2sw_test_rel[_key_t][_lang_sw] = compute_relation_subspace_metrics_for_layer(
                _pack_t,
                _orthonormalize(np.asarray(_WLt, dtype=np.float64)),
                None if _muLt  is None else np.asarray(_muLt,  dtype=np.float64),
                var_prop_en=_e2sw_rv, center=SUBSPACE_CENTERED,
                min_rel_n=MIN_EXAMPLES,
                sv_WL=None if _svWLt is None else np.asarray(_svWLt, dtype=np.float64),
                sv_weight_mode="sv",
            )

    # ── Step 5: Scatter plot for every metric ─────────────────────────────────
    _e2sw_cmap = plt.cm.tab10
    _e2sw_lc   = {L: _e2sw_cmap(i / max(len(valid_langs), 1)) for i, L in enumerate(valid_langs)}

    for _metric_sw in ACTIVE_METRICS:
        _x_label_sw = METRIC_LABELS.get(_metric_sw, _metric_sw)
        fig_sw, ax_sw = plt.subplots(figsize=(8, 5))
        _all_mx_sw, _all_my_sw = [], []
        _seen_lbl_sw = set()
        _json_langs_sw = {}

        for _lang_sw in valid_langs:
            _combo_sw = _e2sw_best.get((_metric_sw, _lang_sw))
            if _combo_sw is None:
                continue
            _lv_sw, _rv_sw, _ly_sw = _combo_sw
            _reld_t = _e2sw_test_rel.get(_combo_sw, {}).get(_lang_sw)
            if not _reld_t:
                continue

            _lx_sw  = np.array([rd.get(_metric_sw, np.nan) for rd in _reld_t.values()], dtype=np.float64)
            _ly2_sw = np.array([rd.get("y_mean",    np.nan) for rd in _reld_t.values()], dtype=np.float64)
            _lfin   = np.isfinite(_lx_sw) & np.isfinite(_ly2_sw)
            _within_r_sw = None
            _lbl_sw = f"{_lang_sw}(L{_ly_sw},lv={_lv_sw},rv={_rv_sw})"
            if _lfin.sum() >= 3 and np.std(_lx_sw[_lfin]) > 0 and np.std(_ly2_sw[_lfin]) > 0:
                _lr_sw, _ = _pr_e2sw(_lx_sw[_lfin], _ly2_sw[_lfin])
                _within_r_sw = float(_lr_sw)
                _lbl_sw = f"{_lang_sw}(L{_ly_sw},lv={_lv_sw},rv={_rv_sw}) r={_lr_sw:.2f}"

            _lang_pts_sw = {}
            for _rel_sw, _rd_sw in _reld_t.items():
                _v_sw  = _rd_sw.get(_metric_sw, np.nan)
                _ym_sw = _rd_sw.get("y_mean",    np.nan)
                if np.isfinite(_v_sw) and np.isfinite(_ym_sw):
                    _kw_sw = {"label": _lbl_sw} if _lbl_sw not in _seen_lbl_sw else {}
                    _seen_lbl_sw.add(_lbl_sw)
                    ax_sw.scatter(_v_sw, _ym_sw, color=_e2sw_lc.get(_lang_sw, "gray"),
                                  s=22, alpha=0.65, linewidths=0, **_kw_sw)
                    _all_mx_sw.append(_v_sw)
                    _all_my_sw.append(_ym_sw)
                    _lang_pts_sw[_rel_sw] = {"x": float(_v_sw), "y": float(_ym_sw)}
            _json_langs_sw[_lang_sw] = {
                "layer": int(_ly_sw), "lang_var_prop": float(_lv_sw),
                "rel_var_prop": float(_rv_sw), "label": _lbl_sw,
                "within_r": _within_r_sw, "relations": _lang_pts_sw,
            }

        _mx_a = np.array(_all_mx_sw, dtype=np.float64)
        _my_a = np.array(_all_my_sw, dtype=np.float64)
        _fin_sw = np.isfinite(_mx_a) & np.isfinite(_my_a)
        _or_sw, _op_sw, _on_sw = None, None, int(_fin_sw.sum())
        if _fin_sw.sum() >= 3 and np.std(_mx_a[_fin_sw]) > 0 and np.std(_my_a[_fin_sw]) > 0:
            _or_sw, _op_sw = _pr_e2sw(_mx_a[_fin_sw], _my_a[_fin_sw])
            _or_sw, _op_sw = float(_or_sw), float(_op_sw)
            _ps_sw = "p<1e-3" if _op_sw < 1e-3 else f"p={_op_sw:.2g}"
            ax_sw.text(0.03, 0.97, f"r={_or_sw:.3f}, {_ps_sw} (n={_on_sw})",
                       transform=ax_sw.transAxes, va="top", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        _title_sw = (
            f"[Exp2 sweep_var_and_layer] {_metric_sw}\n"
            f"sweep-selected (lang_var, rel_var, layer) per lang — test Oscar"
        )
        ax_sw.set_xlabel(_x_label_sw)
        ax_sw.set_ylabel("Mean y per relation (test Oscar W_L)")
        ax_sw.set_title(_title_sw)
        ax_sw.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax_sw.grid(alpha=0.3)
        fig_sw.tight_layout()
        _sp_sw = os.path.join(exp2_sweep_plot_root, f"exp2_sweep_scatter_{_metric_sw}.svg")
        fig_sw.savefig(_sp_sw, bbox_inches="tight", format="svg")
        print("Saved:", _sp_sw)
        plt.close(fig_sw)

        _jd_sw = {
            "type": "exp2_sweep_scatter", "title": _title_sw,
            "x_label": _x_label_sw, "metric": _metric_sw,
            "overall_r": _or_sw, "overall_p": _op_sw, "overall_n": _on_sw,
            "langs": _json_langs_sw,
        }
        _jp_sw = os.path.join(exp2_sweep_plot_root, f"exp2_sweep_scatter_{_metric_sw}.json")
        with open(_jp_sw, "w") as _jf_sw:
            json.dump(_jd_sw, _jf_sw, indent=2)

    # ── Step 6: Save best (lang_vp, rel_vp, layer) selection table ───────────
    _e2sw_sel_table = {}   # metric -> lang -> {lang_var_prop, rel_var_prop, layer}
    for (_metric_sw, _lang_sw), (_lv_sw, _rv_sw, _ly_sw) in _e2sw_best.items():
        _e2sw_sel_table.setdefault(_metric_sw, {})[_lang_sw] = {
            "lang_var_prop": float(_lv_sw),
            "rel_var_prop":  float(_rv_sw),
            "layer":         int(_ly_sw),
        }
    _e2sw_sel_json = os.path.join(exp2_sweep_plot_root, "exp2_sweep_best_combo.json")
    with open(_e2sw_sel_json, "w") as _f:
        json.dump(_e2sw_sel_table, _f, indent=2)
    print("[Exp2 sweep] Saved best combo table:", _e2sw_sel_json)

    del _e2sw_test_outs, _e2sw_test_rel
    print("[Exp 2 sweep_var_and_layer] done")

# ═══════════════════════════════════════════════════════════════════════════
# EXP 3 — En-fact vector vs mu_L (mean of language subspace), dev/test split
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 3] mu_L mean-vector feature with dev/test Oscar split")

lang_mode_muL = ('language_subspace_mean_vector', 'SVD', 0.99)
lang_mode_muL_name = "_".join(str(x) for x in lang_mode_muL)
feature_muL = 'vec_vec_orthogonality'
exp3_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature_muL}_lowest_feature_first", "exp3_muL_dev_test_split")

pr_auc_tables_exp3 = {}

for base_or_random_mode in (['en_fact'] if DO_EXP3 else []):
    print(f"[Exp 3] mode={base_or_random_mode}")

    # DEV phase
    outs_exp3_dev_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp3_dev_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_muL, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_muL, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', store_X_base=False,
        )

    layers_run_exp3 = sorted(outs_exp3_dev_by_layer.keys())
    if not layers_run_exp3:
        print(f"[Exp 3] no dev layers for mode={base_or_random_mode}")
        continue

    tt_exp3_dev = tt_select_best_layer_with_split(
        outs_exp3_dev_by_layer, valid_langs=valid_langs,
        layers=layers_run_exp3, seed=0, y_layer="last",
        min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
    )
    exp3_dev_best_layers = {
        lang: int(info["best_layer"])
        for lang, info in tt_exp3_dev.items()
        if info.get("best_layer") is not None
    }

    # TEST phase
    outs_exp3_test_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp3_test_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_muL, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_muL, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', store_X_base=False,
        )

    y_acc_by_lang_exp3 = None
    if y_mode == "accuracy":
        try:
            y_acc_by_lang_exp3 = build_y_acc_by_lang_from_predictions_jsonl(path, tt_exp3_dev)
        except Exception as e:
            print(f"[Exp 3] warn: could not build y_acc: {e}")

    os.makedirs(exp3_plot_root, exist_ok=True)
    _exp3_lang_colors = {L: plt.cm.tab10(i / max(len(valid_langs), 1)) for i, L in enumerate(valid_langs)}

    plot_cumulative_all_layers_all_langs(
        outs_exp3_test_by_layer, tt_exp3_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp3, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="cumulative",
        title=f"[Exp3] {base_or_random_mode}: dev-selected layer, test mu_L feature",
        save_path=os.path.join(exp3_plot_root, f"exp3_{base_or_random_mode}_cumulative.svg"),
        save_data_path=os.path.join(exp3_plot_root, f"exp3_{base_or_random_mode}_cumulative.json"),
        calculate_auc=True, show=False,
        fixed_layers_to_plot=exp3_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp3_lang_colors,
    )
    plot_cumulative_all_layers_all_langs(
        outs_exp3_test_by_layer, tt_exp3_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp3, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="non_cumulative",
        title=f"[Exp3] {base_or_random_mode}: dev-selected layer, test mu_L (non-cumul)",
        save_path=os.path.join(exp3_plot_root, f"exp3_{base_or_random_mode}_noncumulative.svg"),
        save_data_path=os.path.join(exp3_plot_root, f"exp3_{base_or_random_mode}_noncumulative.json"),
        calculate_auc=False, show=False,
        fixed_layers_to_plot=exp3_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp3_lang_colors,
    )
    # ── PR-AUC (failure prediction) at dev-selected layer ────────────────────
    if y_acc_by_lang_exp3 is not None:
        _exp3_pr_dir = os.path.join(exp3_plot_root, f"pr_auc_{base_or_random_mode}_devselected")
        _exp3_pr_summary = _compute_and_save_pr_auc_all_langs(
            outs_exp3_test_by_layer, tt_exp3_dev, valid_langs,
            best_layers=exp3_dev_best_layers,
            y_acc_by_lang=y_acc_by_lang_exp3,
            save_dir=_exp3_pr_dir,
            label=f"Exp3 {base_or_random_mode} dev-sel",
        )
        pr_auc_tables_exp3[base_or_random_mode] = _exp3_pr_summary

    del outs_exp3_dev_by_layer, outs_exp3_test_by_layer
    print(f"[Exp 3] done mode={base_or_random_mode}")


# ── Exp 3 PR-AUC table ───────────────────────────────────────────────────────
if pr_auc_tables_exp3:
    _pr3_dir = os.path.join(exp3_plot_root, "pr_auc_tables")
    os.makedirs(_pr3_dir, exist_ok=True)

    # Full JSON — keeps precision/recall arrays for replotting
    _pr3_json_path = os.path.join(_pr3_dir, "pr_auc_compare_devselected.json")
    with open(_pr3_json_path, "w") as _f:
        json.dump(pr_auc_tables_exp3, _f, indent=2)
    print(f"[Exp 3] saved PR-AUC summary JSON: {_pr3_json_path}")

    # CSV — rows=langs, cols=modes, for heatmap
    _pr3_df = pd.DataFrame(index=sorted(valid_langs))
    for _mode, _lang_stats in pr_auc_tables_exp3.items():
        _pr3_df[_mode] = pd.Series({lang: v["pr_auc"] for lang, v in _lang_stats.items()})
    _pr3_csv_path = os.path.join(_pr3_dir, "pr_auc_compare_devselected.csv")
    _pr3_df.to_csv(_pr3_csv_path, index_label="language")
    print(f"[Exp 3] saved PR-AUC summary CSV:  {_pr3_csv_path}")

    try:
        _pr3_raw_df, _pr3_en_fact_series = load_auc_matrix(_pr3_csv_path, valid_langs=valid_langs)
        _pr3_base = _pr3_csv_path.replace(".csv", "")
        plot_auc_heatmap(
            _pr3_raw_df, _pr3_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr3_base + "_heatmap.svg",
            title=os.path.basename(_pr3_csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            _pr3_raw_df, _pr3_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr3_base + "_barplot.svg",
            title=os.path.basename(_pr3_csv_path),
        )
    except Exception as _pr3_err:
        print(f"[warn] could not plot Exp3 PR-AUC heatmap: {_pr3_err}")


# ═══════════════════════════════════════════════════════════════════════════
# EXP 3b — En-fact vector vs W_L subspace (ortho_min per example), dev/test split
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 3b] W_L subspace ortho_min feature with dev/test Oscar split")

lang_mode_WL = ('language_subspace', 'SVD', 0.99)
feature_exp3b = 'vec_subspace_angle_by_cumulative_coherence'
exp3b_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature_exp3b}_lowest_feature_first", "exp3b_orthomin_dev_test_split")

pr_auc_tables_exp3b = {}

for base_or_random_mode in (['en_fact']  if DO_EXP3B else []):
    print(f"[Exp 3b] mode={base_or_random_mode}")

    # DEV phase
    outs_exp3b_dev_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp3b_dev_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_WL, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_exp3b, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', store_X_base=False,
        )

    layers_run_exp3b = sorted(outs_exp3b_dev_by_layer.keys())
    if not layers_run_exp3b:
        print(f"[Exp 3b] no dev layers for mode={base_or_random_mode}")
        continue

    tt_exp3b_dev = tt_select_best_layer_with_split(
        outs_exp3b_dev_by_layer, valid_langs=valid_langs,
        layers=layers_run_exp3b, seed=0, y_layer="last",
        min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
    )
    exp3b_dev_best_layers = {
        lang: int(info["best_layer"])
        for lang, info in tt_exp3b_dev.items()
        if info.get("best_layer") is not None
    }

    # TEST phase
    outs_exp3b_test_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp3b_test_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_WL, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_exp3b, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test', store_X_base=False,
        )

    y_acc_by_lang_exp3b = None
    if y_mode == "accuracy":
        try:
            y_acc_by_lang_exp3b = build_y_acc_by_lang_from_predictions_jsonl(path, tt_exp3b_dev)
        except Exception as e:
            print(f"[Exp 3b] warn: could not build y_acc: {e}")

    os.makedirs(exp3b_plot_root, exist_ok=True)
    _exp3b_lang_colors = {L: plt.cm.tab10(i / max(len(valid_langs), 1)) for i, L in enumerate(valid_langs)}

    plot_cumulative_all_layers_all_langs(
        outs_exp3b_test_by_layer, tt_exp3b_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp3b, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="cumulative",
        title=f"[Exp3b] {base_or_random_mode}: dev-selected layer, test ortho_min (W_L)",
        save_path=os.path.join(exp3b_plot_root, f"exp3b_{base_or_random_mode}_cumulative.svg"),
        save_data_path=os.path.join(exp3b_plot_root, f"exp3b_{base_or_random_mode}_cumulative.json"),
        calculate_auc=True, show=False,
        fixed_layers_to_plot=exp3b_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp3b_lang_colors,
    )
    plot_cumulative_all_layers_all_langs(
        outs_exp3b_test_by_layer, tt_exp3b_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp3b, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="non_cumulative",
        title=f"[Exp3b] {base_or_random_mode}: dev-selected layer, test ortho_min (W_L, non-cumul)",
        save_path=os.path.join(exp3b_plot_root, f"exp3b_{base_or_random_mode}_noncumulative.svg"),
        save_data_path=os.path.join(exp3b_plot_root, f"exp3b_{base_or_random_mode}_noncumulative.json"),
        calculate_auc=False, show=False,
        fixed_layers_to_plot=exp3b_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp3b_lang_colors,
    )
    # ── PR-AUC (failure prediction) at dev-selected layer ────────────────────
    if y_acc_by_lang_exp3b is not None:
        _exp3b_pr_dir = os.path.join(exp3b_plot_root, f"pr_auc_{base_or_random_mode}_devselected")
        _exp3b_pr_summary = _compute_and_save_pr_auc_all_langs(
            outs_exp3b_test_by_layer, tt_exp3b_dev, valid_langs,
            best_layers=exp3b_dev_best_layers,
            y_acc_by_lang=y_acc_by_lang_exp3b,
            save_dir=_exp3b_pr_dir,
            label=f"Exp3b {base_or_random_mode} dev-sel",
        )
        pr_auc_tables_exp3b[base_or_random_mode] = _exp3b_pr_summary

    del outs_exp3b_dev_by_layer, outs_exp3b_test_by_layer
    print(f"[Exp 3b] done mode={base_or_random_mode}")


# ── Exp 3b PR-AUC table ──────────────────────────────────────────────────────
if pr_auc_tables_exp3b:
    _pr3b_dir = os.path.join(exp3b_plot_root, "pr_auc_tables")
    os.makedirs(_pr3b_dir, exist_ok=True)

    # Full JSON — keeps precision/recall arrays for replotting
    _pr3b_json_path = os.path.join(_pr3b_dir, "pr_auc_compare_devselected.json")
    with open(_pr3b_json_path, "w") as _f:
        json.dump(pr_auc_tables_exp3b, _f, indent=2)
    print(f"[Exp 3b] saved PR-AUC summary JSON: {_pr3b_json_path}")

    # CSV — rows=langs, cols=modes, for heatmap
    _pr3b_df = pd.DataFrame(index=sorted(valid_langs))
    for _mode, _lang_stats in pr_auc_tables_exp3b.items():
        _pr3b_df[_mode] = pd.Series({lang: v["pr_auc"] for lang, v in _lang_stats.items()})
    _pr3b_csv_path = os.path.join(_pr3b_dir, "pr_auc_compare_devselected.csv")
    _pr3b_df.to_csv(_pr3b_csv_path, index_label="language")
    print(f"[Exp 3b] saved PR-AUC summary CSV:  {_pr3b_csv_path}")

    try:
        _pr3b_raw_df, _pr3b_en_fact_series = load_auc_matrix(_pr3b_csv_path, valid_langs=valid_langs)
        _pr3b_base = _pr3b_csv_path.replace(".csv", "")
        plot_auc_heatmap(
            _pr3b_raw_df, _pr3b_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr3b_base + "_heatmap.svg",
            title=os.path.basename(_pr3b_csv_path),
        )
        plot_auc_diagsum_and_rowsum_bars(
            _pr3b_raw_df, _pr3b_en_fact_series,
            add_en_fact_row=True,
            save_path=_pr3b_base + "_barplot.svg",
            title=os.path.basename(_pr3b_csv_path),
        )
    except Exception as _pr3b_err:
        print(f"[warn] could not plot Exp3b PR-AUC heatmap: {_pr3b_err}")


# ═══════════════════════════════════════════════════════════════════════════
# EXP 4 — Merge all language datapoints; dev-set layer selection;
#          single cumulative plot with permutation test
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 4] Merged all-language cumulative experiment")
exp4_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature}_lowest_feature_first", "exp4_merged_all_langs")

for base_or_random_mode in (base_or_random_mode_list_exp2plus if DO_EXP4 else []):
    print(f"[Exp 4] mode={base_or_random_mode}")

    # DEV phase: find best layer by lowest merged cumulative AUC
    best_layer_exp4 = None
    best_auc_exp4 = np.inf
    for layer in LAYERS_TO_LOOP:
        layer_outs = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev', store_X_base=False,
        )
        x_parts, y_parts = [], []
        for lang in valid_langs:
            if lang not in layer_outs:
                continue
            pack = layer_outs[lang]
            x_arr = np.asarray(pack["x_ortho"], dtype=np.float64)
            y_arr = np.asarray(pack["y"], dtype=np.float64)
            m = np.isfinite(x_arr) & np.isfinite(y_arr)
            if m.sum() > 0:
                x_parts.append(x_arr[m])
                y_parts.append(y_arr[m])
        del layer_outs
        if not x_parts:
            continue
        x_merged = np.concatenate(x_parts)
        y_merged = np.concatenate(y_parts)
        auc_val = _cumulative_auc_from_sorted_y(y_merged[np.argsort(x_merged)])
        if np.isfinite(auc_val) and auc_val < best_auc_exp4:
            best_auc_exp4 = auc_val
            best_layer_exp4 = layer

    if best_layer_exp4 is None:
        print(f"[Exp 4] could not select best layer for mode={base_or_random_mode}")
        continue
    print(f"[Exp 4] dev best layer={best_layer_exp4} (merged AUC={best_auc_exp4:.4f})")

    # TEST phase: compute at best layer using test Oscar
    test_outs = compute_geometric_feature(
        model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
        valid_langs=valid_langs, layer=best_layer_exp4, rep_kind=rep_kind,
        lang_mode=lang_mode, targets_jsonl_path=TARGETS_JSONL,
        target_kind=target_kind, y_transform=y_transform,
        translation_data_dirs=translation_data_dirs,
        min_n=100, separate_correct_incorrect_examples=False,
        feature=feature, base_or_random_mode=base_or_random_mode,
        random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
        rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
        oscar_split='test', store_X_base=False,
    )
    x_test_parts, y_test_parts = [], []
    for lang in valid_langs:
        if lang not in test_outs:
            continue
        pack = test_outs[lang]
        x_arr = np.asarray(pack["x_ortho"], dtype=np.float64)
        y_arr = np.asarray(pack["y"], dtype=np.float64)
        m = np.isfinite(x_arr) & np.isfinite(y_arr)
        if m.sum() > 0:
            x_test_parts.append(x_arr[m])
            y_test_parts.append(y_arr[m])
    del test_outs

    if not x_test_parts:
        print(f"[Exp 4] no test data for mode={base_or_random_mode}")
        continue
    x_test_merged = np.concatenate(x_test_parts)
    y_test_merged = np.concatenate(y_test_parts)

    os.makedirs(exp4_plot_root, exist_ok=True)
    _plot_exp4_cumulative(
        x_test=x_test_merged, y_test=y_test_merged,
        best_layer=best_layer_exp4,
        title=f"[Exp4] {base_or_random_mode}: all langs merged, dev-selected L{best_layer_exp4}",
        save_path=os.path.join(exp4_plot_root, f"exp4_{base_or_random_mode}_cumulative.svg"),
        save_data_path=os.path.join(exp4_plot_root, f"exp4_{base_or_random_mode}_cumulative.json"),
        quantile_step=quantile_step, y_lim=(0.0, 1.0),
        show=False,
    )
    print(f"[Exp 4] done mode={base_or_random_mode}")


# ═══════════════════════════════════════════════════════════════════════════
# EXP 5 — mean_translation_vector: angle(en_fact_vec, unit(mu_L - mu_ENFACT))
#          Dev Oscar W_L for layer selection; test Oscar W_L for evaluation.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Exp 5] mean_translation_vector: angle(en_fact, unit(mu_L - mu_ENFACT))")

lang_mode_exp5 = ('mean_translation_vector', 'SVD', 0.99)
lang_mode_exp5_name = "_".join(str(x) for x in lang_mode_exp5)
feature_exp5 = 'vec_vec_orthogonality'
exp5_plot_root = os.path.join(EXP_ROOT_DIR, f"{feature_exp5}_lowest_feature_first", "exp5_mean_translation_vector")

for base_or_random_mode in (base_or_random_mode_list_exp2plus if DO_EXP5 else []):
    print(f"[Exp 5] mode={base_or_random_mode}")

    # DEV phase: compute features using dev Oscar mu_L
    outs_exp5_dev_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp5_dev_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_exp5, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_exp5, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='dev',
        )

    layers_run_exp5 = sorted(outs_exp5_dev_by_layer.keys())
    if not layers_run_exp5:
        print(f"[Exp 5] no dev layers for mode={base_or_random_mode}")
        continue

    tt_exp5_dev = tt_select_best_layer_with_split(
        outs_exp5_dev_by_layer, valid_langs=valid_langs,
        layers=layers_run_exp5, seed=0, y_layer="last",
        min_n=100, train_frac=dev_frac, x_bar=x_bar, acc_map=_acc_map,
    )
    exp5_dev_best_layers = {
        lang: int(info["best_layer"])
        for lang, info in tt_exp5_dev.items()
        if info.get("best_layer") is not None
    }

    # TEST phase: compute features using test Oscar mu_L
    outs_exp5_test_by_layer = {}
    for layer in LAYERS_TO_LOOP:
        outs_exp5_test_by_layer[layer] = compute_geometric_feature(
            model=model, run_dir=RUN_DIR, shot_tag=shot_tag,
            valid_langs=valid_langs, layer=layer, rep_kind=rep_kind,
            lang_mode=lang_mode_exp5, targets_jsonl_path=TARGETS_JSONL,
            target_kind=target_kind, y_transform=y_transform,
            translation_data_dirs=translation_data_dirs,
            min_n=100, separate_correct_incorrect_examples=False,
            feature=feature_exp5, base_or_random_mode=base_or_random_mode,
            random_seed=0, mean_center_by_cluster=mean_center_by_cluster,
            rel_map_targets=rel_map_targets, rel_map_preds=rel_map_preds,
            oscar_split='test',
        )

    y_acc_by_lang_exp5 = None
    if y_mode == "accuracy":
        try:
            y_acc_by_lang_exp5 = build_y_acc_by_lang_from_predictions_jsonl(path, tt_exp5_dev)
        except Exception as e:
            print(f"[Exp 5] warn: could not build y_acc: {e}")

    os.makedirs(exp5_plot_root, exist_ok=True)
    _exp5_lang_colors = {L: plt.cm.tab10(i / max(len(valid_langs), 1)) for i, L in enumerate(valid_langs)}

    plot_cumulative_all_layers_all_langs(
        outs_exp5_test_by_layer, tt_exp5_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp5, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="cumulative",
        title=f"[Exp5] {base_or_random_mode}: dev-selected layer, mean_translation_vector",
        save_path=os.path.join(exp5_plot_root, f"exp5_{base_or_random_mode}_cumulative.svg"),
        save_data_path=os.path.join(exp5_plot_root, f"exp5_{base_or_random_mode}_cumulative.json"),
        calculate_auc=True, show=False,
        fixed_layers_to_plot=exp5_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp5_lang_colors,
    )
    plot_cumulative_all_layers_all_langs(
        outs_exp5_test_by_layer, tt_exp5_dev,
        valid_langs=valid_langs, y_mode=y_mode,
        y_acc_by_lang=y_acc_by_lang_exp5, y_layer="last",
        quantile_step=quantile_step, cumulative_mode="non_cumulative",
        title=f"[Exp5] {base_or_random_mode}: dev-selected layer, mean_translation_vector (non-cumul)",
        save_path=os.path.join(exp5_plot_root, f"exp5_{base_or_random_mode}_noncumulative.svg"),
        save_data_path=os.path.join(exp5_plot_root, f"exp5_{base_or_random_mode}_noncumulative.json"),
        calculate_auc=False, show=False,
        fixed_layers_to_plot=exp5_dev_best_layers, y_lim=(0.0, 1.0),
        lang_colors=_exp5_lang_colors,
    )
    del outs_exp5_dev_by_layer, outs_exp5_test_by_layer
    print(f"[Exp 5] done mode={base_or_random_mode}")
