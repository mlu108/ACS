"""
Automatic per-layer clustering of single-hop dataset representations.

Replaces the manual A/B/C dataset groupings hand-written at the top of
multihop_experiments_auto_cluster_center.py. Manual clustering assumed the
same three groups hold at every layer; this module fits clusters
independently per layer (cheap, since clustering ~40 dataset centroids is
fast) and returns/saves a per-layer dataset -> cluster_label mapping plus the
exact per-cluster mean representation.

Clustering granularity: each two-hop example produces two representation
vectors — v1 (the QxFx / "first hop" representation) and v2 (the QFxGFx /
"second hop" representation). These belong to two different *single-hop*
datasets (qx and gfx respectively), given by ALL_TASKS_QxFx_QFxGFx[task] =
(qx, gfx). Clustering (and the mean-centering that uses it) therefore
operates over the flattened set of single-hop dataset names
(ALL_TASKS_QxFx_QFxGFx.values(), not .keys()) — matching the granularity the
manual A/B/C lists already used (e.g. 'antonym', 'mod-twenty' are single-hop
dataset names, not composite task names like 'antonym-french'). v1 and v2
for the same example can land in different clusters, since qx and gfx are
independent datasets.

Typical usage from multihop_experiments_auto_cluster_center.py, after
`runs`, `eligible`, `n_layers`, `d_model`, and `ALL_TASKS_QxFx_QFxGFx` are
already loaded:

    from multihop_experiment import run_auto_clustering

    task_names = sorted({d for pair in ALL_TASKS_QxFx_QFxGFx.values() for d in pair})

    result = run_auto_clustering(
        eligible=eligible,
        runs=runs,
        task_names=task_names,
        component_task_map=ALL_TASKS_QxFx_QFxGFx,
        n_layers=n_layers,
        d_model=d_model,
        auto_cluster_using="train",   # "train" (default, no test leakage) or "all"
        train_ratio=0.1,
        k_range=range(2, 7),
        balance_n=200,
        seed=0,
        debug_mode=True,
        manual_clusters={"A": A_datasets, "B": B_datasets, "C": C_datasets},
        save_path=os.path.join(exp1_plot_root, "auto_clusters.json"),
    )

    task_to_cluster = result["task_to_cluster"]   # {layer: {dataset_name: cluster_label}}
    exact_mu        = result["exact_mu"]          # {cluster_label: [vec_per_layer]}

`task_to_cluster` / `exact_mu` are drop-in compatible with the existing
`_get_cluster_center_for_task` / `_maybe_cluster_center_vecs` helpers in
multihop_experiments_auto_cluster_center.py: those go through
`resolve_cluster_label` (below), which detects the per-layer nested shape of
`task_to_cluster` and falls back to a flat (layer-constant) shape otherwise.
Callers there resolve qx/gfx from a composite task name via
ALL_TASKS_QxFx_QFxGFx before calling resolve_cluster_label, since a lookup
now needs a single-hop dataset name, not a composite task name — see
build_example_subspace_scores / compute_tasklevel_geometry_from_runs.
"""

from __future__ import annotations

import os
import json
from collections import defaultdict

import numpy as np


# ============================================================
# Cluster-label resolution (shared with the main experiment script)
# ============================================================

def resolve_cluster_label(task, layer, task_to_cluster, cluster_default="C"):
    """
    task_to_cluster accepts two shapes:
      - flat (manual, layer-constant clustering): {task_name: cluster_label}
      - nested per layer (auto clustering, as returned by run_auto_clustering):
        {layer: {task_name: cluster_label}}

    Layer keys are always ints and task names are always strings, so checking
    whether `layer` is itself a key unambiguously tells flat and nested dicts
    apart.

    `cluster_default` (e.g. "C") only makes sense for the flat/manual shape,
    where it names a real catch-all cluster that exists in `exact_mu`. For
    the nested/auto shape there is no such catch-all — cluster_layer already
    guarantees every task with at least one example gets assigned a real
    cluster id, so a task missing from the per-layer map means it truly has
    no examples anywhere; this returns None in that case rather than a
    default label that likely isn't a key in the auto-generated exact_mu.
    """
    layer_map = task_to_cluster.get(layer)
    if isinstance(layer_map, dict):
        return layer_map.get(task)
    return task_to_cluster.get(task, cluster_default)


# ============================================================
# Representation collection
# ============================================================

def _dataset_example_vectors(
    eligible, runs, dataset_name, layer, component_task_map, *,
    task_field="task", subset_indices=None,
):
    """
    Collect every representation vector at `layer` whose single-hop dataset
    is `dataset_name`, optionally restricted to `subset_indices` (positions
    into `eligible`, e.g. a train-only subset).

    Each example's composite task maps to (qx, gfx) = component_task_map[task].
    v1 (the QxFx vector) is included if qx == dataset_name; v2 (the QFxGFx
    vector) is included if gfx == dataset_name. A composite task where
    qx == gfx contributes both.
    """
    vecs = []
    idx_iter = subset_indices if subset_indices is not None else range(len(eligible))
    for i in idx_iter:
        ex = eligible[i]
        task = ex.get(task_field)
        pair = component_task_map.get(task)
        if pair is None:
            continue
        qx, gfx = pair
        if qx != dataset_name and gfx != dataset_name:
            continue
        if qx == dataset_name:
            run_idx_1 = ex["run_idx_qx_fx"]
            i1 = ex["idx_qx_fx"]
            v1 = np.asarray(runs[run_idx_1]["resid_last"][i1, layer], dtype=np.float32)
            vecs.append(v1)
        if gfx == dataset_name:
            run_idx_2 = ex["run_idx_qfx_gfx"]
            i2 = ex["idx_qfx_gfx"]
            v2 = np.asarray(runs[run_idx_2]["resid_last"][i2, layer], dtype=np.float32)
            vecs.append(v2)
    return vecs


def _dataset_example_counts(eligible, dataset_names, component_task_map, *, task_field="task", subset_indices=None):
    """
    Number of vectors available per single-hop dataset — same counting rule
    as _dataset_example_vectors (v1 counts if qx == dataset_name, v2 counts
    if gfx == dataset_name), but without touching `runs`/resid_last, since
    only the count is needed. Used to help pick a sensible `balance_n`: it
    doesn't help to set balance_n above what the smallest datasets actually
    have available in the fit set.
    """
    counts = defaultdict(int)
    dataset_name_set = set(dataset_names)
    idx_iter = subset_indices if subset_indices is not None else range(len(eligible))
    for i in idx_iter:
        ex = eligible[i]
        task = ex.get(task_field)
        pair = component_task_map.get(task)
        if pair is None:
            continue
        qx, gfx = pair
        if qx in dataset_name_set:
            counts[qx] += 1
        if gfx in dataset_name_set:
            counts[gfx] += 1
    return {d: counts.get(d, 0) for d in dataset_names}


def _balanced_task_centroid(vecs, *, balance_n=None, seed=0):
    """
    Mean vector for a task. If `balance_n` is given and the task has more
    than `balance_n` vectors, subsample down to `balance_n` first so tasks
    with far more examples don't dominate the centroid (and downstream
    clustering) just by being bigger datasets.
    """
    if len(vecs) == 0:
        return None
    arr = np.stack(vecs, axis=0)
    if balance_n is not None and arr.shape[0] > balance_n:
        rng = np.random.default_rng(seed)
        sel = rng.choice(arr.shape[0], size=balance_n, replace=False)
        arr = arr[sel]
    return arr.mean(axis=0)


# ============================================================
# Per-layer clustering
# ============================================================

def _choose_k_by_silhouette(X, k_range, *, seed=0):#, min_improvement=0.02):
    """
    Fit KMeans for each k in k_range and pick k by silhouette score, biased
    towards the smallest/coarsest k that's still competitive.

    Plain argmax over k chases tiny, noise-driven score differences: with few
    points (one per task/dataset) in a high-dimensional space, a k that's
    barely better than a smaller k by e.g. 0.01 can still "win" argmax even
    though that's not meaningful evidence of extra structure — that's how you
    end up with k=5 when the real structure is closer to k=3.

    Instead, scan k from smallest to largest and only move up from the
    current best k when a larger k's score clears the current best by more
    than `min_improvement` (default 0.02, on the [-1, 1] silhouette scale).
    Each candidate is compared against the *best score seen so far*, not the
    immediately preceding k, so a string of small non-significant bumps can't
    accumulate into picking a much larger k either.

    Falls back to a single cluster (k=1) if there aren't enough points to
    try any k in the range.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    valid_ks = sorted(k for k in k_range if 2 <= k < n)
    if not valid_ks:
        return 1, {1: np.zeros(n, dtype=int)}, {}

    labels_by_k = {}
    scores = {}
    for k in valid_ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        labels_by_k[k] = labels
        try:
            scores[k] = float(silhouette_score(X, labels))
        except ValueError:
            scores[k] = float("-inf")

    best_k = valid_ks[0]
    best_score = scores[best_k]
    for k in valid_ks[1:]:
        if scores[k] > best_score:# + min_improvement:
            best_k = k
            best_score = scores[k]

    return best_k, labels_by_k, scores


def _pca_reduce(X, n_components=3, seed=0):
    """
    Project centroids onto their top `n_components` principal components
    before clustering. With few points (one per dataset) in a
    high-dimensional representation space, raw Euclidean distances get noisy
    (curse of dimensionality), which is one likely reason plain KMeans +
    silhouette was inflating the chosen k — PCA first cuts each centroid down
    to the directions that actually carry most of the variance.

    n_components is capped to what's valid for X's shape (can't exceed
    n_samples - 1 or n_features). Returns (X_reduced, pca_model,
    explained_variance_ratio) so callers can both transform new points into
    the same space and report how much variance was kept.
    """
    from sklearn.decomposition import PCA

    n_components = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca, pca.explained_variance_ratio_


def cluster_layer(
    eligible, runs, dataset_names, layer, component_task_map, *,
    task_field="task",
    subset_indices=None,
    k_range=range(2, 7),
    balance_n=200,
    seed=0,
    #min_silhouette_improvement=0.02,
    pca_n_components=3,
    pca_before_clustering=False,
):
    """
    Cluster single-hop dataset centroids at a single layer.

    `dataset_names` should be the flattened single-hop dataset names (e.g.
    sorted({d for pair in ALL_TASKS_QxFx_QFxGFx.values() for d in pair})),
    not composite two-hop task names — see module docstring.
    `component_task_map` (e.g. ALL_TASKS_QxFx_QFxGFx) is required to resolve,
    for each example, which single-hop dataset its v1/v2 vectors belong to.

    pca_before_clustering: if True, centroids are PCA-reduced to
    `pca_n_components` dimensions before KMeans + silhouette-based k
    selection (see _pca_reduce), and the straggler nearest-centroid fallback
    below also operates in that same reduced space, so the whole clustering
    decision is made consistently in PCA space rather than mixing spaces.
    If False (default), KMeans/silhouette and the straggler fallback operate
    directly on the raw (pre-PCA) centroids instead. Either way, a
    `pca_n_components`-dim PCA projection of the centroids is still computed
    and returned via "X_reduced"/"straggler_X_reduced" purely for plotting
    (see save_cluster_pca_plot_html) — it does not affect the actual cluster
    assignments unless pca_before_clustering=True.

    A dataset can end up with zero examples in `subset_indices` (e.g. a small
    dataset combined with a small `train_ratio`), in which case it can't
    contribute to fitting the clusters. Such "straggler" datasets are still
    assigned a cluster afterwards, by nearest-centroid against the clusters
    that *were* fit, using that dataset's centroid computed over *all* of
    `eligible` (not just the fit subset) — this only affects which existing
    cluster the straggler is filed under, not what the clusters themselves
    are, so it doesn't reintroduce train/test leakage into the cluster
    definitions. Every dataset in `dataset_names` that has at least one
    example anywhere in `eligible` is therefore guaranteed a cluster label.

    Returns: {
      "task_to_cluster": {dataset_name: cluster_label},   # cluster_label is a str, e.g. "0", "1"
      "k": chosen number of clusters,
      "silhouette_scores": {k: score},
      "task_centroids": {dataset_name: np.ndarray},   # original (pre-PCA) centroids
      "pca_explained_variance_ratio": np.ndarray,
      "datasets_sorted": [dataset_name, ...],   # order matching rows of X_reduced
      "X_reduced": np.ndarray,                  # (n_fit_datasets, pca_n_components), PCA coords (plotting only, unless pca_before_clustering=True)
      "straggler_datasets": [dataset_name, ...],       # datasets with no fit-set examples, assigned by nearest-centroid
      "straggler_X_reduced": np.ndarray,               # their PCA coords (via the fitted PCA), for plotting alongside X_reduced
    }
    """
    dataset_centroids = {}
    for d in dataset_names:
        vecs = _dataset_example_vectors(
            eligible, runs, d, layer, component_task_map,
            task_field=task_field, subset_indices=subset_indices,
        )
        c = _balanced_task_centroid(vecs, balance_n=balance_n, seed=seed)
        if c is not None:
            dataset_centroids[d] = c

    missing_datasets = [d for d in dataset_names if d not in dataset_centroids]

    if len(dataset_centroids) < 2:
        dataset_to_cluster = {d: "0" for d in dataset_centroids}
        for d in missing_datasets:
            vecs_all = _dataset_example_vectors(
                eligible, runs, d, layer, component_task_map,
                task_field=task_field, subset_indices=None,
            )
            if vecs_all:
                dataset_to_cluster[d] = "0"
        return {
            "task_to_cluster": dataset_to_cluster,
            "k": 1,
            "silhouette_scores": {},
            "task_centroids": dataset_centroids,
            "pca_explained_variance_ratio": np.array([]),
            "datasets_sorted": [],
            "X_reduced": np.zeros((0, 0), dtype=np.float64),
            "straggler_datasets": [],
            "straggler_X_reduced": np.zeros((0, 0), dtype=np.float64),
        }

    datasets_sorted = sorted(dataset_centroids.keys())
    X = np.stack([dataset_centroids[d] for d in datasets_sorted], axis=0)

    # Always computed for the debug plot; only used to drive the actual
    # clustering decision when pca_before_clustering=True (see below).
    X_viz, pca_viz, explained_variance_ratio = _pca_reduce(X, n_components=pca_n_components, seed=seed)

    X_cluster = X_viz if pca_before_clustering else X

    best_k, labels_by_k, scores = _choose_k_by_silhouette(
        X_cluster, k_range, seed=seed#, min_improvement=min_silhouette_improvement,
    )
    labels = labels_by_k[best_k]
    dataset_to_cluster = {d: str(int(c)) for d, c in zip(datasets_sorted, labels)}

    straggler_datasets = []
    straggler_viz_list = []
    if missing_datasets:
        cluster_ids = sorted(set(dataset_to_cluster.values()))
        cluster_means = {
            c: X_cluster[[i for i, d in enumerate(datasets_sorted) if dataset_to_cluster[d] == c]].mean(axis=0)
            for c in cluster_ids
        }
        for d in missing_datasets:
            vecs_all = _dataset_example_vectors(
                eligible, runs, d, layer, component_task_map,
                task_field=task_field, subset_indices=None,
            )
            centroid_all = _balanced_task_centroid(vecs_all, balance_n=balance_n, seed=seed)
            if centroid_all is None:
                continue  # dataset has no examples anywhere in `eligible`; leave unmapped
            centroid_all_viz = pca_viz.transform(centroid_all[None, :])[0]
            centroid_all_for_cluster = centroid_all_viz if pca_before_clustering else centroid_all
            nearest_c = min(cluster_ids, key=lambda c: float(np.linalg.norm(centroid_all_for_cluster - cluster_means[c])))
            dataset_to_cluster[d] = nearest_c
            straggler_datasets.append(d)
            straggler_viz_list.append(centroid_all_viz)

    straggler_X_reduced = (
        np.stack(straggler_viz_list, axis=0) if straggler_viz_list
        else np.zeros((0, X_viz.shape[1]), dtype=X_viz.dtype)
    )

    return {
        "task_to_cluster": dataset_to_cluster,
        "k": int(best_k),
        "silhouette_scores": {int(k): v for k, v in scores.items()},
        "task_centroids": dataset_centroids,
        "pca_explained_variance_ratio": explained_variance_ratio,
        "datasets_sorted": datasets_sorted,
        "X_reduced": X_viz,
        "straggler_datasets": straggler_datasets,
        "straggler_X_reduced": straggler_X_reduced,
    }


def save_cluster_pca_plot_html(
    layer, info, save_path, *,
    k=None,
):
    """
    Interactive HTML scatter of one layer's PCA-reduced dataset centroids,
    colored by assigned cluster label. Meant to be called right after
    `cluster_layer` for that layer, using its returned `info` dict directly
    (datasets_sorted / X_reduced / straggler_datasets / straggler_X_reduced /
    task_to_cluster / pca_explained_variance_ratio) — this is the same PCA
    space the clustering decision was actually made in, so the plot shows
    what silhouette/KMeans saw rather than a re-projection.

    Fit datasets are drawn as circles, "straggler" datasets (no fit-set
    examples, assigned post-hoc by nearest-centroid) as diamonds, so it's
    visible which points actually shaped the clusters.

    No-ops (with a print) if there are fewer than 2 fit datasets, since
    `cluster_layer` skips PCA entirely in that case.
    """
    import plotly.graph_objects as go

    datasets_sorted = info["datasets_sorted"]
    X_reduced = info["X_reduced"]
    straggler_datasets = info.get("straggler_datasets", [])
    straggler_X_reduced = info.get("straggler_X_reduced", np.zeros((0, 0)))
    task_to_cluster = info["task_to_cluster"]
    evr = info.get("pca_explained_variance_ratio", np.array([]))

    if len(datasets_sorted) == 0:
        print(f"[multihop_experiment] layer {layer}: skipping cluster plot (fewer than 2 fit datasets).")
        return

    n_components = X_reduced.shape[1]
    all_names = list(datasets_sorted) + list(straggler_datasets)
    all_coords = np.concatenate([X_reduced, straggler_X_reduced], axis=0) if len(straggler_datasets) else X_reduced
    is_straggler = [False] * len(datasets_sorted) + [True] * len(straggler_datasets)
    cluster_labels = [task_to_cluster[d] for d in all_names]

    fig = go.Figure()
    for c in sorted(set(cluster_labels), key=lambda x: int(x)):
        idxs = [i for i, cl in enumerate(cluster_labels) if cl == c]
        fit_idxs = [i for i in idxs if not is_straggler[i]]
        straggler_idxs = [i for i in idxs if is_straggler[i]]

        for group_idxs, symbol, name_suffix in (
            (fit_idxs, "circle", ""),
            (straggler_idxs, "diamond", " (straggler)"),
        ):
            if not group_idxs:
                continue
            coords = all_coords[group_idxs]
            names = [all_names[i] for i in group_idxs]
            scatter_kwargs = dict(
                name=f"cluster {c}{name_suffix}",
                mode="markers",
                text=names,
                hovertemplate="%{text}<extra>cluster %{fullData.name}</extra>",
                marker=dict(size=9, symbol=symbol),
            )
            if n_components >= 3:
                fig.add_trace(go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **scatter_kwargs,
                ))
            elif n_components == 2:
                fig.add_trace(go.Scatter(
                    x=coords[:, 0], y=coords[:, 1], **scatter_kwargs,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=coords[:, 0], y=np.zeros(len(group_idxs)), **scatter_kwargs,
                ))

    evr_str = f", explained var={float(np.sum(evr)):.3f}" if len(evr) else ""
    k_str = f", k={k}" if k is not None else ""
    title = f"Layer {layer} auto-cluster PCA ({n_components}D){k_str}{evr_str}"

    axis_kwargs = dict(title=title)
    if n_components >= 3:
        fig.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), **axis_kwargs)
    else:
        fig.update_layout(xaxis_title="PC1", yaxis_title=("PC2" if n_components == 2 else ""), **axis_kwargs)

    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.write_html(save_path)


# ============================================================
# Exact cluster means (mirrors the original manual exact_mu computation)
# ============================================================

def compute_cluster_means(
    eligible, runs, dataset_to_cluster_layer, layer, d_model, component_task_map, *,
    task_field="task",
    subset_indices=None,
):
    """
    Exact unnormalized mean representation per cluster at `layer`, restricted
    to `subset_indices` (the fit set). Mirrors the original manual code's
    sum-then-divide computation, just keyed by automatic per-single-hop-dataset
    labels instead of hand-written A/B/C.

    v1 (QxFx) and v2 (QFxGFx) are summed into the cluster of their *own*
    single-hop dataset (qx, gfx respectively, from component_task_map), which
    can differ for the same example — this is the whole point of clustering
    at single-hop-dataset granularity rather than per composite task.
    """
    sums = defaultdict(lambda: np.zeros(d_model, dtype=np.float32))
    counts = defaultdict(int)

    idx_iter = subset_indices if subset_indices is not None else range(len(eligible))
    for i in idx_iter:
        ex = eligible[i]
        task = ex.get(task_field)
        pair = component_task_map.get(task)
        if pair is None:
            continue
        qx, gfx = pair

        cluster_qx = dataset_to_cluster_layer.get(qx)
        if cluster_qx is not None:
            run_idx_1 = ex["run_idx_qx_fx"]
            i1 = ex["idx_qx_fx"]
            v1 = np.asarray(runs[run_idx_1]["resid_last"][i1, layer], dtype=np.float32)
            sums[cluster_qx] += v1
            counts[cluster_qx] += 1

        cluster_gfx = dataset_to_cluster_layer.get(gfx)
        if cluster_gfx is not None:
            run_idx_2 = ex["run_idx_qfx_gfx"]
            i2 = ex["idx_qfx_gfx"]
            v2 = np.asarray(runs[run_idx_2]["resid_last"][i2, layer], dtype=np.float32)
            sums[cluster_gfx] += v2
            counts[cluster_gfx] += 1

    return {c: (sums[c] / counts[c]) for c in sums if counts[c] > 0}


# ============================================================
# Debug: compare against manual ground-truth clusters
# ============================================================

def _manual_task_to_cluster(manual_clusters, component_task_map=None, default_label=None):
    """
    Flatten manual_clusters (e.g. {"A": A_datasets, "B": B_datasets, "C": C_datasets})
    into {dataset_name: label}. Auto clusters are now keyed by single-hop
    dataset name (see module docstring), the same granularity manual_clusters
    already uses, so no composite-task mapping or OR-across-components logic
    is needed anymore — a dataset simply keeps whatever label it was listed
    under. `component_task_map`/`default_label` are accepted only for
    backward-compatible call signatures and are unused.
    """
    component_to_label = {}
    for label, comps in manual_clusters.items():
        for c in comps:
            component_to_label[c] = label
    return component_to_label


def _adjusted_rand_index(labels_a, labels_b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))


def _compare_to_manual(task_to_cluster_by_layer, manual_clusters, component_task_map, default_label="C"):
    """
    Per layer: Adjusted Rand Index between the automatic clustering and the
    manual ground truth, plus a contingency table {manual_label: {auto_label: count}}
    so you can see e.g. "manual A mostly landed in auto cluster 2".
    """
    manual_map = _manual_task_to_cluster(manual_clusters, component_task_map, default_label)
    common_tasks = sorted(manual_map.keys())

    debug = {"ari_by_layer": {}, "contingency_by_layer": {}}
    for layer, auto_map in task_to_cluster_by_layer.items():
        pairs = [(manual_map[t], auto_map[t]) for t in common_tasks if t in auto_map]
        if len(pairs) < 2:
            continue
        y_manual = [p[0] for p in pairs]
        y_auto = [p[1] for p in pairs]
        debug["ari_by_layer"][layer] = _adjusted_rand_index(y_manual, y_auto)

        contingency = defaultdict(lambda: defaultdict(int))
        for m, a in zip(y_manual, y_auto):
            contingency[m][a] += 1
        debug["contingency_by_layer"][layer] = {m: dict(a) for m, a in contingency.items()}

    return debug


def print_cluster_report(
    task_to_cluster_by_layer, *,
    manual_clusters=None,
    component_task_map=None,
    manual_default_label="C",
    debug=None,
    layers=None,
):
    """
    Print, per layer, which tasks landed in each automatic cluster.

    If `debug` (the {"ari_by_layer", "contingency_by_layer"} dict returned by
    run_auto_clustering, e.g. from a saved/loaded result) is given, reuse it
    to also print the Adjusted Rand Index and contingency table against the
    manual ground truth. Otherwise, if `manual_clusters` + `component_task_map`
    are given, compute that comparison on the fly (needed when a saved result
    was loaded without debug info, or with a different manual grouping than
    the one it was originally computed against).
    """
    layers = sorted(task_to_cluster_by_layer.keys()) if layers is None else layers

    if debug is None and manual_clusters is not None and component_task_map is not None:
        debug = _compare_to_manual(
            task_to_cluster_by_layer, manual_clusters, component_task_map,
            default_label=manual_default_label,
        )

    manual_map = None
    if manual_clusters is not None and component_task_map is not None:
        manual_map = _manual_task_to_cluster(manual_clusters, component_task_map, manual_default_label)

    for L in layers:
        auto_map = task_to_cluster_by_layer[L]
        print(f"\n=== Layer {L}: auto clusters ===")
        by_cluster = defaultdict(list)
        for t, c in auto_map.items():
            by_cluster[c].append(t)
        for c in sorted(by_cluster.keys()):
            print(f"  cluster {c} ({len(by_cluster[c])} tasks): {sorted(by_cluster[c])}")

        if manual_map is None:
            continue

        if debug is not None and L in debug.get("ari_by_layer", {}):
            ari = debug["ari_by_layer"][L]
            contingency = debug["contingency_by_layer"][L]
        else:
            common_tasks = [t for t in manual_map if t in auto_map]
            if len(common_tasks) < 2:
                continue
            y_manual = [manual_map[t] for t in common_tasks]
            y_auto = [auto_map[t] for t in common_tasks]
            ari = _adjusted_rand_index(y_manual, y_auto)
            contingency_dd = defaultdict(lambda: defaultdict(int))
            for m, a in zip(y_manual, y_auto):
                contingency_dd[m][a] += 1
            contingency = {m: dict(a) for m, a in contingency_dd.items()}

        print(f"  ARI vs manual labels: {ari:.3f}")
        print("  contingency (manual_label -> {auto_cluster: count}):")
        for m in sorted(contingency.keys()):
            print(f"    {m}: {dict(sorted(contingency[m].items()))}")


# ============================================================
# Orchestration
# ============================================================

def run_auto_clustering(
    eligible, runs, task_names, n_layers, d_model, component_task_map, *,
    task_field="task",
    auto_cluster_using="train",   # "train" or "all"
    train_ratio=0.1,
    k_range=range(2, 7),
    balance_n=200,
    seed=0,
    #min_silhouette_improvement=0.02,
    pca_n_components=3,
    pca_before_clustering=False,
    debug_mode=False,
    manual_clusters=None,          # e.g. {"A": A_datasets, "B": B_datasets, "C": C_datasets}
    manual_default_label="C",
    save_path=None,
    debug_plot_dir=None,
):
    """
    Automatically cluster single-hop datasets into groups at every layer and
    compute each cluster's exact mean representation, restricted to the fit
    set chosen by `auto_cluster_using`.

    `task_names` must be single-hop dataset names (e.g.
    sorted({d for pair in ALL_TASKS_QxFx_QFxGFx.values() for d in pair})),
    not composite two-hop task names — see module docstring for why.
    `component_task_map` (e.g. ALL_TASKS_QxFx_QFxGFx) is required: it's how
    every example's v1/v2 vectors get resolved to their own single-hop
    dataset (qx, gfx) for both clustering and mean-centering.

    pca_n_components: dataset centroids are projected to this many principal
    components for the debug plot (see save_cluster_pca_plot_html), and also
    used to drive KMeans + silhouette-based k selection when
    `pca_before_clustering=True` (see _pca_reduce / cluster_layer) —
    mitigates the high-dimensionality/small-n noise that can otherwise
    inflate the chosen k.

    pca_before_clustering: if True, cluster on the PCA-reduced centroids
    instead of the raw ones (see cluster_layer). Default False: clustering
    runs directly on raw centroids; PCA is still computed for the debug plot
    only.

    auto_cluster_using:
      "train": fit clusters and cluster means on a `train_ratio` random
        subset of `eligible` only. Matches this codebase's convention
        elsewhere (see _split_train_test) of keeping test examples out of
        anything that shapes the reported feature. Recommended default.
      "all": fit on every example, including test. Uses the full large-scale
        structure but is a soft leakage path, since the cluster centers used
        to center TEST vectors are then partly derived from those same TEST
        vectors. Use mainly as a sensitivity check against "train" (compare
        downstream PP/AUC between the two to see whether it actually matters
        empirically) rather than as the default.

    min_silhouette_improvement: a larger k only replaces a smaller one if its
      silhouette score beats the best score seen so far by more than this
      margin (see _choose_k_by_silhouette). Raise this if k still looks
      inflated; lower/zero it to recover plain argmax-over-k behavior.

    debug_plot_dir: if set and `debug_mode=True`, an interactive Plotly HTML
      scatter of each layer's PCA-reduced dataset centroids (colored by
      assigned cluster) is saved to
      f"{debug_plot_dir}/layer_{L:02d}_clusters.html" right after that
      layer's clustering/PCA is computed — see save_cluster_pca_plot_html.
      Ignored if debug_mode=False.

    Returns: {
      "task_to_cluster": {layer: {dataset_name: cluster_label}},
      "exact_mu": {cluster_label: [vec_per_layer]},   # same shape as the old manual exact_mu
      "k_by_layer": {layer: k},
      "silhouette_by_layer": {layer: {k: score}},
      "debug": {"ari_by_layer": {...}, "contingency_by_layer": {...}} or None,
      "hyperparams": {...},
    }
    """
    if auto_cluster_using not in ("train", "all"):
        raise ValueError("auto_cluster_using must be 'train' or 'all'")

    if auto_cluster_using == "train":
        all_idx = np.arange(len(eligible), dtype=np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(all_idx)
        n_train = max(1, int(round(train_ratio * len(perm))))
        fit_indices = perm[:n_train]
    else:
        fit_indices = np.arange(len(eligible), dtype=np.int64)

    if debug_mode:
        fit_counts = _dataset_example_counts(
            eligible, task_names, component_task_map,
            task_field=task_field, subset_indices=fit_indices,
        )
        total_counts = _dataset_example_counts(
            eligible, task_names, component_task_map,
            task_field=task_field, subset_indices=None,
        )
        counts_vals = list(fit_counts.values())
        print(f"[multihop_experiment] dataset sizes (auto_cluster_using={auto_cluster_using!r}, "
              f"fit set = {len(fit_indices)}/{len(eligible)} examples):")
        print(f"  fit-set vector count per dataset: min={min(counts_vals)}, "
              f"median={int(np.median(counts_vals))}, max={max(counts_vals)}")
        print(f"  {'dataset':<30s} {'fit_set':>10s} {'total':>10s}")
        for d in sorted(task_names, key=lambda d: fit_counts[d]):
            print(f"  {d:<30s} {fit_counts[d]:>10d} {total_counts[d]:>10d}")
        if balance_n is not None and balance_n > min(counts_vals):
            print(f"  [note] balance_n={balance_n} exceeds the fit-set count for "
                  f"{sum(1 for v in counts_vals if v < balance_n)}/{len(counts_vals)} dataset(s) — "
                  f"those get all their vectors used (no subsampling), so they'll still weigh less "
                  f"than datasets with more than balance_n vectors.")

    task_to_cluster_by_layer = {}
    k_by_layer = {}
    silhouette_by_layer = {}
    pca_explained_variance_by_layer = {}
    all_cluster_labels = set()

    for L in range(n_layers):
        info = cluster_layer(
            eligible, runs, task_names, L, component_task_map,
            task_field=task_field, subset_indices=fit_indices,
            k_range=k_range, balance_n=balance_n, seed=seed,
            #min_silhouette_improvement=min_silhouette_improvement,
            pca_n_components=pca_n_components,
            pca_before_clustering=pca_before_clustering,
        )
        task_to_cluster_by_layer[L] = info["task_to_cluster"]
        k_by_layer[L] = info["k"]
        silhouette_by_layer[L] = info["silhouette_scores"]

        if debug_mode and debug_plot_dir is not None:
            save_cluster_pca_plot_html(
                L, info,
                os.path.join(debug_plot_dir, f"layer_{L:02d}_clusters.html"),
                k=info["k"],
            )
        pca_explained_variance_by_layer[L] = info["pca_explained_variance_ratio"]
        all_cluster_labels.update(info["task_to_cluster"].values())

    if debug_mode:
        for L in range(n_layers):
            evr = pca_explained_variance_by_layer[L]
            if len(evr) > 0:
                print(f"[multihop_experiment] layer {L}: PCA({pca_n_components}) explained variance "
                      f"= {float(np.sum(evr)):.3f} (per-component: {[round(float(v), 3) for v in evr]})")

    exact_mu = {c: [np.zeros(d_model, dtype=np.float32) for _ in range(n_layers)] for c in all_cluster_labels}
    for L in range(n_layers):
        mu_L = compute_cluster_means(
            eligible, runs, task_to_cluster_by_layer[L], L, d_model, component_task_map,
            task_field=task_field, subset_indices=fit_indices,
        )
        for c, v in mu_L.items():
            exact_mu[c][L] = v

    debug = None
    if debug_mode:
        if manual_clusters is None:
            print("[multihop_experiment] debug_mode=True but manual_clusters not provided; "
                  "skipping ground-truth comparison.")
        else:
            debug = _compare_to_manual(
                task_to_cluster_by_layer, manual_clusters, component_task_map,
                default_label=manual_default_label,
            )
            for L in sorted(debug["ari_by_layer"].keys()):
                print(f"[multihop_experiment] layer {L}: ARI vs manual = {debug['ari_by_layer'][L]:.3f} "
                      f"(auto k={k_by_layer[L]})")

    result = {
        "task_to_cluster": task_to_cluster_by_layer,
        "exact_mu": exact_mu,
        "k_by_layer": k_by_layer,
        "silhouette_by_layer": silhouette_by_layer,
        "debug": debug,
        "hyperparams": {
            "auto_cluster_using": auto_cluster_using,
            "train_ratio": train_ratio,
            "k_range": list(k_range),
            "balance_n": balance_n,
            "seed": seed,
            #"min_silhouette_improvement": min_silhouette_improvement,
            "pca_n_components": pca_n_components,
            "pca_before_clustering": pca_before_clustering,
            "n_fit_examples": int(len(fit_indices)),
            "n_total_examples": int(len(eligible)),
        },
    }

    if save_path is not None:
        save_auto_clusters(result, save_path)

    return result


# ============================================================
# Save / load
# ============================================================

def save_auto_clusters(result, save_path):
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "task_to_cluster": {str(L): v for L, v in result["task_to_cluster"].items()},
        "exact_mu": {
            c: [v.tolist() for v in vecs] for c, vecs in result["exact_mu"].items()
        },
        "k_by_layer": {str(L): k for L, k in result["k_by_layer"].items()},
        "silhouette_by_layer": {
            str(L): {str(k): v for k, v in scores.items()}
            for L, scores in result["silhouette_by_layer"].items()
        },
        "debug": (
            {
                "ari_by_layer": {str(L): v for L, v in result["debug"]["ari_by_layer"].items()},
                "contingency_by_layer": {
                    str(L): v for L, v in result["debug"]["contingency_by_layer"].items()
                },
            }
            if result.get("debug") is not None else None
        ),
        "hyperparams": result["hyperparams"],
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[multihop_experiment] Saved auto clusters to: {save_path}")


def load_auto_clusters(save_path):
    with open(save_path, "r") as f:
        payload = json.load(f)

    task_to_cluster = {int(L): v for L, v in payload["task_to_cluster"].items()}
    exact_mu = {
        c: [np.asarray(v, dtype=np.float32) for v in vecs]
        for c, vecs in payload["exact_mu"].items()
    }
    return {
        "task_to_cluster": task_to_cluster,
        "exact_mu": exact_mu,
        "k_by_layer": {int(L): k for L, k in payload["k_by_layer"].items()},
        "silhouette_by_layer": {
            int(L): {int(k): v for k, v in scores.items()}
            for L, scores in payload["silhouette_by_layer"].items()
        },
        "debug": payload.get("debug"),
        "hyperparams": payload["hyperparams"],
    }
