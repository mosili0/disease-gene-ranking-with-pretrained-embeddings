"""Identical folds: one held-out positive + all unlabelled background."""
import numpy as np
from . import metrics as M

PROTOCOL = "leave-one-positive-out_seed-excluded_v2"


def evaluate(order, positives, scorer):
    """Score every fold with ONE query; remove its seed genes from candidates.

    scorer(seed_indices) returns scores for all genes. Scores are never pooled
    across folds. This is association completion against unlabelled background.
    """
    pos = [i for i, g in enumerate(order) if g in set(positives)]
    bg = [i for i in range(len(order)) if i not in pos]
    if not pos or not bg:
        raise ValueError("evaluation needs positive and background candidates")
    folds = []
    for held in pos:
        seeds = [i for i in pos if i != held]
        candidates = [held] + bg
        scores = np.asarray(scorer(seeds), dtype=float)[candidates]
        y = np.r_[1., np.zeros(len(bg))]
        top = np.argsort(-scores, kind="stable")[:10]
        folds.append({
            "held_out": order[held], "seeds": [order[i] for i in seeds],
            "candidates": [order[i] for i in candidates],
            "scores": [float(v) for v in scores], "n_candidates": len(candidates),
            "auc": M.roc_auc(y, scores), "average_precision": M.average_precision(y, scores),
            "hit_at_10": M.hit_at_k(scores, [0], 10), "mrr": M.mrr(scores, [0]),
            "rank": M.ranks_of(scores, [0])[0],
            "top10": [order[candidates[i]] for i in top],
        })
    summary = {k: float(np.mean([f[k] for f in folds]))
               for k in ["auc", "average_precision", "hit_at_10", "mrr"]}
    summary.update(median_rank=float(np.median([f["rank"] for f in folds])),
                   n_pos=len(pos), n_background=len(bg), n_candidates=len(bg) + 1,
                   protocol=PROTOCOL)
    return summary, folds


def centroid_scorer(X):
    def score(seeds):
        if not seeds:
            raise ValueError("centroid requires at least one seed")
        return M.cosine_scores(X, X[seeds].mean(0))
    return score


def random_expectation(n_candidates):
    """Exact expectations for the single-held-out-positive protocol."""
    n = n_candidates
    h = float(np.sum(1 / np.arange(1, n + 1)))
    return dict(auc=0.5, hit_at_10=min(10 / n, 1), median_rank=(n + 1) / 2,
                mrr=h / n, average_precision=h / n, protocol=PROTOCOL)
