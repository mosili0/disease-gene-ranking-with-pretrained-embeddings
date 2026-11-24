# -*- coding: utf-8 -*-
"""Tie-aware ranking metrics; background labels are not proven negatives."""
import numpy as np
from scipy.stats import rankdata, spearmanr


def cosine_matrix(X):
    X = np.asarray(X)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return Xn @ Xn.T


def cosine_scores(X, q):
    X = np.asarray(X)
    return (X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)) @ (
        q / max(float(np.linalg.norm(q)), 1e-12))


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _inputs(y, scores):
    y, s = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not len(y):
        raise ValueError("labels and scores must be nonempty aligned 1-D arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("labels must be binary and scores finite")
    return y, s


def roc_auc(y, scores):
    """Mann-Whitney AUC; tied positive/background pairs receive 0.5."""
    y, s = _inputs(y, scores)
    p, n = y.sum(), (1 - y).sum()
    if not p or not n:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def roc_curve_points(y, scores, n_points=None):
    """Keep whole equal-score groups; optional thinning preserves endpoints."""
    y, s = _inputs(y, scores)
    p, n = y.sum(), (1 - y).sum()
    if not p or not n:
        return None
    ix = np.argsort(-s, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(s[ix]) != 0), len(s) - 1]
    tp, fp = np.cumsum(y[ix])[ends], np.cumsum(1 - y[ix])[ends]
    fpr, tpr = np.r_[0, fp / n], np.r_[0, tp / p]
    if n_points is not None and len(fpr) > n_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return fpr, tpr


def average_precision(y, scores):
    """Non-interpolated AP with equal-score threshold groups."""
    y, s = _inputs(y, scores)
    if not y.sum():
        return float("nan")
    ix = np.argsort(-s, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(s[ix]) != 0), len(s) - 1]
    tp = np.cumsum(y[ix])[ends]
    return float(np.sum(np.diff(np.r_[0, tp / y.sum()]) * tp / (ends + 1)))


def ranks_of(scores, pos_indices):
    """One-based midranks, independent of order within ties."""
    return rankdata(-np.asarray(scores), method="average")[np.asarray(pos_indices, dtype=int)].tolist()


def hit_at_k(scores, pos_indices, k=10):
    """Expected Recall@k under uniformly random tie-breaking."""
    s = np.asarray(scores)
    credits = []
    for i in pos_indices:
        above, tied = np.sum(s > s[i]), np.sum(s == s[i])
        credits.append(float(np.clip((k - above) / tied, 0, 1)))
    return float(np.mean(credits)) if credits else float("nan")


def mrr(scores, pos_indices):
    """Expected reciprocal rank under random tie-breaking."""
    s = np.asarray(scores)
    vals = []
    for i in pos_indices:
        above, tied = np.sum(s > s[i]), np.sum(s == s[i])
        vals.append(np.mean(1 / np.arange(above + 1, above + tied + 1)))
    return float(np.mean(vals)) if vals else float("nan")


def median_rank(scores, pos_indices):
    r = ranks_of(scores, pos_indices)
    return float(np.median(r)) if r else float("nan")
