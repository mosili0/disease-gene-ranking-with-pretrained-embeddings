# -*- coding: utf-8 -*-
"""Frozen representations: GO similarity and seed-based association completion."""
import os
import numpy as np
import pandas as pd
import config
from . import metrics as M
from .evaluation import evaluate, centroid_scorer, random_expectation, PROTOCOL
from .utils import write_json


def matrix_from_dict(embs, genes):
    order = [g for g in genes if g in embs]
    return np.stack([embs[g] for g in order]).astype("float32"), order


def _bh(pvalues):
    p = np.asarray(pvalues)
    ix = np.argsort(p)
    q = np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[ix] = np.clip(q, 0, 1)
    return out


def task1_functional_similarity(representations, gene_info, genes):
    # GO-IDs, not term-name strings. Both genes must have annotations.
    import re
    go = {g: set(re.findall(r"GO:\d+", " ".join(gene_info[g].get("go_bp", []))))
          for g in genes}
    rows, pair_tables = [], {}
    for name, embs in representations.items():
        order = [g for g in genes if g in embs and go[g]]
        X, order = matrix_from_dict(embs, order)
        C = M.cosine_matrix(X)
        ii, jj = np.triu_indices(len(order), 1)
        jac = np.array([len(go[order[i]] & go[order[j]]) / len(go[order[i]] | go[order[j]])
                        for i, j in zip(ii, jj)])
        cos, y = C[ii, jj], (jac > 0).astype(float)
        precs = []
        for i, g in enumerate(order):
            candidates = [j for j in range(len(order)) if j != i]
            labels = [j for j in range(len(candidates)) if go[g] & go[order[candidates[j]]]]
            # Expected number of labelled neighbours in top-k, including tie boundaries.
            k = min(10, len(candidates))
            precs.append(M.hit_at_k(C[i, candidates], labels, k) * len(labels) / k
                         if labels else 0.)
        auc = M.roc_auc(y, cos)
        # Permute gene identities on BOTH matrix axes, retaining pair dependence.
        rng = np.random.default_rng(config.RANDOM_SEED)
        null = []
        for _ in range(config.PERMUTATIONS):
            perm = rng.permutation(len(order))
            null.append(M.roc_auc(y, C[perm[ii], perm[jj]]))
        rows.append(dict(model=name, n_genes=len(order), n_pairs=len(y),
                         n_pos_pairs=int(y.sum()), n_neg_pairs=int((1-y).sum()),
                         spearman_cosine_vs_go_jaccard=M.spearman(cos, jac),
                         roc_auc_shared_go=auc, precision_at_10=float(np.mean(precs)),
                         random_precision_at_10=float(y.mean()),
                         permutation_p_auc=(1 + np.sum(np.asarray(null) >= auc)) / (1 + len(null)),
                         n_permutations=len(null)))
        pair_tables[name] = (M.roc_curve_points(y, cos), np.ones(len(y), bool), cos, jac)
    df = pd.DataFrame(rows)
    df["permutation_q_auc_bh"] = _bh(df.permutation_p_auc.values)
    df.to_csv(os.path.join(config.RESULTS_DIR, "task1_functional_similarity.csv"),
              index=False, encoding="utf-8-sig")
    arrays = {}
    for name, (_, _, cos, jac) in pair_tables.items():
        arrays[name + "_cos"] = cos
        arrays[name + "_jac"] = jac
    np.savez_compressed(os.path.join(config.RESULTS_DIR, "task1_pair_data.npz"), **arrays)
    return df, pair_tables


def evaluate_entity(key, d, field, representations, query_embs, tfidf, order,
                    prefix, positives=None, network_scorer=None):
    """Use identical held-out/candidate splits for every comparable method."""
    pos = [g for g in order if g in set(d[field] if positives is None else positives)]
    if not pos or len(pos) == len(order):
        return [], {"note": "needs at least one positive and one background", "positives": pos}
    methods = []
    if len(pos) >= 2:
        for name, embs in representations.items():
            X, aligned = matrix_from_dict(embs, order)
            assert aligned == order
            methods.append((name, "centroid", centroid_scorer(X)))
        if network_scorer is not None:
            methods.append(("string", "network-baseline", network_scorer))
    query_keys = [("mechanism-query", prefix + "_" + key)]
    if prefix == "drug":
        query_keys.append(("indication-query", "drug_indication_" + key))
    for method, qkey in query_keys:
        if "biobert" in representations:
            X, _ = matrix_from_dict(representations["biobert"], order)
            sc = M.cosine_scores(X, query_embs[qkey])
            methods.append(("biobert", method, lambda seeds, sc=sc: sc))
        sc = M.cosine_scores(tfidf.X, tfidf.transform([config.all_queries()[qkey]])[0])
        methods.append(("tfidf", method, lambda seeds, sc=sc: sc))
    rows, detail = [], {"positives": pos, "background_status": "unlabelled", "methods": {}}
    for name, method, scorer in methods:
        summary, folds = evaluate(order, pos, scorer)
        rows.append(dict(model=name, name=name, method=method, **summary))
        detail["methods"][name + "/" + method] = folds
    rnd = random_expectation(len(order) - len(pos) + 1)
    rows.append(dict(model="random", name="random", method="random-expectation",
                     n_pos=len(pos), n_background=len(order)-len(pos),
                     n_candidates=len(order)-len(pos)+1, **rnd))
    return rows, detail


def _manual(entities, field, reps, queries, tfidf, order, prefix, filename):
    rows, detail = [], {}
    for key, d in entities.items():
        rr, dd = evaluate_entity(key, d, field, reps, queries, tfidf, order, prefix)
        detail[key] = dd
        for r in rr:
            r.update({prefix if prefix == "drug" else "disease": key,
                      "drug_label" if prefix == "drug" else "disease_label": d["label"]})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, filename + ".csv"), index=False, encoding="utf-8-sig")
    write_json(os.path.join(config.RESULTS_DIR, filename.split("_")[0] + "_detail.json"), detail)
    return df, detail


def run_all_tasks(representations, query_embs, gene_info, tfidf, genes):
    t1, pairs = task1_functional_similarity(representations, gene_info, genes)
    t2, d2 = _manual(config.DISEASES, "genes", representations, query_embs, tfidf, genes,
                     "disease", "task2_disease_genes")
    t3, d3 = _manual(config.DRUGS, "targets", representations, query_embs, tfidf, genes,
                     "drug", "task3_drug_targets")
    return dict(task1=t1, task2=t2, task3=t3, task1_pairs=pairs, task2_detail=d2,
                task3_detail=d3, protocol=PROTOCOL)
