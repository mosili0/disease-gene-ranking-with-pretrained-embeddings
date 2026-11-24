# -*- coding: utf-8 -*-
"""结果可视化（matplotlib，图表标签使用英文以避免字体问题）。"""
import os

import numpy as np

import config
from . import metrics as M

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(config.RESULTS_DIR, "figures")
CAT_COLORS = plt.cm.tab20(np.linspace(0, 1, len(config.GENE_CATEGORIES)))
CAT_OF = {g: i for i, genes in enumerate(config.GENE_CATEGORIES.values())
          for g in genes}


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[viz] saved %s" % name)


def fig_heatmap(gene_embs, genes, name="esm", per_cat=4, out="fig1_similarity_heatmap.png"):
    """部分基因的余弦相似度热图（按类别分组）。"""
    sel = []
    for gs in config.GENE_CATEGORIES.values():
        sel += [g for g in gs if g in gene_embs[name]][:per_cat]
    from .tasks import matrix_from_dict
    X, _ = matrix_from_dict(gene_embs[name], sel)
    C = M.cosine_matrix(X)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sel)))
    ax.set_yticks(range(len(sel)))
    ax.set_xticklabels(sel, rotation=90, fontsize=7)
    ax.set_yticklabels(sel, fontsize=7)
    for i, g in enumerate(sel):
        ax.get_xticklabels()[i].set_color(CAT_COLORS[CAT_OF[g]])
        ax.get_yticklabels()[i].set_color(CAT_COLORS[CAT_OF[g]])
    ax.set_title("Gene-gene cosine similarity (%s embeddings)" % name.upper())
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)


def fig_roc(pair_tables, out="fig2_roc_functional_similarity.png"):
    """任务 1 的 ROC 曲线（共享 GO 术语的基因对为正类）。"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (pts, both, cos, jac) in pair_tables.items():
        if pts is None:
            continue
        y = (jac > 0).astype(float)
        auc = M.roc_auc(y, cos)
        ax.plot(pts[0], pts[1], lw=2, label="%s (AUC=%.3f)" % (name.upper(), auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Task 1: predict gene pairs sharing GO-BP terms")
    ax.legend()
    _save(fig, out)


def _en_label(label):
    """'乳腺癌 (Breast cancer)' -> 'Breast cancer'（图表字体不含中文字形）。"""
    return label.split("(")[-1].rstrip(")").strip() if "(" in label else label


def fig_disease_auc(df2, out="fig3_disease_gene_auc.png"):
    """任务 2：各疾病 × 方法 的 AUC 分组柱状图。"""
    df = df2[df2["method"].isin(["centroid", "mechanism-query"])].copy()
    if df.empty:
        return
    df["label_en"] = df["disease_label"].map(_en_label)
    pivot = df.pivot_table(index="label_en", columns=["model", "method"],
                           values="auc", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("AUC")
    ax.set_xlabel("Disease")
    ax.set_title("Task 2: manual labels, fold-mean AUC (seed genes excluded)")
    ax.axhline(.5, color="black", ls="--", lw=1)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.35))
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=9)
    fig = ax.get_figure()
    _save(fig, out)


def fig_drug_targets(df3, out="fig4_drug_target_metrics.png"):
    """任务 3：药物文本查询 AUC 与多靶点药物质心留一 hit@10。"""
    q = df3[df3["method"].isin(["mechanism-query", "indication-query"])].copy()
    q["label_en"] = q["drug_label"].map(_en_label)
    q["series"] = q["model"] + "/" + q["method"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    q.pivot_table(index="label_en", columns="series", values="auc").plot.bar(ax=axes[0])
    axes[0].axhline(0.5, color="k", ls="--", lw=1)
    axes[0].set_ylabel("Fold-mean AUC")
    axes[0].set_title("Task 3: mechanism-text retrieval vs indication-only control")
    axes[0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xlabel("Drug")
    axes[0].set_ylim(0, 1.05)
    c = df3[df3["method"] == "centroid"].copy()
    if not c.empty:
        c["label_en"] = c["drug_label"].map(_en_label)
        c.pivot_table(index="label_en", columns="model", values="hit_at_10").plot.bar(ax=axes[1])
        axes[1].set_ylabel("LOO Recall@10 (seed-excluded)")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[1].legend(fontsize=8, ncol=5)
        axes[1].set_xlabel("Drug")
        axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, out)


def fig_pca(gene_embs, name="esm", out="fig5_pca_gene_space.png"):
    """基因嵌入 PCA 二维投影，按功能类别着色。"""
    from .tasks import matrix_from_dict
    X, order = matrix_from_dict(gene_embs[name], config.ALL_GENES)
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, (cat, genes) in enumerate(config.GENE_CATEGORIES.items()):
        idx = [order.index(g) for g in genes if g in order]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=55, color=CAT_COLORS[ci],
                   label=cat, edgecolor="white", lw=0.5)
        for i in idx:
            ax.annotate(order[i], (Z[i, 0], Z[i, 1]), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Gene embedding space (%s), PCA projection" % name.upper())
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, out)


def fig_scatter_jaccard(pair_tables, out="fig6_cosine_vs_go_jaccard.png"):
    """任务 1：嵌入余弦相似度 vs GO-BP Jaccard 散点。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    primary = [(name, pair_tables[name]) for name in ["esm", "biobert"] if name in pair_tables]
    for ax, (name, (pts, both, cos, jac)) in zip(axes, primary):
        rng = np.random.default_rng(0)
        n = min(4000, len(cos))
        idx = rng.choice(len(cos), n, replace=False)
        ax.scatter(cos[idx], jac[idx], s=4, alpha=0.35, c="#4C72B0")
        ax.set_xlabel("embedding cosine similarity")
        ax.set_ylabel("GO-BP Jaccard similarity")
        rho = M.spearman(cos, jac)
        ax.set_title("%s (Spearman rho=%.3f)" % (name.upper(), rho))
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Task 1: embedding similarity vs functional annotation similarity")
    _save(fig, out)


def fig_public_benchmark(df2, out="fig7_public_benchmark_auc.png"):
    """公开基准（Open Targets）：嵌入方法 vs 网络/随机基线的 AUC 对比。"""
    if df2 is None or df2.empty or "auc" not in df2.columns:
        return
    df = df2[df2["auc"].notna() & df2["method"].isin([
        "centroid", "network-baseline", "random-expectation"])].copy()
    df["label_en"] = df["disease_label"].map(_en_label)
    df["series"] = df["name"] + "/" + df["method"]
    pivot = df.pivot_table(index="label_en", columns="series",
                           values="auc", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(11, 5.5))
    ax.set_ylabel("AUC")
    ax.set_xlabel("Disease")
    ax.set_title("Open Targets: complete candidate annotations, fold-mean AUC")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.34))
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=9)
    fig = ax.get_figure()
    _save(fig, out)


def fig_sensitivity(out="fig8_threshold_sensitivity.png"):
    import pandas as pd
    path = os.path.join(config.RESULTS_DIR, "task2_threshold_sensitivity.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = df[df.method.isin(["centroid", "network-baseline", "random-expectation"])]
    for name, group in sub.groupby("model"):
        mean = group.groupby("threshold").auc.mean()
        axes[0].plot(mean.index, mean.values, marker="o", label=name)
    axes[0].set(xlabel="Open Targets score threshold", ylabel="Disease-macro mean AUC",
                ylim=(0, 1), title="Exploratory threshold sensitivity")
    axes[0].legend(fontsize=8)
    one = df[df.model.eq("random")]
    for key, group in one.groupby("disease"):
        axes[1].plot(group.threshold, group.n_background, marker="o", label=key.replace("_", " "))
    axes[1].set(xlabel="Open Targets score threshold", ylabel="Unlabelled background count",
                title="Threshold changes candidate difficulty")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def make_all_figures(results, gene_embs, gene_info, bench2=None):
    if "esm" in gene_embs:
        fig_heatmap(gene_embs, config.ALL_GENES, name="esm")
    fig_roc(results["task1_pairs"])
    fig_disease_auc(results["task2"])
    fig_drug_targets(results["task3"])
    if "esm" in gene_embs:
        fig_pca(gene_embs, name="esm")
    fig_scatter_jaccard(results["task1_pairs"])
    fig_public_benchmark(bench2)
    fig_sensitivity()
