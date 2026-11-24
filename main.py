# -*- coding: utf-8 -*-
"""Portable entry point; --offline reuses verified local inputs without HTTP calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# Optional legacy vendor dependencies are used only when present.
if os.path.isdir(os.path.join(ROOT, "vendor", "transformers")):
    sys.path.insert(0, os.path.join(ROOT, "vendor"))

import numpy as np
import pandas as pd
import config
from src.utils import ensure_dir, write_json

STEPS_ALL = ["data", "embeddings", "tasks", "benchmark", "viz", "report"]


def environment_record():
    packages = {}
    for name in ["numpy", "pandas", "scipy", "matplotlib", "torch", "transformers", "tokenizers", "regex"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    sources = ["main.py", "config.py"]
    sources += [os.path.join(folder, f) for folder in ["src", "scripts", "tests"]
                for f in os.listdir(os.path.join(ROOT, folder)) if f.endswith(".py")]
    fingerprints = {}
    for path in sources:
        with open(os.path.join(ROOT, path), "rb") as f:
            fingerprints[path.replace(os.sep, "/")] = hashlib.sha256(f.read()).hexdigest()
    return dict(python=sys.version, platform=platform.platform(), packages=packages, source_sha256=fingerprints,
                execution_date=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                protocol="leave-one-positive-out_seed-excluded_v2")


def build_summary(results):
    t2, t3 = results["task2"], results["task3"]
    return dict(protocol=results["protocol"], task1=results["task1"].to_dict("records"),
                task2_centroid_mean_auc=t2[t2.method == "centroid"].groupby("model").auc.mean().to_dict(),
                task2_query_mean_auc=t2[t2.method == "mechanism-query"].groupby("model").auc.mean().to_dict(),
                task3_query_mean_auc=t3[t3.method.isin(["mechanism-query", "indication-query"])].groupby(
                    ["model", "method"]).auc.mean().rename_axis(["model", "method"]).reset_index().to_dict("records"))


def load_results():
    from src import metrics as M
    from src.utils import read_json
    result = {"task%d" % n: pd.read_csv(os.path.join(config.RESULTS_DIR, f)) for n, f in [
        (1, "task1_functional_similarity.csv"), (2, "task2_disease_genes.csv"), (3, "task3_drug_targets.csv")]}
    with np.load(os.path.join(config.RESULTS_DIR, "task1_pair_data.npz")) as z:
        pairs = {}
        for name in result["task1"].model:
            cos, jac = z[name + "_cos"], z[name + "_jac"]
            pairs[name] = (M.roc_curve_points((jac > 0).astype(float), cos), None, cos, jac)
    result["task1_pairs"] = pairs
    result["task2_detail"] = read_json(os.path.join(config.RESULTS_DIR, "task2_detail.json"))
    result["task3_detail"] = read_json(os.path.join(config.RESULTS_DIR, "task3_detail.json"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Frozen biomedical representations: reproducible coursework")
    parser.add_argument("--steps", default="all", help=",".join(STEPS_ALL) + " or all")
    parser.add_argument("--models", default="esm,biobert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="fail instead of requesting missing online data")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-emb", action="store_true")
    parser.add_argument("--force-bench", action="store_true")
    args = parser.parse_args()
    steps = STEPS_ALL if args.steps == "all" else [s.strip() for s in args.steps.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not steps or set(steps) - set(STEPS_ALL):
        parser.error("unknown or empty steps")
    if not models or set(models) - set(config.MODEL_IDS):
        parser.error("models must be esm, biobert, or both")
    if args.threads < 1:
        parser.error("threads must be positive")
    if "report" in steps and set(models) != set(config.MODEL_IDS):
        parser.error("complete report requires both esm and biobert; use --steps embeddings,tasks,benchmark for a single model")
    if args.offline and (args.force_data or args.force_bench):
        parser.error("offline cannot force online data refresh")
    ensure_dir(config.RESULTS_DIR)
    from src import data_fetch, benchmark
    data_fetch.OFFLINE = benchmark.OFFLINE = args.offline
    if "data" in steps or set(steps) & {"embeddings", "tasks", "benchmark", "viz"}:
        gene_info, seqs = data_fetch.ensure_data(force=args.force_data)
    if set(steps) & {"embeddings", "tasks", "benchmark", "viz"}:
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(config.RANDOM_SEED)
        from src import embeddings
        gene_embs, queries = embeddings.load_all_embeddings(
            gene_info, seqs, models, args.device, force=args.force_emb)
    if set(steps) & {"tasks", "benchmark"}:
        from src.baselines import build_representations
        reps, tfidf, order = build_representations(gene_embs, gene_info, seqs, config.ALL_GENES)
    results = None
    if "tasks" in steps:
        from src import tasks
        results = tasks.run_all_tasks(reps, queries, gene_info, tfidf, order)
        write_json(os.path.join(config.RESULTS_DIR, "summary.json"), build_summary(results))
    bench2 = None
    if "benchmark" in steps:
        bench2, _ = benchmark.run_public_benchmark(reps, queries, gene_info, tfidf, order, args.force_bench)
    if "viz" in steps:
        from src import visualize
        results = results or load_results()
        path = os.path.join(config.RESULTS_DIR, "task2_public_benchmark.csv")
        if bench2 is None and os.path.exists(path):
            bench2 = pd.read_csv(path)
        visualize.make_all_figures(results, gene_embs, gene_info, bench2)
    # A documentation-only run must not replace the environment of the actual experiment.
    if set(steps) & {"embeddings", "tasks", "benchmark"}:
        record = environment_record()
        record.update(device=args.device, threads=args.threads, offline=args.offline)
        write_json(os.path.join(config.RESULTS_DIR, "environment.json"), record)
    if "report" in steps:
        from scripts.build_report import build_report
        build_report()
    print("[main] completed " + ",".join(steps), flush=True)


if __name__ == "__main__":
    main()
