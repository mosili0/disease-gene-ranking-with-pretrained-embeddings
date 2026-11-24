"""Audit saved folds, source coverage and report image references independently of the run."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src import metrics as M


def verify():
    checked_folds, checked_entities = 0, 0
    for filename in ["task2_detail.json", "task3_detail.json", "task2_public_detail.json", "task3_public_detail.json"]:
        data = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
        for entity, entry in data.items():
            reference = None
            positives = set(entry["positives"])
            for method, folds in entry.get("methods", {}).items():
                current = {}
                for fold in folds:
                    held, seeds, candidates = fold["held_out"], set(fold["seeds"]), fold["candidates"]
                    assert held not in seeds and held in positives
                    assert not seeds.intersection(candidates)
                    assert seeds == positives - {held}
                    assert set(candidates).intersection(positives) == {held}
                    assert len(candidates) == len(set(candidates)) == fold["n_candidates"]
                    assert set(candidates) | seeds == set(config.ALL_GENES)
                    scores = np.array(fold["scores"])
                    y = np.array([g == held for g in candidates])
                    assert np.isfinite(scores).all()
                    # Direct pairwise AUC is an independent oracle for the saved metric.
                    p, bg = scores[y], scores[~y]
                    oracle = float(np.mean((p[:, None] > bg) + .5 * (p[:, None] == bg)))
                    assert abs(oracle - fold["auc"]) < 1e-12
                    assert abs(M.hit_at_k(scores, np.flatnonzero(y), 10) - fold["hit_at_10"]) < 1e-12
                    current[held] = (tuple(sorted(seeds)), tuple(candidates))
                    checked_folds += 1
                if reference is None:
                    reference = current
                else:
                    assert current == reference, (entity, method, "unequal comparison folds")
            checked_entities += 1
    for key in config.DISEASES:
        d = json.loads((ROOT / "data/benchmark" / ("open_targets_" + key + ".json")).read_text(encoding="utf-8"))
        assert d["schema_version"] == 3 and d["complete"]
        assert d["universe"] == config.ALL_GENES
        assert len(d["associations"]) == d["candidate_associated"]
        assert len({r["target_id"] for r in d["associations"]}) == d["candidate_associated"]
    report = (ROOT / "实验报告.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert images and all((ROOT / f).is_file() for f in images)
    metrics = {}
    for path in sorted((ROOT / "results").glob("*.csv")):
        metrics[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = dict(status="passed", checked_entities=checked_entities, checked_method_folds=checked_folds,
               covered_candidate_genes=len(config.ALL_GENES), checked_report_images=len(images),
               result_csv_sha256=metrics,
               note="Separate from 10 unit regression tests; no network requests or new model inference.")
    (ROOT / "results/verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ["result_csv_sha256", "note"]}))


if __name__ == "__main__":
    verify()
