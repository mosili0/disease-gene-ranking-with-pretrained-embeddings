"""Regression tests for the evaluation errors found during review."""
import itertools
import unittest
from unittest.mock import patch
import tempfile
import os
import json
import numpy as np
from scipy.integrate import trapezoid
from src import metrics as M
from src.evaluation import evaluate, centroid_scorer, random_expectation
from src.embeddings import chunks
from src.baselines import Tfidf


class MetricTests(unittest.TestCase):
    def test_all_ties_are_chance_regardless_of_order(self):
        for y in set(itertools.permutations([0, 0, 1, 1])):
            self.assertEqual(M.roc_auc(y, [0] * 4), 0.5)
            self.assertEqual(M.average_precision(y, [0] * 4), 0.5)

    def test_auc_matches_pairwise_definition_with_ties(self):
        rng = np.random.default_rng(123)
        for _ in range(50):
            y = np.array([1] * 5 + [0] * 12)
            s = rng.integers(0, 4, len(y))
            expected = np.mean([float(p > n) + .5 * (p == n) for p in s[y == 1] for n in s[y == 0]])
            self.assertAlmostEqual(M.roc_auc(y, s), expected)
            fpr, tpr = M.roc_curve_points(y, s)
            self.assertAlmostEqual(float(trapezoid(tpr, fpr)), expected)

    def test_tied_boundary_has_fractional_hit(self):
        self.assertEqual(M.hit_at_k([1] * 20, [0], 10), .5)
        self.assertEqual(M.ranks_of([1] * 20, [0]), [10.5])

    def test_known_retrieval_metrics(self):
        self.assertEqual(M.roc_auc([1, 0, 0], [3, 2, 1]), 1.)
        self.assertEqual(M.average_precision([0, 1, 0, 1], [4, 3, 2, 1]), (1/2 + 2/4)/2)
        self.assertAlmostEqual(M.mrr([1, 1], [0]), .75)


class ProtocolTests(unittest.TestCase):
    def test_same_query_and_no_seed_candidates(self):
        X = np.eye(4)
        called = []
        def scorer(seeds):
            called.append(seeds)
            return centroid_scorer(X)(seeds)
        summary, folds = evaluate(["a", "b", "c", "d"], ["a", "b"], scorer)
        self.assertEqual(called, [[1], [0]])
        # Orthogonal held-out positive and backgrounds tie at 0; self-seeds excluded.
        self.assertEqual(summary["auc"], .5)
        for f in folds:
            self.assertFalse(set(f["seeds"]) & set(f["candidates"]))
            self.assertNotIn(f["held_out"], f["seeds"])
            self.assertEqual(f["n_candidates"], 3)

    def test_fold_auc_not_pooled_across_queries(self):
        def scorer(seeds):
            # Both held-out positives beat the background in THEIR fold, despite different scales.
            return np.array([100, 200, 99]) if seeds == [1] else np.array([0, 1, 0])
        summary, _ = evaluate(["a", "b", "c"], ["a", "b"], scorer)
        self.assertEqual(summary["auc"], 1.)

    def test_exact_random_baseline(self):
        r = random_expectation(20)
        self.assertEqual(r["auc"], .5)
        self.assertEqual(r["hit_at_10"], .5)
        self.assertEqual(r["median_rank"], 10.5)

    def test_full_sequence_chunks(self):
        seq = "A" * 3418
        parts = chunks(seq)
        self.assertEqual("".join(parts), seq)
        self.assertTrue(all(len(p) <= 1022 for p in parts))

    def test_tfidf_queries_cannot_change_corpus(self):
        tfidf = Tfidf(["alpha beta", "beta gamma"])
        before = tfidf.X.copy()
        q = tfidf.transform(["unknownword alpha"])
        np.testing.assert_array_equal(before, tfidf.X)
        self.assertNotIn("unknownword", tfidf.vocab)
        self.assertEqual(q.shape[1], len(tfidf.vocab))

    def test_legacy_top400_not_accepted_as_complete(self):
        from src import benchmark as B
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
            cache = os.path.join(tmp, "open_targets_breast_cancer.json")
            with open(cache, "w") as f:
                json.dump({"top400": [], "fetch_date": "2026-11-02"}, f)
            with patch.object(B, "BENCH_DIR", tmp), patch.object(B, "OFFLINE", True):
                with self.assertRaises(RuntimeError):
                    B.fetch_open_targets("breast_cancer", mapping={g: {"id": g} for g in B.config.ALL_GENES})


if __name__ == "__main__":
    unittest.main()
