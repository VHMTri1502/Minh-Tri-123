"""Meaningful CPU checks for data leakage, scoring and the measured reference artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from masheart.baseline import make_classifier
from masheart.data import (assert_disjoint, compose_text, load_frame, public_payload,
                           split_development, verify_sources)
from masheart.metrics import evaluate, paired_cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = load_frame(ROOT / "data" / "raw" / "a_dev.jsonl", "dev")

    def test_preserved_source_hashes_and_default_group_split(self):
        self.assertTrue(verify_sources(ROOT / "data")["passed"])
        train, validation = split_development(self.dev)
        self.assertEqual((len(train), len(validation)), (95, 30))
        self.assertEqual((train.group_id.nunique(), validation.group_id.nunique()), (8, 3))
        self.assertTrue(assert_disjoint(train, validation)["passed"])

    def test_public_input_does_not_depend_on_gold_or_source_identity(self):
        row = self.dev.iloc[0].to_dict()
        altered = dict(row, label="NO" if row["label"] == "YS" else "YS", pmid="LEAK", group_id="LEAK")
        self.assertEqual(public_payload(row), public_payload(altered))
        self.assertEqual(set(public_payload(row)), {"question", "text"})
        with self.assertRaises(ValueError):
            public_payload(dict(row, text="   "))

    def test_source_overlap_is_rejected_even_with_new_id(self):
        left = self.dev.iloc[[0]].copy()
        right = left.copy()
        right["id"], right["group_id"], right["pmid"] = "new-id", "new-group", "new-pmid"
        right["text"] = right.text.str.upper().map(lambda value: "  " + value + "  ")
        with self.assertRaises(ValueError):
            assert_disjoint(left, right)

    def test_manifest_path_escape_is_rejected(self):
        scratch = ROOT / ".verification"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            target = Path(directory)
            manifest = {"schema_version": 1, "artifacts": [
                {"path": "../escape.jsonl", "sha256": "0" * 64, "bytes": 0}]}
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_sources(target)

    def test_test_cannot_be_supplied_as_training_development(self):
        forged = self.dev.copy()
        forged["split"] = "test"
        with self.assertRaises(ValueError):
            split_development(forged)

    def test_transform_does_not_learn_held_out_vocabulary(self):
        classifier = make_classifier()
        classifier.fit(["the synthetic result improved", "the synthetic result did not improve"], ["YS", "NO"])
        vectorizer = classifier["features"].transformer_list[0][1]
        vocabulary = copy.deepcopy(vectorizer.vocabulary_)
        classifier.predict(["never_seen_quasar_identifier"])
        self.assertEqual(vectorizer.vocabulary_, vocabulary)
        self.assertNotIn("never_seen_quasar_identifier", vectorizer.vocabulary_)


class MetricTests(unittest.TestCase):
    def test_invalid_outputs_remain_errors_in_the_denominator(self):
        scores = evaluate(["NO", "NO", "YS", "YS"], ["NO", "INVALID", "YS", "NO"], "test")
        self.assertEqual(scores["n"], 4)
        self.assertEqual(scores["accuracy"], .5)
        self.assertEqual(scores["invalid_count"], 1)
        self.assertEqual(scores["recall_NO"], .5)
        self.assertEqual(scores["valid_rate"], .75)

    def test_majority_can_be_accurate_and_miss_the_minority(self):
        scores = evaluate(["NO", "YS", "YS", "YS"], ["YS"] * 4, "majority")
        self.assertEqual(scores["accuracy"], .75)
        self.assertEqual(scores["recall_NO"], 0)
        self.assertLess(scores["macro_f1"], .5)

    def test_paired_bootstrap_identical_systems_have_zero_difference(self):
        gold = ["NO", "YS", "NO", "YS"]
        scores = paired_cluster_bootstrap(gold, gold, gold, ["a", "a", "b", "b"], rounds=20)
        np.testing.assert_array_equal(scores["differences"], np.zeros(20))
        self.assertEqual(scores["ci_lower"], 0)
        self.assertEqual(scores["ci_upper"], 0)

    def test_invalid_metrics_inputs_are_rejected(self):
        for gold, pred in [([], []), (["NO"], []), (["TRUE"], ["YS"]), (["NO"], ["UNKNOWN"])]:
            with self.assertRaises(ValueError):
                evaluate(gold, pred, "invalid")


class ReferenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_dir = ROOT / "results" / "cpu_reference"

    def test_reference_predictions_are_complete_and_metrics_recalculate(self):
        predictions = pd.read_csv(self.reference_dir / "tables" / "test_predictions.csv")
        metrics = pd.read_csv(self.reference_dir / "tables" / "test_metrics.csv").set_index("system")
        self.assertEqual(len(predictions), 134)
        self.assertTrue(predictions.id.is_unique)
        for name in ("majority", "tfidf_logreg"):
            measured = evaluate(predictions.gold, predictions[name], name)
            self.assertAlmostEqual(measured["macro_f1"], metrics.loc[name, "macro_f1"], places=12)
            self.assertAlmostEqual(measured["accuracy"], metrics.loc[name, "accuracy"], places=12)

    def test_frozen_training_ids_and_loaded_model_reproduce_all_predictions(self):
        frozen = json.loads((self.reference_dir / "reports" / "frozen_protocol.json").read_text())
        self.assertEqual(len(frozen["train_ids"]), 95)
        self.assertEqual(len(frozen["validation_ids"]), 30)
        self.assertFalse(set(frozen["train_ids"]) & set(frozen["validation_ids"]))
        self.assertTrue(frozen["frozen_before_test_records_loaded"])
        model = joblib.load(self.reference_dir / "models" / "tfidf_logreg.joblib")
        predictions = pd.read_csv(self.reference_dir / "tables" / "test_predictions.csv")
        texts = [compose_text(row) for row in predictions.to_dict("records")]
        self.assertEqual(model.predict(texts).tolist(), predictions.tfidf_logreg.tolist())

    def test_measured_figure_sources_and_files_match_their_hashes(self):
        manifest = json.loads((self.reference_dir / "figures" / "figure_data_manifest.json").read_text())
        self.assertEqual(len(manifest["figures"]), 8)
        for figure in manifest["figures"]:
            source = self.reference_dir / "figures" / figure["data"]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), figure["data"]["sha256"])
            for artifact in figure["files"]:
                path = self.reference_dir / "figures" / artifact["path"]
                self.assertGreater(path.stat().st_size, 100)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
