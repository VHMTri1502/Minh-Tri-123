"""Trainable, leakage-resistant CPU comparators for the classroom research protocol."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make_classifier(config: dict | None = None) -> Pipeline:
    """Build word(1,2)+char_wb(3,5) TF-IDF and balanced logistic regression.

    Call fit only after group splitting. The Pipeline learns its vocabulary,
    IDF and classifier weights from the same permitted training examples.
    config accepts C, max_iter and seed; all other architecture choices are fixed.
    """
    config = dict(config or {})
    unknown = set(config) - {"C", "max_iter", "seed"}
    if unknown:
        raise ValueError(f"Unknown classifier settings: {sorted(unknown)}")
    strength = float(config.get("C", 1.0))
    iterations = config.get("max_iter", 2000)
    seed = config.get("seed", 17)
    if strength <= 0 or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("C and max_iter must be positive.")
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=None)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(
            C=strength, class_weight="balanced", max_iter=iterations, random_state=seed,
        )),
    ])


def make_dummy_classifier() -> DummyClassifier:
    """Build a majority-label control whose chosen class is learned from training labels only."""
    return DummyClassifier(strategy="most_frequent")
