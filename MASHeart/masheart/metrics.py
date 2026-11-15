"""Fixed-label metrics and paired source-group bootstrap for Task-A predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ("NO", "YS")
INVALID = "INVALID"


def _validated_predictions(gold, pred) -> tuple[np.ndarray, np.ndarray]:
    gold = np.asarray(list(gold), dtype=str)
    pred = np.asarray(list(pred), dtype=str)
    if gold.ndim != 1 or pred.ndim != 1 or len(gold) == 0 or len(gold) != len(pred):
        raise ValueError("Gold and predictions must be nonempty, equally sized one-dimensional sequences.")
    if not set(gold) <= set(LABELS) or not set(pred) <= set(LABELS + (INVALID,)):
        raise ValueError("Gold must be NO/YS; predictions must be NO/YS/INVALID.")
    return gold, pred


def evaluate(gold, pred, name: str) -> dict:
    """Score every example, counting INVALID as an error instead of dropping failed outputs."""
    gold, pred = _validated_predictions(gold, pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=list(LABELS), zero_division=0,
    )
    result = {
        "system": name, "n": int(len(gold)), "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(np.mean(f1)),
        "valid_rate": float(np.mean(np.isin(pred, LABELS))),
        "invalid_count": int(np.sum(pred == INVALID)),
    }
    for index, label in enumerate(LABELS):
        result.update({
            f"precision_{label}": float(precision[index]),
            f"recall_{label}": float(recall[index]),
            f"f1_{label}": float(f1[index]),
            f"support_{label}": int(support[index]),
        })
    return result


def paired_cluster_bootstrap(gold, pred_a, pred_b, groups, rounds: int = 500,
                             seed: int = 17) -> dict:
    """Resample whole source groups and estimate macro-F1(B)-macro-F1(A) on paired rows.

    NO and YS remain in the macro average even when a bootstrap draw has no
    examples of one class. The interval is a percentile description of this
    resampling scheme, not proof of independent samples or clinical utility.
    """
    gold, pred_a = _validated_predictions(gold, pred_a)
    _, pred_b = _validated_predictions(gold, pred_b)
    groups = np.asarray(list(groups), dtype=str)
    if groups.ndim != 1 or len(groups) != len(gold) or np.any(groups == ""):
        raise ValueError("Every paired example requires a nonempty source-group ID.")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("rounds must be a positive integer.")
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(rounds):
        sampled = random.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        a = f1_score(gold[selected], pred_a[selected], labels=list(LABELS),
                     average="macro", zero_division=0)
        b = f1_score(gold[selected], pred_b[selected], labels=list(LABELS),
                     average="macro", zero_division=0)
        differences.append(float(b - a))
    values = np.asarray(differences)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "difference_direction": "system_b_minus_system_a",
        "metric": "macro_f1", "rounds": rounds, "seed": int(seed),
        "group_count": int(len(unique_groups)), "n": int(len(gold)),
        "point_difference": evaluate(gold, pred_b, "b")["macro_f1"]
                            - evaluate(gold, pred_a, "a")["macro_f1"],
        "ci_lower": float(lower), "ci_upper": float(upper),
        "confidence_level": 0.95, "method": "paired source-group percentile bootstrap",
        "differences": values,
    }
