"""Run the complete, measured CPU research workflow without external services."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .baseline import make_classifier, make_dummy_classifier
from .data import (LABELS, assert_disjoint, compose_text, file_sha256, load_frame,
                   public_payload, split_development, verify_sources)
from .metrics import evaluate, paired_cluster_bootstrap
from .visualization import generate_research_figures

DEFAULT_CONFIG = {"seed": 17, "validation_fraction": 0.25, "C": 1.0,
                  "max_iter": 2000, "bootstrap_rounds": 500}
SYSTEMS = ("majority", "tfidf_logreg")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"):
        versions[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "execution": "CPU only; no model downloads or API calls"}


def _split_row(name: str, frame: pd.DataFrame) -> dict:
    return {"split": name, "n": len(frame), "groups": frame["group_id"].nunique(),
            **{label: int(frame["label"].eq(label).sum()) for label in LABELS}}


def _predictions(frame: pd.DataFrame, classifier, dummy) -> pd.DataFrame:
    """Generate before attaching gold to the result table; only two public fields enter the model."""
    sources = [public_payload(row) for row in frame.to_dict("records")]
    texts = [compose_text(source) for source in sources]
    predicted = classifier.predict(texts)
    majority = dummy.predict(np.zeros((len(sources), 1)))
    probabilities = classifier.predict_proba(texts)
    table = frame[["id", "group_id", "pmid", "question", "text", "label"]].rename(columns={"label": "gold"}).copy()
    table["majority"] = majority
    table["tfidf_logreg"] = predicted
    for index, label in enumerate(classifier.classes_):
        table[f"score_{label}_uncalibrated"] = probabilities[:, index]
    return table


def _score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([evaluate(predictions["gold"], predictions[name], name) for name in SYSTEMS])


def _group_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in predictions.groupby("group_id", sort=True):
        for system in SYSTEMS:
            rows.append({"group_id": group_id, **evaluate(group["gold"], group[system], system)})
    return pd.DataFrame(rows)


def run_cpu_experiment(data_dir: Path, output_dir: Path, config: dict | None = None) -> dict:
    """Train, freeze, test, evaluate, visualize and export one complete educational CPU run.

    Existing nonempty output directories are refused. Test bytes may be hashed
    during source verification, but test records and labels are not parsed until
    frozen_protocol.json and its hash have been written to disk. There is no
    automatic resume, hyperparameter search, GPU execution or test-driven tuning.
    """
    started = time.perf_counter()
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    settings = {**DEFAULT_CONFIG, **supplied}
    if not isinstance(settings["seed"], int) or isinstance(settings["seed"], bool):
        raise ValueError("seed must be an integer.")
    if (not isinstance(settings["bootstrap_rounds"], int) or isinstance(settings["bootstrap_rounds"], bool)
            or settings["bootstrap_rounds"] <= 0):
        raise ValueError("bootstrap_rounds must be a positive integer.")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Choose a new or empty output directory; completed and partial runs are preserved.")
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = {name: output_dir / name for name in ("tables", "figures", "models", "reports")}
    for directory in directories.values():
        directory.mkdir()
    tables, reports, models = directories["tables"], directories["reports"], directories["models"]

    source_audit = verify_sources(data_dir)
    _write_json(reports / "source_audit.json", source_audit)
    environment = _environment()
    _write_json(reports / "environment.json", environment)
    _write_json(reports / "config.json", settings)
    dev = load_frame(data_dir / "raw" / "a_dev.jsonl", "dev")
    train, validation = split_development(dev, settings["seed"], settings["validation_fraction"])
    train_validation_audit = assert_disjoint(train, validation)
    split_manifest = {
        "protocol": "educational-development-fit-group-split",
        "seed": settings["seed"], "validation_fraction": settings["validation_fraction"],
        "train_ids": train["id"].tolist(), "validation_ids": validation["id"].tolist(),
        "train_groups": sorted(train["group_id"].unique()),
        "validation_groups": sorted(validation["group_id"].unique()),
        "train_validation_audit": train_validation_audit,
        "official_test_policy": "Preserve released test membership; do not parse records before freezing.",
    }
    _write_json(reports / "split_manifest.json", split_manifest)
    classifier = make_classifier({"C": settings["C"], "max_iter": settings["max_iter"], "seed": settings["seed"]})
    dummy = make_dummy_classifier()
    train_texts = [compose_text(row) for row in train.to_dict("records")]
    training_started = time.perf_counter()
    classifier.fit(train_texts, train["label"])
    dummy.fit(np.zeros((len(train), 1)), train["label"])
    training_seconds = time.perf_counter() - training_started
    validation_predictions = _predictions(validation, classifier, dummy)
    validation_metrics = _score_table(validation_predictions)
    validation_predictions.to_csv(tables / "validation_predictions.csv", index=False)
    validation_metrics.to_csv(tables / "validation_metrics.csv", index=False)

    model_path, dummy_path = models / "tfidf_logreg.joblib", models / "majority.joblib"
    joblib.dump(classifier, model_path)
    joblib.dump(dummy, dummy_path)
    restored = joblib.load(model_path)
    validation_texts = [compose_text(row) for row in validation.to_dict("records")]
    if restored.predict(validation_texts).tolist() != validation_predictions["tfidf_logreg"].tolist():
        raise RuntimeError("Serialized model predictions differ from the trained model.")
    restored_dummy = joblib.load(dummy_path)
    if restored_dummy.predict(np.zeros((len(validation), 1))).tolist() != validation_predictions["majority"].tolist():
        raise RuntimeError("Serialized majority control changed its predictions.")
    model_hash = joblib.hash(classifier)
    dummy_hash = joblib.hash(dummy)
    module_hashes = {name: file_sha256(Path(__file__).with_name(name)) for name in
                     ("data.py", "metrics.py", "baseline.py", "visualization.py", "experiment.py")}
    frozen_protocol = {
        "schema_version": 1, "protocol": "MASHeart educational CPU development-fit v1",
        "config": settings, "config_sha256": _json_hash(settings),
        "source_manifest_sha256": source_audit["manifest_sha256"],
        "source_artifacts": source_audit["artifacts"],
        "split_manifest_sha256": file_sha256(reports / "split_manifest.json"),
        "train_ids": split_manifest["train_ids"], "validation_ids": split_manifest["validation_ids"],
        "model_joblib_hash": model_hash, "dummy_joblib_hash": dummy_hash,
        "model_file_sha256": file_sha256(model_path), "dummy_file_sha256": file_sha256(dummy_path),
        "environment": environment, "module_sha256": module_hashes,
        "selection": "Fixed CPU hyperparameters; no search, no refit on full development, no use of test labels.",
        "frozen_before_test_records_loaded": True,
        "history": "Original HeartMAS developed on dev and an earlier v1 encountered test; this is not a historically pristine holdout.",
    }
    frozen_path = reports / "frozen_protocol.json"
    _write_json(frozen_path, frozen_protocol)
    frozen_sha256 = file_sha256(frozen_path)
    (reports / "frozen_protocol.sha256").write_text(frozen_sha256 + "\n", encoding="utf-8")

    # The first interpretation of official test records occurs after the on-disk freeze above.
    if joblib.hash(classifier) != model_hash or joblib.hash(dummy) != dummy_hash:
        raise RuntimeError("A model changed after configuration freeze.")
    if (file_sha256(model_path) != frozen_protocol["model_file_sha256"]
            or file_sha256(dummy_path) != frozen_protocol["dummy_file_sha256"]):
        raise RuntimeError("A serialized model changed after configuration freeze.")
    if file_sha256(frozen_path) != frozen_sha256 or verify_sources(data_dir) != source_audit:
        raise RuntimeError("The frozen protocol or preserved source files changed before test.")
    test = load_frame(data_dir / "raw" / "a_test.jsonl", "test")
    leakage_audit = {
        "train_validation": train_validation_audit,
        "train_test": assert_disjoint(train, test),
        "validation_test": assert_disjoint(validation, test),
    }
    _write_json(reports / "leakage_audit.json", leakage_audit)
    split_table = pd.DataFrame([_split_row("train", train), _split_row("validation", validation), _split_row("test", test)])
    split_table.to_csv(tables / "split_summary.csv", index=False)
    test_predictions = _predictions(test, classifier, dummy)
    test_metrics = _score_table(test_predictions)
    test_predictions.to_csv(tables / "test_predictions.csv", index=False)
    test_metrics.to_csv(tables / "test_metrics.csv", index=False)
    group_scores = _group_scores(test_predictions)
    group_scores.to_csv(tables / "test_group_scores.csv", index=False)
    errors = test_predictions.loc[~test_predictions["gold"].eq(test_predictions["tfidf_logreg"])].copy()
    errors.to_csv(tables / "test_errors.csv", index=False)

    class_rows, confusion_rows = [], []
    for name in SYSTEMS:
        report = classification_report(test_predictions["gold"], test_predictions[name],
                                       labels=list(LABELS), output_dict=True, zero_division=0)
        _write_json(reports / f"classification_report_{name}.json", report)
        for label in LABELS:
            class_rows.append({"system": name, "label": label, **report[label]})
        order = list(LABELS) + (["INVALID"] if test_predictions[name].eq("INVALID").any() else [])
        matrix = confusion_matrix(test_predictions["gold"], test_predictions[name], labels=order)
        confusion_rows.extend({"system": name, "gold": gold, "prediction": predicted,
                               "count": int(matrix[i, j])}
                              for i, gold in enumerate(order) for j, predicted in enumerate(order))
    pd.DataFrame(class_rows).to_csv(tables / "test_per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(tables / "test_confusion_counts.csv", index=False)
    bootstrap = paired_cluster_bootstrap(test_predictions["gold"], test_predictions["majority"],
                                         test_predictions["tfidf_logreg"], test_predictions["group_id"],
                                         rounds=settings["bootstrap_rounds"], seed=settings["seed"])
    bootstrap_summary = {key: value for key, value in bootstrap.items() if key != "differences"}
    bootstrap_summary.update({"system_a": "majority", "system_b": "tfidf_logreg"})
    _write_json(reports / "bootstrap_summary.json", bootstrap_summary)
    pd.DataFrame({"draw": np.arange(1, len(bootstrap["differences"]) + 1),
                  "macro_f1_difference": bootstrap["differences"]}).to_csv(tables / "bootstrap_differences.csv", index=False)
    figures = generate_research_figures(dev, split_table, test_predictions, test_metrics,
                                        group_scores, bootstrap, directories["figures"], tables)

    if joblib.hash(classifier) != model_hash or joblib.hash(dummy) != dummy_hash:
        raise RuntimeError("A frozen model changed during evaluation.")
    original = train.iloc[0].to_dict()
    altered = dict(original, label="NO" if original["label"] == "YS" else "YS", pmid="EXCLUDED")
    if public_payload(original) != public_payload(altered):
        raise RuntimeError("Gold or source metadata crossed the inference boundary.")
    report_path = reports / "CPU_RESEARCH_REPORT.md"
    model_metrics = test_metrics.set_index("system")
    majority_row, baseline_row = model_metrics.loc["majority"], model_metrics.loc["tfidf_logreg"]
    interpretation = ("The percentile interval includes zero, so this resampling comparison does not establish a clear improvement."
                      if bootstrap_summary["ci_lower"] <= 0 <= bootstrap_summary["ci_upper"] else
                      "The percentile interval excludes zero under this resampling scheme; the small number of groups and prior data exposure still limit the claim.")
    metric_lines = "\n".join(
        f"| {row['system']} | {row['n']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['recall_NO']:.4f} | {row['recall_YS']:.4f} | {row['invalid_count']} |"
        for row in test_metrics.to_dict("records"))
    report_path.write_text(f"""# MASHeart CPU research report

This report records an executed educational Task-A experiment. It predicts the stance of a supplied English statement toward a supplied research question. YS means affirmation; NO means negation or lack of affirmation. Neither label establishes clinical truth, diagnostic accuracy or treatment quality.

## Data and protocol

PCC statements and stance labels originate with Abdulaziz Alamri and Mark Stevenson (2016). Official development/test membership follows the linked BioClaim release preserved in the data manifest. The original release has no independent training set. This run uses {len(train)} training examples in {train.group_id.nunique()} groups and {len(validation)} validation examples in {validation.group_id.nunique()} groups, both drawn only from official development; all {len(test)} official test examples in {test.group_id.nunique()} groups are retained.

The training and validation split is by linked source group with seed {settings['seed']}. ID, group, PMID and normalized-statement intersections are empty across these partitions and test. The complete source audit and split IDs are exported. The original project used development data during method design and an earlier v1 encountered test; repackaging does not create a historically unseen holdout.

Only question and statement are model features. Word TF-IDF (1–2 grams) and character-within-word TF-IDF (3–5 grams) are fitted inside a Pipeline on training only. Balanced logistic regression uses C={settings['C']} and max_iter={settings['max_iter']}. The majority control learns its class from training labels only. There is no hyperparameter search or full-development refit. Both serialized models were reloaded and checked on validation. Configuration, source checksums, model hashes, environment and split IDs were written before official test records were parsed.

Frozen protocol SHA-256: `{frozen_sha256}`.

## Complete-test results

| System | n | Accuracy | Macro-F1 | Recall NO | Recall YS | Invalid |
|---|---:|---:|---:|---:|---:|---:|
{metric_lines}

Accuracy uses every example. Macro-F1 always averages NO and YS. INVALID outputs, if present, count as errors rather than being discarded. Per-class precision, recall, F1 and support, confusion counts, predictions with gold, source-group scores and all {len(errors)} TF-IDF errors are available in tables. Scores from a different Qwen model or from the original HeartMAS study are not used here.

## Uncertainty and interpretation

The paired bootstrap resamples whole test source groups, gives both systems identical sampled rows and retains both labels in every macro average. Across {settings['bootstrap_rounds']} draws, the observed macro-F1 difference (TF-IDF minus majority) is {bootstrap_summary['point_difference']:.4f}, with a 95% percentile interval [{bootstrap_summary['ci_lower']:.4f}, {bootstrap_summary['ci_upper']:.4f}]. {interpretation}

The majority control's test accuracy is {majority_row['accuracy']:.4f}; its minority-class recall must be considered alongside that score. The trained baseline achieves accuracy {baseline_row['accuracy']:.4f} and macro-F1 {baseline_row['macro_f1']:.4f}. Question wording, negation scope, population, intervention, outcome and study conditions can all matter; a word's fitted coefficient or an uncalibrated score is not medical evidence.

## Artifacts and limits

Eight measured figures are exported as 300-dpi PNG and vector SVG, with exact CSV inputs and hashes in `../figures/figure_data_manifest.json`. Model loading was verified; source and frozen model hashes are preserved. Environment versions and source-code hashes document the implementation used in this run. Training took {training_seconds:.3f} seconds in this environment; timing varies by machine.

This CPU workflow does not train a language model or execute the GPU MAS. It establishes measured controls for the separate QLoRA/MAS workflow. No GPU score, model-call cost, selective coverage or claimed MAS advantage is fabricated. Dataset size, class imbalance, related observations, limited source groups, historical data exposure and corpus annotation choices restrict generalization. The data license and upstream version discrepancy are documented in the data card. Use this experiment for education and text-method research, not treatment decisions.
""", encoding="utf-8")

    completed_artifacts = []
    for directory in (tables, models):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                completed_artifacts.append({"path": path.relative_to(output_dir).as_posix(),
                                            "sha256": file_sha256(path), "bytes": path.stat().st_size})
    for path in (frozen_path, reports / "split_manifest.json", reports / "bootstrap_summary.json",
                 reports / "environment.json", reports / "config.json", reports / "source_audit.json"):
        completed_artifacts.append({"path": path.relative_to(output_dir).as_posix(),
                                    "sha256": file_sha256(path), "bytes": path.stat().st_size})
    _write_json(reports / "complete_manifest.json", {"schema_version": 1, "status": "complete",
                "frozen_protocol_sha256": frozen_sha256, "artifacts": completed_artifacts})
    summary = {
        "status": "complete", "protocol": "educational-development-fit-group-split",
        "run_dir": str(output_dir), "config": settings,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split_counts": split_table.to_dict("records"),
        "validation_metrics": validation_metrics.to_dict("records"),
        "test_metrics": test_metrics.to_dict("records"), "bootstrap": bootstrap_summary,
        "source_audit_passed": True, "leakage_audit_passed": True,
        "model_roundtrip_verified": True, "test_loaded_after_freeze": True,
        "frozen_protocol_sha256": frozen_sha256, "figures": figures,
        "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
        "gpu_executed": False,
        "paths": {"tables": "tables", "figures": "figures", "models": "models", "reports": "reports",
                  "report": "reports/CPU_RESEARCH_REPORT.md", "frozen_protocol": "reports/frozen_protocol.json",
                  "complete_manifest": "reports/complete_manifest.json"},
    }
    _write_json(reports / "run_summary.json", summary)
    return summary
