"""Create publication-sized figures and hash their exact tabular inputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .data import LABELS, file_sha256

COLORS = {"NO": "#c56c39", "YS": "#287da8", "majority": "#8b929a", "tfidf_logreg": "#287da8"}
DISPLAY_NAMES = {"majority": "Training-majority control", "tfidf_logreg": "TF-IDF + logistic regression"}


def generate_research_figures(dev_frame: pd.DataFrame, split_table: pd.DataFrame,
                              predictions: pd.DataFrame, test_metrics: pd.DataFrame,
                              group_scores: pd.DataFrame, bootstrap: dict,
                              output_dir: str | Path, tables_dir: str | Path) -> dict:
    """Save eight PNG/SVG figure pairs from actual measurements, never placeholder scores.

    Each plot has an exported CSV input and an entry in figure_data_manifest.json.
    Training and validation examples remain identifiable by the recorded split IDs;
    figures summarize the existing experiment and do not choose its configuration.
    """
    directory, tables = Path(output_dir), Path(tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "renderer": "matplotlib", "matplotlib_version": matplotlib.__version__,
        "style": {"dpi": 300, "font_family": "DejaVu Sans", "font_size": 10,
                  "formats": ["png", "svg"], "colors": COLORS},
        "figures": [],
    }

    def finish(figure, stem: str, table: pd.DataFrame, description: str):
        """Persist the measured inputs before recording hashes of data and rendered files."""
        data_path = tables / f"{stem}_data.csv"
        if data_path.exists():
            # Preserve the original CSV bytes when rendering an already completed run.
            pd.testing.assert_frame_equal(pd.read_csv(data_path), table.reset_index(drop=True),
                                          check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
        else:
            table.to_csv(data_path, index=False)
        figure.tight_layout()
        files = []
        for extension in ("png", "svg"):
            path = directory / f"{stem}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight",
                           metadata={"Creator": "MASHeart", "Date": None} if extension == "svg" else None)
            files.append({"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
        plt.close(figure)
        manifest["figures"].append({
            "id": stem, "description": description,
            "data": {"path": f"../tables/{data_path.name}", "sha256": file_sha256(data_path),
                     "rows": len(table)},
            "files": files,
        })

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "MASHeart-v1"}):
        counts = dev_frame["label"].value_counts().reindex(LABELS, fill_value=0)
        label_table = counts.rename_axis("label").reset_index(name="examples")
        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        bars = axis.bar(counts.index, counts.values, color=[COLORS[label] for label in counts.index])
        axis.bar_label(bars, padding=3)
        axis.set(title="Official development label balance", xlabel="Stance label", ylabel="Examples")
        axis.set_ylim(0, max(counts) * 1.18)
        finish(figure, "01_development_label_balance", label_table,
               "Official development counts before creating the internal group split.")

        length_table = dev_frame[["id", "group_id", "label"]].copy()
        length_table["statement_words"] = dev_frame["text"].map(lambda value: len(value.split()))
        length_table["question_words"] = dev_frame["question"].map(lambda value: len(value.split()))
        figure, axis = plt.subplots(figsize=(6.2, 3.6))
        bins = np.histogram_bin_edges(length_table["statement_words"], bins=10)
        axis.hist([length_table.loc[length_table.label.eq(label), "statement_words"] for label in LABELS],
                  bins=bins, stacked=True, color=[COLORS[label] for label in LABELS], label=LABELS,
                  edgecolor="white", linewidth=0.5)
        axis.set(title="Development statement lengths", xlabel="Whitespace-delimited words", ylabel="Examples")
        axis.legend(title="Gold stance", frameon=False)
        finish(figure, "02_development_statement_lengths", length_table,
               "Source-preserving whitespace word counts; these are not LLM subword-token counts.")

        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
        for axis, field, title in zip(axes, ["n", "groups"], ["Examples", "Source groups"]):
            bars = axis.bar(split_table["split"], split_table[field], color=["#287da8", "#e4ac54", "#7c669d"])
            axis.bar_label(bars, padding=3)
            axis.set(title=title, ylabel=title)
            axis.set_ylim(0, split_table[field].max() * 1.20)
            axis.tick_params(axis="x", labelrotation=12)
        figure.suptitle("Educational split: train/validation inside development")
        finish(figure, "03_source_group_split", split_table,
               "Training and validation are development subsets; official test membership is preserved.")

        order = list(LABELS) + (["INVALID"] if predictions["tfidf_logreg"].eq("INVALID").any() else [])
        matrix = confusion_matrix(predictions["gold"], predictions["tfidf_logreg"], labels=order)
        matrix_table = pd.DataFrame([
            {"gold": gold, "prediction": predicted, "count": int(matrix[i, j])}
            for i, gold in enumerate(order) for j, predicted in enumerate(order)
        ])
        figure, axis = plt.subplots(figsize=(4.9, 4.0))
        ConfusionMatrixDisplay(matrix, display_labels=order).plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title("TF-IDF baseline: full official test")
        finish(figure, "04_test_confusion_matrix", matrix_table,
               "Rows are gold and columns are predictions; invalid outputs are retained when present.")

        figure, axis = plt.subplots(figsize=(7.4, 3.7))
        positions = np.arange(len(test_metrics))
        for offset, metric, color in [(-0.18, "accuracy", "#287da8"), (0.18, "macro_f1", "#c56c39")]:
            values = test_metrics[metric].to_numpy()
            bars = axis.bar(positions + offset, values, width=0.34, label=metric.replace("_", " "), color=color)
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        axis.set_xticks(positions, [DISPLAY_NAMES.get(name, name) for name in test_metrics["system"]])
        axis.set(title="CPU comparators on every test example", ylabel="Score", ylim=(0, 1.12))
        axis.legend(frameon=False, loc="upper left")
        finish(figure, "05_test_metric_comparison", test_metrics,
               "Observed accuracy and fixed-two-class macro-F1 on the complete official test split.")

        recall_table = pd.DataFrame([
            {"system": row["system"], "label": label, "recall": row[f"recall_{label}"],
             "support": row[f"support_{label}"]}
            for row in test_metrics.to_dict("records") for label in LABELS
        ])
        figure, axis = plt.subplots(figsize=(7.4, 3.7))
        for offset, label in [(-0.18, "NO"), (0.18, "YS")]:
            values = [float(recall_table.loc[(recall_table.system == name) &
                                             (recall_table.label == label), "recall"].iloc[0])
                      for name in test_metrics["system"]]
            bars = axis.bar(positions + offset, values, width=0.34, color=COLORS[label], label=label)
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        axis.set_xticks(positions, [DISPLAY_NAMES.get(name, name) for name in test_metrics["system"]])
        axis.set(title="Recall separates majority and minority behavior", ylabel="Recall", ylim=(0, 1.12))
        axis.legend(title="Gold class", frameon=False, loc="upper left")
        finish(figure, "06_test_per_class_recall", recall_table,
               "Per-class recall uses each class's full gold support, including failed predictions.")

        groups = sorted(group_scores["group_id"].unique())
        figure, axis = plt.subplots(figsize=(8.4, 4.3))
        group_positions = np.arange(len(groups))
        for offset, system in [(-0.18, "majority"), (0.18, "tfidf_logreg")]:
            subset = group_scores[group_scores.system.eq(system)].set_index("group_id").reindex(groups)
            axis.bar(group_positions + offset, subset["accuracy"], width=0.34,
                     color=COLORS[system], label=DISPLAY_NAMES[system])
        axis.set_xticks(group_positions, [group.removeprefix("qg-") for group in groups], rotation=40, ha="right")
        axis.set(title="Test accuracy by linked source group", xlabel="Group ID (qg- prefix omitted)",
                 ylabel="Accuracy", ylim=(0, 1.13))
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        finish(figure, "07_test_group_accuracy", group_scores,
               "Groups have different sample counts; the overall score is not the unweighted mean of these bars.")

        differences = np.asarray(bootstrap["differences"])
        bootstrap_table = pd.DataFrame({"draw": np.arange(1, len(differences) + 1),
                                        "macro_f1_difference": differences})
        figure, axis = plt.subplots(figsize=(7.1, 3.7))
        axis.hist(differences, bins=25, color="#287da8", edgecolor="white", linewidth=0.5)
        axis.axvline(0, color="#333333", linestyle="--", label="No difference")
        axis.axvline(bootstrap["ci_lower"], color="#c56c39", linestyle=":", label="95% percentile limits")
        axis.axvline(bootstrap["ci_upper"], color="#c56c39", linestyle=":")
        axis.set(title="Paired source-group bootstrap on full test",
                 xlabel="Macro-F1(TF-IDF) - Macro-F1(majority)", ylabel="Bootstrap draws")
        axis.legend(frameon=False, fontsize=8)
        finish(figure, "08_paired_cluster_bootstrap", bootstrap_table,
               "Whole groups are sampled with replacement; both systems receive the same sampled rows.")

    manifest_path = directory / "figure_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"figure_count": len(manifest["figures"]), "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path), "formats": ["png", "svg"]}


def regenerate_from_run(data_dir: str | Path, run_dir: str | Path) -> dict:
    """Verify a completed run and redraw its figures without fitting or selecting a model."""
    from .data import load_frame, verify_sources

    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    reports, tables = run_dir / "reports", run_dir / "tables"
    complete = json.loads((reports / "complete_manifest.json").read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("schema_version") != 1:
        raise ValueError("Figure regeneration requires an audited, completed run.")
    frozen_path = reports / "frozen_protocol.json"
    frozen_hash = file_sha256(frozen_path)
    if frozen_hash != complete["frozen_protocol_sha256"]:
        raise ValueError("The frozen protocol changed after the run completed.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    audit = verify_sources(data_dir)
    if audit["manifest_sha256"] != frozen["source_manifest_sha256"] or audit["artifacts"] != frozen["source_artifacts"]:
        raise ValueError("The supplied data differ from the frozen experiment's preserved sources.")
    for artifact in complete["artifacts"]:
        path = (run_dir / artifact["path"]).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError("Run artifact paths must remain inside the run directory.")
        if file_sha256(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"A completed run artifact changed: {artifact['path']}")
    splits = pd.read_csv(tables / "split_summary.csv")
    predictions = pd.read_csv(tables / "test_predictions.csv")
    metrics = pd.read_csv(tables / "test_metrics.csv")
    group_scores = pd.read_csv(tables / "test_group_scores.csv")
    bootstrap = json.loads((reports / "bootstrap_summary.json").read_text(encoding="utf-8"))
    bootstrap["differences"] = pd.read_csv(tables / "bootstrap_differences.csv")["macro_f1_difference"].to_numpy()
    test_n = int(splits.loc[splits["split"].eq("test"), "n"].iloc[0])
    if len(predictions) != test_n or not predictions["id"].is_unique or not metrics["n"].eq(test_n).all():
        raise ValueError("Stored figure inputs do not describe a complete test evaluation.")
    if len(bootstrap["differences"]) != bootstrap["rounds"]:
        raise ValueError("The stored bootstrap sample count is incomplete.")
    result = generate_research_figures(load_frame(data_dir / "raw" / "a_dev.jsonl", "dev"), splits,
                                       predictions, metrics, group_scores, bootstrap,
                                       run_dir / "figures", tables)
    return {**result, "verified_existing_run": True, "frozen_protocol_sha256": frozen_hash,
            "training_performed": False, "test_configuration_changed": False}
