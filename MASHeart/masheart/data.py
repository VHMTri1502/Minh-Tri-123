"""Load the preserved Task-A release and enforce source-disjoint feature boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LABELS = ("NO", "YS")
REQUIRED_COLUMNS = {
    "id", "question", "text", "label", "group_id", "pmid", "split",
    "review_id", "review_no", "review_title", "claim_type", "source", "source_url",
}


def file_sha256(path: str | Path) -> str:
    """Hash a file incrementally without interpreting its labels or other contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_audit(text: str) -> str:
    """Normalize case and whitespace for duplicate audits; never rewrite source quotes."""
    return " ".join(text.casefold().split())


def verify_sources(data_dir: str | Path) -> dict:
    """Verify manifest byte counts and SHA-256 values without parsing any JSONL labels."""
    directory = Path(data_dir).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("artifacts"), list):
        raise ValueError("Unsupported data manifest schema.")
    checked = []
    seen = set()
    for artifact in manifest["artifacts"]:
        relative = artifact["path"]
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or relative in seen:
            raise ValueError("Manifest paths must be unique and remain inside the data directory.")
        seen.add(relative)
        actual_hash = file_sha256(path)
        actual_bytes = path.stat().st_size
        if actual_hash != artifact["sha256"] or actual_bytes != artifact["bytes"]:
            raise ValueError(f"Source verification failed: {relative}")
        checked.append({"path": relative, "sha256": actual_hash, "bytes": actual_bytes})
    if not {"raw/a_dev.jsonl", "raw/a_test.jsonl"} <= seen:
        raise ValueError("The manifest must include both preserved Task-A source files.")
    return {
        "passed": True,
        "manifest_sha256": file_sha256(manifest_path),
        "files_verified": len(checked),
        "artifacts": checked,
        "method": "byte-count and SHA-256 verification; JSONL labels were not parsed",
    }


def load_frame(path: str | Path, expected_split: str) -> pd.DataFrame:
    """Read UTF-8 JSONL, validate its schema and return a new, source-preserving table."""
    if expected_split not in {"dev", "test"}:
        raise ValueError("expected_split must be 'dev' or 'test'.")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty or not REQUIRED_COLUMNS <= set(frame.columns):
        raise ValueError("The source must be a nonempty Task-A table with the full release schema.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any() or not frame["id"].is_unique:
        raise ValueError("Null values or repeated example IDs were found.")
    for column in REQUIRED_COLUMNS - {"review_no"}:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Column {column} must contain nonempty strings.")
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Gold labels must be NO or YS.")
    if not frame["split"].eq(expected_split).all():
        raise ValueError("The source contains a different official split.")
    return frame.reset_index(drop=True).copy()


def assert_disjoint(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Reject overlapping IDs, source groups, PMIDs or normalized statement texts."""
    overlaps = {}
    for field in ("id", "group_id", "pmid"):
        overlap = set(left[field]) & set(right[field])
        overlaps[field] = sorted(overlap)
    overlaps["normalized_text"] = sorted(
        set(left["text"].map(normalize_for_audit))
        & set(right["text"].map(normalize_for_audit))
    )
    if any(overlaps.values()):
        counts = {key: len(values) for key, values in overlaps.items() if values}
        raise ValueError(f"Source leakage detected: {counts}")
    return {"passed": True, "overlap_counts": {key: 0 for key in overlaps}}


def split_development(dev: pd.DataFrame, seed: int = 17,
                      validation_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete source groups inside development; official test is never an input."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if dev.empty or not dev["split"].eq("dev").all() or dev["group_id"].nunique() < 2:
        raise ValueError("A nonempty development table with at least two groups is required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, validation_indices = next(splitter.split(dev, groups=dev["group_id"]))
    train = dev.iloc[train_indices].reset_index(drop=True).copy()
    validation = dev.iloc[validation_indices].reset_index(drop=True).copy()
    assert_disjoint(train, validation)
    if set(train["label"]) != set(LABELS) or set(validation["label"]) != set(LABELS):
        raise ValueError("Both training and validation must contain both classes; revise the split before test.")
    return train, validation


def public_payload(row: Mapping) -> dict[str, str]:
    """Expose question and statement only; gold labels and source metadata cannot enter inference."""
    result = {field: row[field] for field in ("question", "text")}
    if not all(isinstance(value, str) and value.strip() for value in result.values()):
        raise ValueError("question and text must be nonempty strings.")
    return result


def compose_text(row: Mapping) -> str:
    """Combine the two allowed fields for a text baseline without IDs, groups or targets."""
    payload = public_payload(row)
    return f"Question: {payload['question']}\nStatement: {payload['text']}"
