"""Extract the ZIP safely, verify its inventory and run the actual pipeline in isolation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run_check(archive_path: Path) -> dict:
    import pandas as pd

    scratch = ROOT / ".verification"
    scratch.mkdir(exist_ok=True)
    extraction = Path(tempfile.mkdtemp(prefix="portable_zip_", dir=scratch)).resolve()
    assert extraction.is_relative_to(scratch.resolve())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for member in archive.infolist():
            target = (extraction / member.filename).resolve()
            if not target.is_relative_to(extraction) or not member.filename.startswith("MASHeart/"):
                raise ValueError("Unsafe or unexpected ZIP member.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    extracted_project = extraction / "MASHeart"
    inventory = (extracted_project / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    for line in inventory:
        digest, relative = line.split("  ", 1)
        path = (extracted_project / relative).resolve()
        assert path.is_relative_to(extracted_project.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative

    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", MASHEART_CPU_ONLY="1")
    # Prevent a parent checkout injected through PYTHONPATH from satisfying imports.
    env.pop("PYTHONPATH", None)
    commands = [
        [sys.executable, "-m", "masheart", "audit"],
        [sys.executable, "-m", "masheart", "setup-data", "--output", ".verification/data_setup"],
        [sys.executable, "-m", "masheart", "run", "--mode", "cpu", "--output", ".verification/cpu_repeat"],
        [sys.executable, "-m", "masheart", "visualize", "--run-dir", ".verification/cpu_repeat"],
        [sys.executable, "-m", "masheart", "predict", "--model", "results/cpu_reference/models/tfidf_logreg.joblib",
         "--question", "Does treatment X reduce outcome Y?", "--statement", "The synthetic trial found no reduction."],
        [sys.executable, "scripts/verify_project.py"],
    ]
    records = []
    for command in commands:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=extracted_project, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
        records.append({"command": command[1:], "exit_code": result.returncode,
                        "elapsed_seconds": time.perf_counter() - started})
        if result.returncode:
            raise RuntimeError("Portable command failed: " + " ".join(command[1:]) + "\n" + result.stderr + result.stdout)

    repeated = pd.read_csv(extracted_project / ".verification/cpu_repeat/tables/test_predictions.csv")
    reference = pd.read_csv(extracted_project / "results/cpu_reference/tables/test_predictions.csv")
    assert repeated[["id", "gold", "majority", "tfidf_logreg"]].equals(reference[["id", "gold", "majority", "tfidf_logreg"]])
    completed = json.loads((extracted_project / ".verification/cpu_repeat/reports/complete_manifest.json").read_text())
    for artifact in completed["artifacts"]:
        path = extracted_project / ".verification/cpu_repeat" / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    source_hashes = {}
    for path in sorted((extracted_project / "masheart").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"))
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"project": "MASHeart", "status": "passed", "archive_crc_passed": True,
            "inventory_files_verified": len(inventory), "isolated_extraction": True,
            "parent_pythonpath_removed": True, "commands": records,
            "cpu_repeat_rows": len(repeated), "predictions_match_reference": True,
            "visualization_preserves_completed_artifact_hashes": True,
            "code_sha256": source_hashes,
            "scope": "New extraction path, existing validated Python environment, CPU workflow and CLI only.",
            "gpu_execution": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "MASHeart.zip")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "portability.json")
    args = parser.parse_args()
    report = run_check(args.archive.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
