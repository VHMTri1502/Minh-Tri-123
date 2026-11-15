"""Verify shipped research artifacts; optionally execute both notebooks in clean CPU kernels."""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_project(execute_notebooks: bool = False) -> dict:
    import nbformat
    from masheart.data import verify_sources

    started = time.perf_counter()
    checks = {}
    python_files = sorted(p for directory in ["masheart", "scripts", "tests"]
                          for p in (ROOT / directory).glob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    checks["python_sources_parse"] = len(python_files)
    checks["data_verification"] = verify_sources(ROOT / "data")
    unit_log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    unit = unittest.TextTestRunner(stream=unit_log, verbosity=2).run(suite)
    checks["tests"] = {"run": unit.testsRun, "failures": len(unit.failures), "errors": len(unit.errors),
                       "skipped": len(unit.skipped), "successful": unit.wasSuccessful(),
                       "log": unit_log.getvalue()}
    if not unit.wasSuccessful():
        raise AssertionError(unit_log.getvalue())

    reference = ROOT / "results" / "cpu_reference"
    complete = json.loads((reference / "reports" / "complete_manifest.json").read_text(encoding="utf-8"))
    for artifact in complete["artifacts"]:
        path = reference / artifact["path"]
        if not path.resolve().is_relative_to(reference.resolve()):
            raise ValueError("An artifact path escapes its run directory.")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], artifact["path"]
    frozen = json.loads((reference / "reports" / "frozen_protocol.json").read_text(encoding="utf-8"))
    for name, digest in frozen["module_sha256"].items():
        assert hashlib.sha256((ROOT / "masheart" / name).read_bytes()).hexdigest() == digest, name
    checks["reference_complete_manifest_artifacts"] = len(complete["artifacts"])
    checks["frozen_reference_source_code_matches"] = True
    checks["gpu_status"] = json.loads((ROOT / "results" / "gpu" / "STATUS.json").read_text())
    assert checks["gpu_status"]["execution_status"] == "not_run"

    invalid_links = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "data" / "DATA_CARD.md"]:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text, path
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target and not target.startswith("#"):
                target = target.split("#")[0].strip("<>")
                if not (path.parent / target).exists():
                    invalid_links.append({"document": str(path.relative_to(ROOT)), "target": target})
    assert not invalid_links, invalid_links
    checks["portable_document_links_valid"] = True

    notebook_results = []
    if execute_notebooks:
        from nbclient import NotebookClient
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        os.environ["MASHEART_CPU_ONLY"] = "1"
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["IPYTHONDIR"] = str(ROOT / ".verification" / "ipython")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".verification" / "jupyter")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        result = {"path": path.relative_to(ROOT).as_posix(), "cells": len(notebook.cells),
                  "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                  "schema_and_syntax": "passed", "execution": "not_requested"}
        if execute_notebooks:
            runtime = ROOT / ".verification" / (path.stem + "_runtime")
            runtime.mkdir(parents=True, exist_ok=True)
            client = NotebookClient(notebook, timeout=300, kernel_name="python3",
                                    resources={"metadata": {"path": str(runtime)}})
            executed = client.execute()
            errors = [output for cell in executed.cells if cell.cell_type == "code"
                      for output in cell.get("outputs", []) if output.get("output_type") == "error"]
            assert not errors, errors
            assert all(cell.get("execution_count") is not None for cell in executed.cells if cell.cell_type == "code")
            nbformat.write(executed, path)
            result.update(execution="passed_in_clean_cpu_kernel", cell_errors=0,
                          bootstrap="embedded assets; no parent repository import",
                          note="Optional GPU branches skipped; GPU metrics were not generated.")
        notebook_results.append(result)
    assert len(notebook_results) >= 2
    checks["notebooks"] = notebook_results
    checks["elapsed_seconds"] = time.perf_counter() - started
    return {"project": "MASHeart", "status": "passed", "validation_host_scope": "CPU",
            "gpu_execution": "not_run", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebooks", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_project(args.execute_notebooks)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
