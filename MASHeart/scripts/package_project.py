"""Create one portable MASHeart ZIP with a complete SHA-256 inventory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", ".verification", ".pytest_cache", "__pycache__",
                  ".ipynb_checkpoints", ".cache", "runs", "trainer_checkpoints", "qlora_adapter"}


def payload_files() -> list[Path]:
    selected = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS & set(relative.parts) or path.suffix in {".zip", ".pyc", ".pyo"}:
            continue
        if path.name == "MANIFEST.sha256" or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[0] == "results" and relative.parts[1] not in {"cpu_reference", "gpu"}:
            continue
        selected.append(path)
    return selected


def package(output: Path) -> dict:
    paths = payload_files()
    required = {"README.md", "masheart/gpu.py", "notebooks/MASHeart_Step_by_Step.ipynb",
                "notebooks/01_Data_Exploration.ipynb", "results/cpu_reference/tables/test_metrics.csv",
                "reports/verification.json"}
    assert required <= {p.relative_to(ROOT).as_posix() for p in paths}
    manifest = ROOT / "MANIFEST.sha256"
    manifest.write_text("\n".join(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(ROOT).as_posix()
        for path in paths) + "\n", encoding="utf-8")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths + [manifest]:
            archive.write(path, arcname="MASHeart/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None, "ZIP CRC verification failed."
        assert len(archive.namelist()) == len(paths) + 1
        assert all(name.startswith("MASHeart/") and ".." not in Path(name).parts for name in archive.namelist())
    archive_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(archive_hash + "  " + output.name + "\n", encoding="utf-8")
    return {"archive": str(output), "bytes": output.stat().st_size, "files": len(paths) + 1,
            "sha256": archive_hash, "single_root": "MASHeart", "crc_validated": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "MASHeart.zip")
    args = parser.parse_args()
    import json
    print(json.dumps(package(args.output), indent=2))
