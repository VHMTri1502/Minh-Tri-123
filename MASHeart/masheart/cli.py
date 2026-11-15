"""Small, explicit entry points for every stage of the research pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def project_root() -> Path:
    """Resolve resources relative to the package, never to the parent HeartMAS repo."""
    return Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A configuration must be a JSON object.")
    return value


def setup_data(data_dir: Path, output_dir: Path, config: dict) -> dict:
    """Verify the supplied sources and prepare group-disjoint train/validation files."""
    from .data import assert_disjoint, load_frame, split_development, verify_sources

    audit = verify_sources(data_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Data setup output is not empty; choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = load_frame(data_dir / "raw" / "a_dev.jsonl", "dev")
    train, validation = split_development(
        dev, seed=config.get("seed", 17), validation_fraction=config.get("validation_fraction", .25))
    assert_disjoint(train, validation)
    for name, frame in [("train", train), ("validation", validation)]:
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in frame.to_dict("records"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"protocol": "educational-dev-fit-group-split", "seed": config.get("seed", 17),
                "train_ids": train.id.tolist(), "validation_ids": validation.id.tolist(),
                "train_groups": sorted(train.group_id.unique().tolist()),
                "validation_groups": sorted(validation.group_id.unique().tolist()),
                "train_rows": len(train), "validation_rows": len(validation),
                "test_policy": "Test labels are not loaded by setup-data; preserve the official test file.",
                "data_audit": audit}
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(prog="masheart", description="MASHeart step-by-step research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Verify source checksums and provenance without parsing test labels")
    audit_parser.add_argument("--data", type=Path, default=root / "data")
    setup_parser = sub.add_parser("setup-data", help="Prepare train/validation using development groups only")
    setup_parser.add_argument("--data", type=Path, default=root / "data")
    setup_parser.add_argument("--config", type=Path, default=root / "configs" / "cpu.json")
    setup_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run", help="Execute training, frozen evaluation, figures, and report generation")
    run_parser.add_argument("--mode", choices=["cpu", "t4"], default="cpu")
    run_parser.add_argument("--data", type=Path, default=root / "data")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--output", type=Path)
    plot_parser = sub.add_parser("visualize", help="Regenerate scientific figures from a completed CPU run")
    plot_parser.add_argument("--data", type=Path, default=root / "data")
    plot_parser.add_argument("--run-dir", type=Path, required=True)
    prediction_parser = sub.add_parser("predict", help="Apply a saved CPU model to a new question and statement")
    prediction_parser.add_argument("--model", type=Path, required=True)
    prediction_parser.add_argument("--question", required=True)
    prediction_parser.add_argument("--statement", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            from .data import verify_sources
            answer = verify_sources(args.data.resolve())
        elif args.command == "setup-data":
            answer = setup_data(args.data.resolve(), args.output.resolve(), read_config(args.config))
        elif args.command == "run":
            config_path = args.config or root / "configs" / ("t4.json" if args.mode == "t4" else "cpu.json")
            config = read_config(config_path)
            output = args.output or root / "results" / f"{args.mode}_run_{time.time_ns()}"
            if args.mode == "cpu":
                from .experiment import run_cpu_experiment
                answer = run_cpu_experiment(args.data.resolve(), output.resolve(), config)
            else:
                from .gpu import run_gpu_experiment
                answer = run_gpu_experiment(args.data.resolve(), output.resolve(), config)
        elif args.command == "visualize":
            from .visualization import regenerate_from_run
            answer = regenerate_from_run(args.data.resolve(), args.run_dir.resolve())
        else:
            import joblib
            from .data import compose_text, public_payload
            source = public_payload({"question": args.question, "text": args.statement})
            model = joblib.load(args.model)
            features = [compose_text(source)]
            answer = {"question": source["question"], "statement": source["text"],
                      "label": str(model.predict(features)[0]),
                      "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      "task": "Stance toward the supplied question; not clinical truth."}
            if hasattr(model, "predict_proba"):
                answer["uncalibrated_model_scores"] = dict(zip(
                    model.classes_.tolist(), model.predict_proba(features)[0].tolist()))
        print(json.dumps(answer, ensure_ascii=False, indent=2, default=str))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError, AssertionError) as error:
        print(f"MASHeart: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
