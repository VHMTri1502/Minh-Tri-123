"""QLoRA training and a source-checked three-stage MAS for one T4 GPU.

Importing this module never imports torch, transformers, PEFT, or bitsandbytes.
The parser, revision gate, token masking, and checkpoint code run on a CPU.
"""
from __future__ import annotations

import contextlib
import copy
import csv
from dataclasses import asdict, dataclass, fields
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


LABELS = ("NO", "YS")
INVALID = "INVALID"
POLICY_VERSION = "masheart-t4-stance-v1"
PACKAGE_PINS = {
    "transformers": "4.57.1", "peft": "0.17.1",
    "accelerate": "1.10.1", "bitsandbytes": "0.48.1",
}
ANALYST_SYSTEM = (
    "Read only the supplied research QUESTION and STATEMENT. Predict stance toward "
    "the question, not clinical truth. YS means the statement affirms the proposition "
    "asked, including a tentative finding. NO means the statement negates it OR does "
    "not affirm it. Check the asked outcome, population, intervention, and scope. "
    "Negation about a different outcome is not enough for NO. "
    'Return exactly one JSON object: {"label":"YS"} or {"label":"NO"}.'
)
REVIEWER_SYSTEM = (
    "You review stance toward a supplied research QUESTION, not clinical truth. "
    "YS means the STATEMENT affirms the asked proposition, even tentatively. NO means "
    "it negates OR does not affirm the proposition. Read the original statement and "
    "the analyst draft. Do not disagree merely to disagree. Check outcome, negation, "
    "population, intervention, and scope. Return one JSON object with label (YS/NO), "
    "quote (a nonempty exact passage occurring once in the statement, preserving "
    "relevant negation), and rationale (one short sentence). No markdown or medical advice."
)
VERIFIER_SYSTEM = (
    "You verify stance toward a supplied research QUESTION, not clinical truth. "
    "Check the original STATEMENT against the analyst and reviewer. YS means affirmation "
    "of the asked proposition, even tentatively; NO means negation or lack of affirmation. "
    "Do not automatically follow the reviewer. Return one JSON object with label (YS/NO), "
    "quote (a nonempty exact passage occurring once in the statement, preserving "
    "negation), and rationale (one short sentence). No markdown or medical advice."
)


class GPUUnavailableError(RuntimeError):
    """The requested GPU experiment cannot run in the current environment."""


@dataclass(frozen=True)
class GPUConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    seed: int = 17
    validation_fraction: float = 0.25
    max_train_tokens: int = 512
    max_input_tokens: int = 1536
    max_new_tokens: int = 192
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    minimum_vram_gib: float = 12.0

    @classmethod
    def from_dict(cls, value: dict | None = None) -> "GPUConfig":
        options = dict(value or {})
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown T4 configuration fields: " + ", ".join(sorted(unknown)))
        if "target_modules" in options:
            if not isinstance(options["target_modules"], (list, tuple)):
                raise ValueError("target_modules must be a list of module names.")
            options["target_modules"] = tuple(options["target_modules"])
        config = cls(**options)
        integer_fields = (config.seed, config.max_train_tokens, config.max_input_tokens,
                          config.max_new_tokens, config.lora_r, config.lora_alpha,
                          config.epochs, config.batch_size, config.gradient_accumulation_steps)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
            raise ValueError("Seeds, token limits, ranks, epochs, and batch sizes must be integers.")
        if config.seed < 0:
            raise ValueError("seed must be nonnegative.")
        numeric_fields = (config.learning_rate, config.lora_dropout,
                          config.validation_fraction, config.minimum_vram_gib)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in numeric_fields):
            raise ValueError("Learning rate, dropout, validation fraction, and VRAM limit must be finite numbers.")
        if not config.target_modules or any(not isinstance(name, str) or not name for name in config.target_modules):
            raise ValueError("At least one nonempty LoRA target module name is required.")
        if not isinstance(config.model_id, str) or not config.model_id.strip():
            raise ValueError("model_id must be a nonempty model identifier.")
        if config.batch_size != 1:
            raise ValueError("The supplied T4 configuration supports batch_size=1 only.")
        positive = (config.max_train_tokens, config.max_input_tokens, config.max_new_tokens,
                    config.lora_r, config.lora_alpha, config.epochs,
                    config.gradient_accumulation_steps, config.learning_rate, config.minimum_vram_gib)
        if any(value <= 0 for value in positive):
            raise ValueError("Token limits and training parameters must be positive.")
        if config.max_train_tokens % 8:
            raise ValueError("max_train_tokens must be a multiple of eight for padded batches.")
        if not 0 < config.validation_fraction < 1 or not 0 <= config.lora_dropout < 1:
            raise ValueError("Invalid validation fraction or LoRA dropout.")
        if not isinstance(config.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", config.revision):
            raise ValueError("revision must be an immutable 40-character model commit SHA.")
        return config


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_payload(row: Any) -> dict[str, str]:
    """A feature boundary: gold labels, identifiers, and groups never enter prompts."""
    payload = {key: row[key] for key in ("question", "text")}
    if any(not isinstance(value, str) or not value.strip() for value in payload.values()):
        raise ValueError("question and text must be nonempty strings.")
    return payload


def analyst_messages(payload: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user", "content": _canonical_json(source_payload(payload))}]


def encode_supervised(row: dict, tokenizer: Any, max_tokens: int = 512) -> dict[str, list[int]]:
    """Mask prompt loss, keep every answer token and EOS, and reject truncation."""
    if row.get("label") not in LABELS:
        raise ValueError("A supervised row must have a NO or YS label.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    prompt = list(tokenizer.apply_chat_template(
        analyst_messages(source_payload(row)), tokenize=True, add_generation_prompt=True,
    ))
    answer = tokenizer(json.dumps({"label": row["label"]}, separators=(",", ":")),
                       add_special_tokens=False)["input_ids"]
    target = list(answer) + [int(tokenizer.eos_token_id)]
    input_ids = prompt + target
    if len(input_ids) > max_tokens:
        raise ValueError(f"Row {row.get('id', '<unknown>')} has {len(input_ids)} tokens; "
                         f"limit is {max_tokens}. The source was not truncated.")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt) + target}


def collate_examples(examples: list[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad source tokens and attention; pad loss labels with -100."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    longest = max(len(item["input_ids"]) for item in examples)
    padded_length = int(math.ceil(longest / 8) * 8)
    output: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in examples:
        if len(item["input_ids"]) != len(item["labels"]) or len(item["labels"]) != len(item["attention_mask"]):
            raise ValueError("Token, label, and attention arrays must have equal lengths.")
        padding = padded_length - len(item["input_ids"])
        output["input_ids"].append(list(item["input_ids"]) + [pad_token_id] * padding)
        output["attention_mask"].append(list(item["attention_mask"]) + [0] * padding)
        output["labels"].append(list(item["labels"]) + [-100] * padding)
    return output


def parse_packet(generation: dict, payload: dict, require_quote: bool = False) -> dict:
    """Parse strict JSON and exact unique source quotations without consulting gold."""
    parsed = {"label": INVALID, "quote": "", "rationale": "",
              "schema_valid": False, "quote_valid": False}
    if generation.get("status") != "ok":
        return parsed
    try:
        value = json.loads(generation.get("raw", ""))
    except (json.JSONDecodeError, TypeError):
        return parsed
    if not isinstance(value, dict) or value.get("label") not in LABELS:
        return parsed
    parsed.update(label=value["label"], schema_valid=True)
    quote = value.get("quote", "")
    rationale = value.get("rationale", "")
    parsed["quote"] = quote if isinstance(quote, str) else ""
    parsed["rationale"] = rationale[:500] if isinstance(rationale, str) else ""
    source = source_payload(payload)["text"]
    parsed["quote_valid"] = (bool(parsed["quote"].strip())
                             and source.find(parsed["quote"]) >= 0
                             and source.find(parsed["quote"]) == source.rfind(parsed["quote"]))
    if require_quote and (not isinstance(quote, str) or not quote.strip()
                          or not isinstance(rationale, str) or not rationale.strip()):
        parsed["schema_valid"] = False
    return parsed


def gate_packets(initial: dict, reviewer: dict, final: dict) -> dict:
    """A structural revision gate; a matching quotation is not a semantic proof."""
    initial_ok = initial.get("schema_valid", False) and initial.get("label") in LABELS
    reviewer_ok = (reviewer.get("schema_valid", False) and reviewer.get("quote_valid", False)
                   and reviewer.get("label") in LABELS)
    final_ok = (final.get("schema_valid", False) and final.get("quote_valid", False)
                and final.get("label") in LABELS)
    if final_ok and not initial_ok:
        return {"label": final["label"], "review": True, "gate_status": "repaired_invalid_initial"}
    if initial_ok and final_ok and final["label"] == initial["label"]:
        return {"label": initial["label"],
                "review": not reviewer_ok or reviewer["label"] != final["label"],
                "gate_status": "kept_initial"}
    if initial_ok and final_ok and reviewer_ok and reviewer["label"] == final["label"]:
        return {"label": final["label"], "review": False, "gate_status": "accepted_sourced_revision"}
    if initial_ok:
        return {"label": initial["label"], "review": True,
                "gate_status": "blocked_revision_or_invalid_final"}
    return {"label": INVALID, "review": True, "gate_status": "no_valid_answer"}


class TraceCache:
    """Append-only, source- and protocol-bound checkpoints, including failed outputs."""

    def __init__(self, directory: Path, split_name: str, protocol_sha256: str,
                 sources: list[dict]):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", split_name):
            raise ValueError("Invalid split name.")
        ids = [row["id"] for row in sources]
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("Checkpoint sources need unique, nonempty string IDs.")
        public_sources = [{"id": row["id"], "payload": source_payload(row.get("payload", row))}
                          for row in sources]
        self.protocol_sha256 = protocol_sha256
        self.expected = {row["id"]: fingerprint(row["payload"]) for row in public_sources}
        self.ids = ids
        input_sha256 = fingerprint(public_sources)
        manifest = {"protocol_sha256": protocol_sha256, "input_sha256": input_sha256,
                    "ids": ids, "split": split_name}
        key = fingerprint(manifest)[:20]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{split_name}_{key}.jsonl"
        self.manifest_path = self.path.with_suffix(".manifest.json")
        if self.manifest_path.exists():
            if json.loads(self.manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Checkpoint manifest differs from the current protocol or input.")
        else:
            if self.path.exists():
                raise ValueError("A checkpoint without its manifest was retained for inspection.")
            _write_json(self.manifest_path, manifest)
        self.completed: dict[str, dict] = {}
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Checkpoint line {number} is incomplete or invalid; "
                                     "the file was retained, with no hidden retry.") from error
                if not isinstance(record, dict):
                    raise ValueError("Checkpoint lines must be JSON objects; the file was retained.")
                identifier = record.get("id")
                if identifier in self.completed or identifier not in self.expected:
                    raise ValueError("Checkpoint contains a duplicate or unexpected ID.")
                if record.get("protocol_sha256") != protocol_sha256 or record.get("payload_sha256") != self.expected[identifier]:
                    raise ValueError("Checkpoint result does not match its source and protocol.")
                if not isinstance(record.get("result"), dict):
                    raise ValueError("Checkpoint result must be a JSON object.")
                if fingerprint(record["result"]) != record.get("result_sha256"):
                    raise ValueError("Checkpoint result content failed its checksum; the file was retained.")
                self.completed[identifier] = record["result"]

    def append(self, identifier: str, result: dict) -> None:
        if identifier not in self.expected or identifier in self.completed:
            raise ValueError("Cannot append an unknown or already completed ID.")
        record = {"id": identifier, "payload_sha256": self.expected[identifier],
                  "protocol_sha256": self.protocol_sha256,
                  "result_sha256": fingerprint(result), "result": result}
        encoded = _canonical_json(record)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed[identifier] = result


def _require_gpu_stack(config: GPUConfig) -> dict:
    """Fail before model downloads if CUDA or the pinned optional stack is absent."""
    try:
        import torch
    except ImportError as error:
        raise GPUUnavailableError("T4 mode requires PyTorch with CUDA and requirements-t4.txt.") from error
    if not torch.cuda.is_available():
        raise GPUUnavailableError("No CUDA GPU is available. In Colab select Runtime > "
                                  "Change runtime type > T4 GPU, then rerun. No model was downloaded.")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < config.minimum_vram_gib * 2**30:
        raise GPUUnavailableError(f"The configured run needs at least {config.minimum_vram_gib:g} GiB VRAM.")
    for name, expected in PACKAGE_PINS.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise GPUUnavailableError(f"Install requirements-t4.txt before T4 mode; {name} is missing.") from error
        loaded = getattr(sys.modules[name], "__version__", installed) if name in sys.modules else installed
        if installed != expected or loaded != expected:
            raise GPUUnavailableError(f"{name} must be {expected}; installed={installed}, loaded={loaded}. "
                                      "Install requirements-t4.txt, restart the kernel, and rerun.")
    try:
        import transformers
        import peft
    except ImportError as error:
        raise GPUUnavailableError("The optional T4 stack could not be imported; inspect dependency errors.") from error
    versions = {name: importlib.metadata.version(name) for name in ["torch", *PACKAGE_PINS]}
    return {"torch": torch, "transformers": transformers, "peft": peft,
            "versions": versions, "gpu": {"name": props.name, "vram_gib": props.total_memory / 2**30}}


class T4Runtime:
    """Train once, select on validation, freeze, and run fixed-call source-only MAS."""

    def __init__(self, output_dir: Path, config: GPUConfig | dict | None = None,
                 stack: dict | None = None):
        self.config = config if isinstance(config, GPUConfig) else GPUConfig.from_dict(config)
        self.output_dir = Path(output_dir)
        self.adapter_dir = self.output_dir / "models" / "qlora_adapter"
        self.checkpoint_dir = self.output_dir / "models" / "trainer_checkpoints"
        self.stack = stack
        self.model = None
        self.tokenizer = None
        self.analyst_use_adapter = False
        self.protocol: dict | None = None
        self.protocol_sha256: str | None = None
        self.train_ids: list[str] = []
        self.validation_ids: list[str] = []
        self.training_seconds = 0.0
        self.training_signature: dict | None = None
        self.trained_adapter_sha256: str | None = None

    def load_model(self) -> None:
        if self.protocol is not None:
            raise RuntimeError("The protocol is frozen. Create a new runtime to load or train another model.")
        self.stack = self.stack or _require_gpu_stack(self.config)
        torch = self.stack["torch"]
        transformers = self.stack["transformers"]
        peft = self.stack["peft"]
        transformers.set_seed(self.config.seed)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_id, revision=self.config.revision,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        base = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_id, revision=self.config.revision,
            quantization_config=quantization, torch_dtype=torch.float16,
            device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        base = peft.prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.model = peft.get_peft_model(base, peft.LoraConfig(
            task_type="CAUSAL_LM", r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.target_modules), bias="none",
        ))
        self.model.print_trainable_parameters()

    def train(self, train_frame, validation_frame) -> None:
        if self.protocol is not None:
            raise RuntimeError("The protocol is frozen. Create a new run before retraining.")
        self.stack = self.stack or _require_gpu_stack(self.config)
        self.train_ids = train_frame["id"].tolist()
        self.validation_ids = validation_frame["id"].tolist()
        self.training_signature = {
            "policy": POLICY_VERSION, "code_sha256": file_sha256(Path(__file__)),
            "config": asdict(self.config), "analyst_prompt": ANALYST_SYSTEM,
            "train_sha256": fingerprint(train_frame.to_dict("records")),
            "validation_sha256": fingerprint(validation_frame.to_dict("records")),
            "versions": self.stack["versions"],
        }
        signature_path = self.output_dir / "logs" / "training_manifest.json"
        if signature_path.exists():
            previous = json.loads(signature_path.read_text(encoding="utf-8"))
            if fingerprint(previous) != fingerprint(self.training_signature):
                raise ValueError("This output directory contains a different training protocol. Use a new output directory.")
        else:
            _write_json(signature_path, self.training_signature)
        # Reject incompatible resumed training before downloading any model weights.
        if self.model is None:
            self.load_model()
        adapter_path = self.adapter_dir / "adapter_model.safetensors"
        completion_path = self.output_dir / "logs" / "training_completed.json"
        if completion_path.exists():
            completed = json.loads(completion_path.read_text(encoding="utf-8"))
            if not adapter_path.exists() or file_sha256(adapter_path) != completed.get("adapter_sha256"):
                raise ValueError("Saved adapter is missing or differs from the completed-training manifest.")
            self.model.load_adapter(str(self.adapter_dir), adapter_name="restored", is_trainable=False)
            self.model.set_adapter("restored")
            self.model.eval()
            self.model.config.use_cache = True
            self.training_seconds = completed["training_seconds"]
            self.trained_adapter_sha256 = completed["adapter_sha256"]
            print("Restored the completed adapter; training was not repeated.", flush=True)
            return
        torch = self.stack["torch"]
        transformers = self.stack["transformers"]
        train_items = [encode_supervised(row, self.tokenizer, self.config.max_train_tokens)
                       for row in train_frame.to_dict("records")]
        validation_items = [encode_supervised(row, self.tokenizer, self.config.max_train_tokens)
                            for row in validation_frame.to_dict("records")]

        class StanceDataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                return self.examples[index]

        def tensor_collator(examples):
            batch = collate_examples(examples, self.tokenizer.pad_token_id)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

        args = transformers.TrainingArguments(
            output_dir=str(self.checkpoint_dir), num_train_epochs=self.config.epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate, weight_decay=0.01,
            warmup_ratio=0.05, lr_scheduler_type="linear", optim="paged_adamw_8bit",
            max_grad_norm=0.3, fp16=True, bf16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
            prediction_loss_only=True, save_safetensors=True, logging_steps=1, report_to=[],
            seed=self.config.seed, data_seed=self.config.seed,
            remove_unused_columns=False, label_names=["labels"],
        )
        trainer = transformers.Trainer(
            model=self.model, args=args, train_dataset=StanceDataset(train_items),
            eval_dataset=StanceDataset(validation_items), data_collator=tensor_collator,
            processing_class=self.tokenizer,
        )
        resume_checkpoint = None
        if self.checkpoint_dir.exists():
            from transformers.trainer_utils import get_last_checkpoint
            resume_checkpoint = get_last_checkpoint(str(self.checkpoint_dir))
        if resume_checkpoint:
            print("Resuming the matching training checkpoint:", resume_checkpoint, flush=True)
        started = time.perf_counter()
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        self.training_seconds = time.perf_counter() - started
        history = list(trainer.state.log_history)
        self.model.save_pretrained(self.adapter_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(self.adapter_dir)
        _write_json(self.output_dir / "logs" / "training_history.json", history)
        self.trained_adapter_sha256 = file_sha256(adapter_path)
        _write_json(completion_path, {"training_seconds": self.training_seconds,
                                     "adapter_sha256": self.trained_adapter_sha256})
        self.model.zero_grad(set_to_none=True)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        self.model.config.use_cache = True
        self.model.eval()

    def generate(self, messages: list[dict], use_adapter: bool = False) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load the T4 model before generating.")
        if self.protocol is not None:
            self.assert_protocol()
        torch = self.stack["torch"]
        started = time.perf_counter()
        packet = {"raw": "", "status": "not_started", "input_tokens": 0,
                  "output_tokens": 0, "model_calls": 0, "elapsed_seconds": 0.0}
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                           add_generation_prompt=True)
            encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            packet["input_tokens"] = int(encoded["input_ids"].shape[1])
            if packet["input_tokens"] > self.config.max_input_tokens:
                packet["status"] = "input_too_long"
                return packet
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            mode = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
            self.model.eval()
            torch.cuda.synchronize()
            packet["model_calls"] = 1
            with mode, torch.inference_mode():
                generated = self.model.generate(
                    **encoded, generation_config=self.stack["transformers"].GenerationConfig(
                        max_new_tokens=self.config.max_new_tokens, do_sample=False, use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                    ),
                )
            suffix = generated[0, packet["input_tokens"]:]
            packet["output_tokens"] = int(len(suffix))
            packet["raw"] = self.tokenizer.decode(suffix, skip_special_tokens=True)
            packet["status"] = (
                "truncated" if len(suffix) >= self.config.max_new_tokens
                and int(suffix[-1]) != self.tokenizer.eos_token_id else "ok"
            )
            torch.cuda.synchronize()
        except Exception as error:
            packet["status"] = "generation_error"
            packet["error"] = type(error).__name__ + ": " + str(error)[:300]
            torch.cuda.empty_cache()
        finally:
            packet["elapsed_seconds"] = float(time.perf_counter() - started)
        return packet

    def run_direct(self, payload: dict, use_adapter: bool = False) -> dict:
        public = source_payload(payload)
        generation = self.generate(analyst_messages(public), use_adapter=use_adapter)
        return {"parsed": parse_packet(generation, public), "generation": generation}

    def validate_and_select(self, validation_frame) -> list[dict]:
        if self.protocol is not None:
            raise RuntimeError("Analyst selection is already frozen. Create a new run before selecting again.")
        self.assert_training_state(validation_frame)
        from .metrics import evaluate
        signature = {"training": self.training_signature,
                     "adapter_sha256": file_sha256(self.adapter_dir / "adapter_model.safetensors"),
                     "selection": "validation-macro-f1-ties-pretrained-v1"}
        rows = validation_frame.to_dict("records")
        cache = TraceCache(self.output_dir / "logs", "validation_direct", fingerprint(signature), rows)
        results = []
        for position, row in enumerate(rows, 1):
            result = cache.completed.get(row["id"])
            if result is None:
                result = {"pretrained": self.run_direct(row, use_adapter=False),
                          "finetuned": self.run_direct(row, use_adapter=True)}
                cache.append(row["id"], result)
            results.append(result)
            if position % 10 == 0 or position == len(rows):
                print(f"Validation direct: {position}/{len(rows)}", flush=True)
        gold = validation_frame["label"].tolist()
        metrics = [evaluate(gold, [row[mode]["parsed"]["label"] for row in results],
                            f"qwen_{mode}_validation") for mode in ("pretrained", "finetuned")]
        self.analyst_use_adapter = bool(metrics[1]["macro_f1"] > metrics[0]["macro_f1"])
        _write_json(self.output_dir / "tables" / "validation_direct_metrics.json", metrics)
        _write_csv(self.output_dir / "tables" / "validation_direct_predictions.csv", [
            {"id": row["id"], "gold": row["label"],
             "pretrained": result["pretrained"]["parsed"]["label"],
             "finetuned": result["finetuned"]["parsed"]["label"]}
            for row, result in zip(rows, results)
        ])
        return metrics

    def assert_training_state(self, validation_frame=None) -> None:
        """Do not reuse validation outputs if the live training definition has changed."""
        if self.training_signature is None or self.trained_adapter_sha256 is None:
            raise RuntimeError("Complete or restore the matching training run before validation selection.")
        current = {"policy": POLICY_VERSION, "code_sha256": file_sha256(Path(__file__)),
                   "config": asdict(self.config), "analyst_prompt": ANALYST_SYSTEM,
                   "versions": self.stack["versions"]}
        changed = [key for key, value in current.items()
                   if fingerprint(value) != fingerprint(self.training_signature.get(key))]
        if validation_frame is not None:
            if fingerprint(validation_frame.to_dict("records")) != self.training_signature["validation_sha256"]:
                changed.append("validation_source")
        if changed:
            raise RuntimeError("Runtime differs from the training snapshot before selection: "
                               + ", ".join(changed) + ". Use a new run.")
        if file_sha256(self.adapter_dir / "adapter_model.safetensors") != self.trained_adapter_sha256:
            raise RuntimeError("Saved adapter changed after training; validation caches cannot be reused.")

    def _policy_snapshot(self) -> dict:
        return {"policy": POLICY_VERSION, "config": asdict(self.config),
                "analyst_system": ANALYST_SYSTEM, "reviewer_system": REVIEWER_SYSTEM,
                "verifier_system": VERIFIER_SYSTEM,
                "analyst_use_adapter": self.analyst_use_adapter,
                "train_ids": self.train_ids, "validation_ids": self.validation_ids,
                "code_sha256": file_sha256(Path(__file__))}

    def freeze_protocol(self, data_audit: dict) -> dict:
        if self.protocol is not None:
            self.assert_protocol(check_adapter_file=True)
            if fingerprint(data_audit) != fingerprint(self.protocol["data_audit"]):
                raise RuntimeError("Data audit differs from the frozen protocol. Create a new run.")
            return copy.deepcopy(self.protocol)
        if self.training_signature is not None:
            self.assert_training_state()
        protocol = {**self._policy_snapshot(), "data_audit": data_audit,
                    "adapter_sha256": file_sha256(self.adapter_dir / "adapter_model.safetensors"),
                    "versions": self.stack["versions"], "gpu": self.stack["gpu"],
                    "gate": "valid-unique-source-quotes-and-reviewed-revision-v1"}
        proposed_hash = fingerprint(protocol)
        protocol_path = self.output_dir / "logs" / "frozen_gpu_protocol.json"
        if protocol_path.exists():
            previous = json.loads(protocol_path.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or not isinstance(previous.get("protocol"), dict):
                raise RuntimeError("The existing frozen protocol has an invalid schema; its file was retained.")
            if fingerprint(previous["protocol"]) != previous.get("sha256"):
                raise RuntimeError("The existing frozen protocol failed its checksum; its file was retained.")
            if previous["sha256"] != proposed_hash:
                raise RuntimeError("This output directory has a different frozen GPU protocol. "
                                   "Use a new output directory; the previous protocol was retained.")
        else:
            _write_json(protocol_path, {"sha256": proposed_hash, "protocol": protocol})
        self.protocol = copy.deepcopy(protocol)
        self.protocol_sha256 = proposed_hash
        return copy.deepcopy(protocol)

    def assert_protocol(self, check_adapter_file: bool = False) -> None:
        if self.protocol is None or self.protocol_sha256 is None:
            raise RuntimeError("Freeze the GPU protocol before MAS inference.")
        if fingerprint(self.protocol) != self.protocol_sha256:
            raise RuntimeError("The frozen protocol object was modified; create a new run.")
        for key, value in self._policy_snapshot().items():
            if fingerprint(value) != fingerprint(self.protocol.get(key)):
                raise RuntimeError(f"Runtime field {key} differs from the frozen protocol; old caches cannot be reused.")
        if check_adapter_file and file_sha256(self.adapter_dir / "adapter_model.safetensors") != self.protocol["adapter_sha256"]:
            raise RuntimeError("The adapter file differs from the frozen protocol.")

    def run_mas(self, payload: dict) -> dict:
        self.assert_protocol()
        public = source_payload(payload)
        direct = self.run_direct(public, use_adapter=self.analyst_use_adapter)
        initial = direct["parsed"]
        review_generation = self.generate([
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": _canonical_json({"input": public, "initial": initial})},
        ], use_adapter=False)
        reviewer = parse_packet(review_generation, public, require_quote=True)
        final_generation = self.generate([
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": _canonical_json(
                {"input": public, "initial": initial, "reviewer": reviewer})},
        ], use_adapter=False)
        final = parse_packet(final_generation, public, require_quote=True)
        applied = gate_packets(initial, reviewer, final)
        trace = [
            {"role": "analyst", "use_adapter": self.analyst_use_adapter,
             "parsed": initial, "generation": direct["generation"]},
            {"role": "reviewer", "use_adapter": False,
             "parsed": reviewer, "generation": review_generation},
            {"role": "verifier", "use_adapter": False,
             "parsed": final, "generation": final_generation},
        ]
        return {**applied, "direct_label": initial["label"], "trace": trace,
                **{field: sum(item["generation"][field] for item in trace)
                   for field in ("input_tokens", "output_tokens", "model_calls", "elapsed_seconds")}}

    def run_split(self, frame, split_name: str) -> list[dict]:
        self.assert_protocol(check_adapter_file=True)
        rows = frame.to_dict("records")
        cache = TraceCache(self.output_dir / "logs", split_name, self.protocol_sha256, rows)
        results = []
        for position, row in enumerate(rows, 1):
            result = cache.completed.get(row["id"])
            if result is None:
                result = self.run_mas(source_payload(row))
                cache.append(row["id"], result)
            results.append({"id": row["id"], **result})
            if position % 10 == 0 or position == len(rows):
                print(f"MAS {split_name}: {position}/{len(rows)}", flush=True)
        return results

    def reload_adapter(self, payload: dict) -> dict:
        """Check save/reload using a small second adapter on the same base model."""
        self.assert_protocol(check_adapter_file=True)
        original = self.model.active_adapter
        before = self.run_direct(payload, use_adapter=True)
        try:
            if "reload_check" not in self.model.peft_config:
                self.model.load_adapter(str(self.adapter_dir), adapter_name="reload_check", is_trainable=False)
            self.model.set_adapter("reload_check")
            after = self.run_direct(payload, use_adapter=True)
            first, second = before["parsed"]["label"], after["parsed"]["label"]
            if first not in LABELS or second not in LABELS or first != second:
                raise RuntimeError("Adapter reload did not produce the same valid stance label.")
            return {"before": first, "after": second, "adapter_dir": str(self.adapter_dir)}
        finally:
            self.model.set_adapter(original)
            self.model.eval()


def _split_metrics(frame, records: list[dict], split_name: str) -> list[dict]:
    from .metrics import evaluate
    gold = frame["label"].tolist()
    return [evaluate(gold, [row[key] for row in records], f"{name}_{split_name}")
            for key, name in (("direct_label", "qwen_analyst"), ("label", "qwen_mas"))]


def _cost_summary(records: list[dict]) -> dict:
    direct = {field: sum(row["trace"][0]["generation"][field] for row in records)
              for field in ("input_tokens", "output_tokens", "model_calls", "elapsed_seconds")}
    mas = {field: sum(row[field] for row in records)
           for field in ("input_tokens", "output_tokens", "model_calls", "elapsed_seconds")}
    return {"analyst": direct, "mas": mas,
            "flagged_for_review": sum(row["review"] for row in records),
            "invalid_analyst": sum(row["direct_label"] == INVALID for row in records),
            "invalid_mas": sum(row["label"] == INVALID for row in records)}


def selective_metrics(frame, records: list[dict]) -> dict:
    """Score analyst and MAS on the same accepted IDs, with explicit coverage.

    Selection is the existing review flag, not a new rule tuned on test labels.
    No accuracy or F1 is reported when the accepted subset is empty.
    """
    from .metrics import evaluate

    sources = frame.to_dict("records")
    source_ids = [row["id"] for row in sources]
    result_ids = [row["id"] for row in records]
    if not sources or len(set(source_ids)) != len(source_ids):
        raise ValueError("Selective scoring requires a nonempty source table with unique IDs.")
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != set(source_ids):
        raise ValueError("Selective predictions must contain exactly the same unique IDs as the source table.")
    by_id = {row["id"]: row for row in records}
    selected_sources = []
    selected_results = []
    for source in sources:
        result = by_id[source["id"]]
        if not isinstance(result.get("review"), bool):
            raise ValueError("Every selective prediction needs a Boolean review flag.")
        if source["label"] not in LABELS or any(result.get(key) not in LABELS + (INVALID,)
                                                for key in ("direct_label", "label")):
            raise ValueError("Selective scoring requires valid gold and prediction label codes.")
        if not result["review"]:
            selected_sources.append(source)
            selected_results.append(result)
    gold = [row["label"] for row in selected_sources]
    summary = {
        "mask_policy": "accept_when_review_is_false",
        "total": len(sources), "accepted_count": len(selected_sources),
        "flagged_count": len(sources) - len(selected_sources),
        "coverage": len(selected_sources) / len(sources),
        "accepted_ids": [row["id"] for row in selected_sources],
        "accepted_gold_support": {label: gold.count(label) for label in LABELS},
        "total_gold_support": {label: sum(row["label"] == label for row in sources) for label in LABELS},
        "metrics": [],
        "interpretation": "Both methods use exactly the same accepted rows; compare full-test scores separately.",
    }
    if gold:
        summary["metrics"] = [
            evaluate(gold, [row[key] for row in selected_results], name)
            for key, name in (("direct_label", "qwen_analyst_same_accepted_subset"),
                              ("label", "qwen_mas_same_accepted_subset"))
        ]
    return summary


def generate_gpu_figures(output_dir: Path, test_frame, records: list[dict],
                         test_metrics: list[dict]) -> dict:
    """Export measured GPU figure inputs and PNG/SVG files with source hashes."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    output_dir = Path(output_dir)
    figures_dir, tables_dir = output_dir / "figures", output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "renderer": "matplotlib",
                "matplotlib_version": matplotlib.__version__, "dpi": 300, "figures": []}

    def finish(figure, stem, rows, description):
        table_path = tables_dir / f"{stem}_data.csv"
        _write_csv(table_path, rows)
        figure.tight_layout()
        files = []
        for extension in ("png", "svg"):
            path = figures_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight",
                           metadata={"Creator": "MASHeart"} if extension == "svg" else None)
            files.append({"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
        plt.close(figure)
        manifest["figures"].append({"id": stem, "description": description,
                                    "data": {"path": str(table_path.relative_to(output_dir)),
                                             "sha256": file_sha256(table_path), "rows": len(rows)},
                                    "files": files})

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "MASHeart-GPU-v1"}):
        history_path = output_dir / "logs" / "training_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            loss_rows = [{"step": row["step"], "epoch": row.get("epoch"),
                          "loss": row.get("loss"), "eval_loss": row.get("eval_loss")}
                         for row in history if "step" in row and ("loss" in row or "eval_loss" in row)]
            if loss_rows:
                figure, axis = plt.subplots(figsize=(6.6, 3.7))
                for field, name, marker in [("loss", "Training loss", "."),
                                             ("eval_loss", "Validation loss", "o")]:
                    selected = [row for row in loss_rows if row[field] is not None]
                    if selected:
                        axis.plot([row["step"] for row in selected], [row[field] for row in selected],
                                  marker=marker, label=name)
                axis.set(title="QLoRA assistant-token loss", xlabel="Optimizer update step", ylabel="Cross-entropy loss")
                axis.legend(frameon=False)
                finish(figure, "gpu_01_training_loss", loss_rows,
                       "Loss is computed on answer tokens only; checkpoint selection uses validation loss.")

        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        positions = np.arange(len(test_metrics))
        for offset, field, color in [(-0.18, "accuracy", "#287da8"), (0.18, "macro_f1", "#c56c39")]:
            bars = axis.bar(positions + offset, [row[field] for row in test_metrics],
                            width=0.34, label=field.replace("_", " "), color=color)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_xticks(positions, [row["system"].replace("_test", "").replace("_", " ") for row in test_metrics])
        axis.set(title="Full-test GPU comparison", ylabel="Score", ylim=(0, 1.15))
        axis.legend(frameon=False)
        finish(figure, "gpu_02_test_metric_comparison", test_metrics,
               "Every official test row is scored, including INVALID outputs.")

        order = list(LABELS) + ([INVALID] if any(row["label"] == INVALID for row in records) else [])
        matrix = confusion_matrix(test_frame["label"], [row["label"] for row in records], labels=order)
        matrix_rows = [{"gold": gold, "prediction": prediction, "count": int(matrix[i, j])}
                       for i, gold in enumerate(order) for j, prediction in enumerate(order)]
        figure, axis = plt.subplots(figsize=(5.0, 4.1))
        axis.imshow(matrix, cmap="Blues")
        for i in range(len(order)):
            for j in range(len(order)):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        axis.set(xticks=range(len(order)), yticks=range(len(order)),
                 xticklabels=order, yticklabels=order, xlabel="Prediction", ylabel="Gold",
                 title="MAS with gate: full-test confusion matrix")
        finish(figure, "gpu_03_test_confusion_matrix", matrix_rows,
               "INVALID predictions remain visible as a separate column when present.")

        costs = _cost_summary(records)
        cost_rows = [{"system": name, **costs[name]} for name in ("analyst", "mas")]
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
        for axis, field, title in [(axes[0], "tokens", "Actual input + output tokens"),
                                  (axes[1], "elapsed_seconds", "Actual model-stage elapsed seconds")]:
            values = [(row["input_tokens"] + row["output_tokens"]) if field == "tokens" else row[field]
                      for row in cost_rows]
            bars = axis.bar([row["system"] for row in cost_rows], values, color=["#287da8", "#c56c39"])
            axis.bar_label(bars, fmt="%.1f", padding=3)
            axis.set(title=title, ylabel="Tokens" if field == "tokens" else "Seconds")
            axis.set_ylim(0, max(values + [1.0]) * 1.20)
        finish(figure, "gpu_04_test_cost", cost_rows,
               "Analyst cost is the first stage from the same MAS traces; totals include failed attempted calls.")

        edit_counts = {"Correct to correct": 0, "Correct to wrong": 0,
                       "Wrong to correct": 0, "Wrong to wrong": 0}
        edit_rows = []
        for source, row in zip(test_frame.to_dict("records"), records):
            initial_correct = row["direct_label"] == source["label"]
            final_correct = row["label"] == source["label"]
            category = ("Correct" if initial_correct else "Wrong") + " to " + ("correct" if final_correct else "wrong")
            edit_counts[category] += 1
            edit_rows.append({"id": source["id"], "group_id": source["group_id"],
                              "gold": source["label"], "analyst": row["direct_label"],
                              "mas": row["label"], "category": category})
        figure, axis = plt.subplots(figsize=(7.3, 3.7))
        bars = axis.bar(list(edit_counts), list(edit_counts.values()),
                        color=["#7c9c86", "#c56c39", "#287da8", "#8b929a"])
        axis.bar_label(bars, padding=3)
        axis.set(title="How review and verification change correctness", ylabel="Test examples")
        axis.set_ylim(0, max(edit_counts.values()) * 1.20 + 1)
        finish(figure, "gpu_05_revision_outcomes", edit_rows,
               "Paired initial/final correctness on the same test rows; a structural gate may still accept harmful edits.")
    _write_json(figures_dir / "gpu_figure_data_manifest.json", manifest)
    return manifest


def run_gpu_experiment(data_dir: Path, output_dir: Path, config: dict | None = None) -> dict:
    """Run real QLoRA/validation/MAS/test work or fail clearly before downloading.

    The complete test set is opened only after selection and protocol freeze.
    Every prediction, including INVALID, remains in the evaluation denominator.
    """
    resolved = GPUConfig.from_dict(config)
    stack = _require_gpu_stack(resolved)
    from .data import assert_disjoint, load_frame, split_development, verify_sources
    from .metrics import evaluate, paired_cluster_bootstrap
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    audit = verify_sources(data_dir)
    dev = load_frame(data_dir / "raw" / "a_dev.jsonl", "dev")
    train, validation = split_development(dev, seed=resolved.seed,
                                         validation_fraction=resolved.validation_fraction)
    assert_disjoint(train, validation)
    audit_path = output_dir / "logs" / "data_audit.json"
    if audit_path.exists() and fingerprint(json.loads(audit_path.read_text(encoding="utf-8"))) != fingerprint(audit):
        raise ValueError("Existing data audit differs from these sources. Use a new output directory; "
                         "the previous audit was retained.")
    runtime = T4Runtime(output_dir, resolved, stack=stack)
    runtime.train(train, validation)
    if not audit_path.exists():
        _write_json(audit_path, audit)
    validation_direct_metrics = runtime.validate_and_select(validation)
    runtime.freeze_protocol(audit)
    validation_records = runtime.run_split(validation, "validation_mas")
    validation_metrics = validation_direct_metrics + _split_metrics(validation, validation_records, "validation")
    # This is the first parsing of the official test rows in this experiment.
    test = load_frame(data_dir / "raw" / "a_test.jsonl", "test")
    assert_disjoint(train, test)
    assert_disjoint(validation, test)
    test_records = runtime.run_split(test, "test_mas")
    test_metrics = _split_metrics(test, test_records, "test")
    majority = train["label"].value_counts().index[0]
    test_metrics.insert(0, evaluate(test["label"].tolist(), [majority] * len(test), "majority_test"))
    bootstrap = paired_cluster_bootstrap(
        test["label"], [row["direct_label"] for row in test_records],
        [row["label"] for row in test_records], test["group_id"], rounds=500, seed=resolved.seed,
    )
    bootstrap_summary = {key: value for key, value in bootstrap.items() if key != "differences"}
    _write_json(output_dir / "tables" / "gpu_paired_bootstrap.json", bootstrap_summary)
    _write_csv(output_dir / "tables" / "gpu_paired_bootstrap_draws.csv",
               [{"draw": index, "macro_f1_difference": float(value)}
                for index, value in enumerate(bootstrap["differences"])])
    costs = {"validation": _cost_summary(validation_records), "test": _cost_summary(test_records)}
    selective = selective_metrics(test, test_records)
    _write_json(output_dir / "tables" / "gpu_metrics.json",
                {"validation": validation_metrics, "test": test_metrics})
    _write_json(output_dir / "tables" / "gpu_costs.json", costs)
    _write_json(output_dir / "tables" / "gpu_selective_metrics.json", selective)
    prediction_rows = []
    for source, prediction in zip(test.to_dict("records"), test_records):
        prediction_rows.append({
            "id": source["id"], "group_id": source["group_id"], "gold": source["label"],
            **{key: prediction[key] for key in ("direct_label", "label", "review", "gate_status",
                                                "input_tokens", "output_tokens", "model_calls", "elapsed_seconds")},
        })
    _write_csv(output_dir / "tables" / "gpu_test_predictions.csv", prediction_rows)
    figure_manifest = generate_gpu_figures(output_dir, test, test_records, test_metrics)
    summary = {
        "mode": "t4", "status": "completed", "model_id": resolved.model_id,
        "revision": resolved.revision, "gpu": stack["gpu"],
        "selected_analyst": "finetuned" if runtime.analyst_use_adapter else "pretrained",
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_groups": int(train["group_id"].nunique()),
        "validation_groups": int(validation["group_id"].nunique()),
        "test_groups": int(test["group_id"].nunique()),
        "training_seconds": runtime.training_seconds,
        "protocol_sha256": runtime.protocol_sha256,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics, "costs": costs,
        "selective": selective,
        "paired_bootstrap": bootstrap_summary, "figure_count": len(figure_manifest["figures"]),
        "artifacts": {"adapter": str(runtime.adapter_dir),
                      "protocol": str(output_dir / "logs" / "frozen_gpu_protocol.json"),
                      "predictions": str(output_dir / "tables" / "gpu_test_predictions.csv"),
                      "metrics": str(output_dir / "tables" / "gpu_metrics.json"),
                      "figures": str(output_dir / "figures" / "gpu_figure_data_manifest.json")},
    }
    runtime.assert_protocol(check_adapter_file=True)
    _write_json(output_dir / "reports" / "gpu_summary.json", summary)
    table = "\n".join(f"| {row.get('name', row.get('system', 'method'))} | {row['n']} | "
                      f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} |" for row in test_metrics)
    report = (
        "# MASHeart GPU experiment\n\n"
        f"Completed a real run of `{resolved.model_id}` at revision `{resolved.revision}`. "
        f"GPU: {stack['gpu']['name']}. Analyst selected on validation: {summary['selected_analyst']}.\n\n"
        f"Training/validation/test: {len(train)}/{len(validation)}/{len(test)} rows. "
        "The educational fit uses development groups; this is not the original Qwen3-8B v4 protocol.\n\n"
        "| Method | N | Accuracy | Macro-F1 |\n|---|---:|---:|---:|\n" + table + "\n\n"
        "All test rows remain in the denominator, including invalid outputs. "
        "Reviewer and verifier use the same pretrained base with the classification adapter disabled. "
        "Quotes are checked for exact unique source occurrence; this does not establish semantic truth. "
        "The structural gate may keep a valid initial answer when a revision fails its checks, "
        "and every such fallback is recorded. No generation was retried to improve an answer.\n\n"
        f"The fixed review flag accepted {selective['accepted_count']}/{selective['total']} test rows "
        f"(coverage {selective['coverage']:.2%}). Selective analyst/MAS metrics in "
        "`tables/gpu_selective_metrics.json` use exactly the same accepted IDs and report both class supports. "
        "If none are accepted, that table contains an empty metrics list. Selective scores describe a "
        "smaller subset and must not be compared with another method's full-test score as a performance gain.\n\n"
        "Metrics describe stance toward supplied questions, not clinical validity, diagnostic accuracy, "
        "or treatment advice. The data and project have prior development/test exposure. "
        "A higher point score alone does not prove that MAS is superior.\n\n"
        "Configuration and tokenizer APIs follow the [Qwen model card]"
        "(https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), "
        "[PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization), "
        "and [Trainer 4.57.1 documentation]"
        "(https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).\n"
    )
    (output_dir / "reports" / "GPU_REPORT.md").write_text(report, encoding="utf-8")
    return summary
