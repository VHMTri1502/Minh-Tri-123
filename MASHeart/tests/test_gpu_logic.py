"""CPU tests of GPU-independent safeguards; these are not GPU performance runs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from masheart.gpu import (
    GPUConfig, GPUUnavailableError, T4Runtime, TraceCache, collate_examples,
    encode_supervised, file_sha256, fingerprint, gate_packets, parse_packet,
    run_gpu_experiment, selective_metrics,
)


SOURCE = {"question": "Does treatment X lower blood pressure?",
          "text": "The synthetic trial did not lower blood pressure after eight weeks."}


def generation(value, status="ok"):
    return {"status": status, "raw": json.dumps(value)}


def parsed(label, quote="not lower blood pressure", source=SOURCE):
    return parse_packet(generation({"label": label, "quote": quote,
                                    "rationale": "A synthetic rationale for this unit test."}),
                        source, require_quote=True)


class FakeTokenizer:
    """Known token lengths make prompt-mask and EOS mistakes observable."""

    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        self.messages = messages
        return [10, 20, 30]

    def __call__(self, answer, add_special_tokens):
        assert not add_special_tokens
        self.answer = answer
        return {"input_ids": [41, 42]}


class ParserAndGateTests(unittest.TestCase):
    def test_strict_json_and_failures_are_not_repaired(self):
        for value in ["not JSON", '```json\n{"label":"NO"}\n```', '{"label":"MAYBE"}']:
            with self.subTest(value=value):
                self.assertEqual(parse_packet({"status": "ok", "raw": value}, SOURCE)["label"], "INVALID")
        self.assertEqual(parse_packet(generation({"label": "NO"}, "truncated"), SOURCE)["label"], "INVALID")

    def test_source_quote_keeps_negation_and_requires_unique_exact_occurrence(self):
        self.assertTrue(parsed("NO")["quote_valid"])
        self.assertFalse(parsed("NO", "Not lower blood pressure")["quote_valid"])
        repeated = {**SOURCE, "text": "not lower; not lower"}
        self.assertFalse(parsed("NO", "not lower", repeated)["quote_valid"])
        self.assertFalse(parsed("NO", "ana", {**SOURCE, "text": "banana"})["quote_valid"])
        self.assertFalse(parsed("NO", "")["schema_valid"])
        missing_reason = parse_packet(generation({"label": "NO", "quote": "not lower"}),
                                      SOURCE, require_quote=True)
        self.assertFalse(missing_reason["schema_valid"])

    def test_revision_without_grounded_review_is_blocked(self):
        initial = parse_packet(generation({"label": "YS"}), SOURCE)
        result = gate_packets(initial, parsed("NO", "invented words"), parsed("NO"))
        self.assertEqual(result["label"], "YS")
        self.assertTrue(result["review"])
        self.assertEqual(result["gate_status"], "blocked_revision_or_invalid_final")

    def test_supported_revision_and_invalid_initial_are_distinguished(self):
        initial = parse_packet(generation({"label": "YS"}), SOURCE)
        revised = gate_packets(initial, parsed("NO"), parsed("NO"))
        self.assertEqual(revised["label"], "NO")
        self.assertEqual(revised["gate_status"], "accepted_sourced_revision")
        invalid = parse_packet({"status": "generation_error", "raw": ""}, SOURCE)
        repaired = gate_packets(invalid, parsed("NO"), parsed("NO"))
        self.assertEqual(repaired["gate_status"], "repaired_invalid_initial")
        self.assertTrue(repaired["review"])
        self.assertEqual(gate_packets(invalid, invalid, invalid)["label"], "INVALID")

    def test_disagreement_is_flagged_when_final_keeps_initial(self):
        initial = parse_packet(generation({"label": "YS"}), SOURCE)
        result = gate_packets(initial, parsed("NO"), parsed("YS"))
        self.assertEqual(result["label"], "YS")
        self.assertTrue(result["review"])


class TokenMaskTests(unittest.TestCase):
    def test_assistant_only_mask_includes_last_answer_token_and_eos(self):
        tokenizer = FakeTokenizer()
        row = {**SOURCE, "id": "example", "label": "NO", "group_id": "hidden-group"}
        encoded = encode_supervised(row, tokenizer, max_tokens=8)
        self.assertEqual(encoded["input_ids"], [10, 20, 30, 41, 42, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 41, 42, 99])
        self.assertEqual(encoded["attention_mask"], [1] * 6)
        public = json.loads(tokenizer.messages[-1]["content"])
        self.assertEqual(set(public), {"question", "text"})
        self.assertNotIn("label", public)
        self.assertNotIn("hidden-group", tokenizer.messages[-1]["content"])

    def test_long_source_is_rejected_without_silent_target_truncation(self):
        with self.assertRaisesRegex(ValueError, "not truncated"):
            encode_supervised({**SOURCE, "label": "NO"}, FakeTokenizer(), max_tokens=5)

    def test_padding_cannot_train_on_pad_tokens_or_erase_answer_eos(self):
        first = encode_supervised({**SOURCE, "label": "NO"}, FakeTokenizer(), max_tokens=8)
        second = {"input_ids": [1, 2, 99], "attention_mask": [1, 1, 1], "labels": [-100, 2, 99]}
        batch = collate_examples([first, second], pad_token_id=99)
        self.assertEqual(batch["labels"][0], [-100, -100, -100, 41, 42, 99, -100, -100])
        self.assertEqual(batch["labels"][1], [-100, 2, 99, -100, -100, -100, -100, -100])
        self.assertEqual(batch["attention_mask"][1], [1, 1, 1, 0, 0, 0, 0, 0])


class SelectiveScoringTests(unittest.TestCase):
    def test_shared_mask_aligns_ids_and_denominators_for_both_methods(self):
        import pandas as pd

        frame = pd.DataFrame([{"id": "a", "label": "NO"},
                              {"id": "b", "label": "YS"},
                              {"id": "c", "label": "YS"}])
        # Deliberately shuffled: ID alignment must determine the gold/prediction pairing.
        records = [
            {"id": "c", "direct_label": "YS", "label": "NO", "review": True},
            {"id": "b", "direct_label": "NO", "label": "YS", "review": False},
            {"id": "a", "direct_label": "NO", "label": "NO", "review": False},
        ]
        result = selective_metrics(frame, records)
        self.assertEqual(result["accepted_ids"], ["a", "b"])
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["accepted_count"], 2)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["accepted_gold_support"], {"NO": 1, "YS": 1})
        direct, mas = result["metrics"]
        self.assertEqual(direct["n"], mas["n"])
        self.assertEqual(direct["n"], 2)
        self.assertEqual(direct["accuracy"], 0.5)
        self.assertEqual(mas["accuracy"], 1.0)
        self.assertEqual(direct["support_NO"], mas["support_NO"])
        self.assertEqual(direct["support_YS"], mas["support_YS"])

    def test_none_accepted_has_no_fabricated_accuracy_or_f1(self):
        import pandas as pd

        frame = pd.DataFrame([{"id": "a", "label": "NO"}])
        records = [{"id": "a", "direct_label": "INVALID", "label": "NO", "review": True}]
        result = selective_metrics(frame, records)
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["accepted_gold_support"], {"NO": 0, "YS": 0})
        self.assertEqual(result["metrics"], [])
        self.assertNotIn("accuracy", result)
        self.assertNotIn("macro_f1", result)
        with self.assertRaisesRegex(ValueError, "same unique IDs"):
            selective_metrics(frame, [{**records[0], "id": "different"}])


class CheckpointAndFreezeTests(unittest.TestCase):
    def test_fresh_runtime_restores_identical_freeze_but_never_overwrites_different_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stack = {"versions": {}, "gpu": {"name": "unit-test-fixture"}}
            first = T4Runtime(output, stack=stack)
            first.adapter_dir.mkdir(parents=True)
            (first.adapter_dir / "adapter_model.safetensors").write_bytes(b"fixture-only-not-a-model")
            first.freeze_protocol({"fixture_only": True})
            protocol_file = output / "logs" / "frozen_gpu_protocol.json"
            before = protocol_file.read_bytes()
            before_mtime = protocol_file.stat().st_mtime_ns
            identical = T4Runtime(output, stack=stack)
            identical.freeze_protocol({"fixture_only": True})
            self.assertEqual(identical.protocol_sha256, first.protocol_sha256)
            self.assertEqual(protocol_file.stat().st_mtime_ns, before_mtime)
            changed = T4Runtime(output, stack=stack)
            with patch("masheart.gpu.REVIEWER_SYSTEM", "changed reviewer prompt"):
                with self.assertRaisesRegex(RuntimeError, "previous protocol was retained"):
                    changed.freeze_protocol({"fixture_only": True})
            self.assertEqual(protocol_file.read_bytes(), before)
            self.assertIsNone(changed.protocol)

    def test_corrupt_persistent_freeze_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stack = {"versions": {}, "gpu": {"name": "unit-test-fixture"}}
            runtime = T4Runtime(output, stack=stack)
            runtime.adapter_dir.mkdir(parents=True)
            (runtime.adapter_dir / "adapter_model.safetensors").write_bytes(b"fixture-only-not-a-model")
            runtime.freeze_protocol({"fixture_only": True})
            path = output / "logs" / "frozen_gpu_protocol.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["sha256"] = "corrupt"
            path.write_text(json.dumps(record), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                T4Runtime(output, stack=stack).freeze_protocol({"fixture_only": True})
            self.assertEqual(path.read_bytes(), before)

    def test_changed_data_resume_cannot_overwrite_prior_audit(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            logs = output / "logs"
            logs.mkdir()
            audit_path = logs / "data_audit.json"
            audit_path.write_text('{"release":"original-fixture"}\n', encoding="utf-8")
            before = audit_path.read_bytes()
            frame = pd.DataFrame([{**SOURCE, "id": "a", "label": "NO"}])
            with patch("masheart.gpu._require_gpu_stack", return_value={}), \
                    patch("masheart.data.verify_sources", return_value={"release": "changed-fixture"}), \
                    patch("masheart.data.load_frame", return_value=frame), \
                    patch("masheart.data.split_development", return_value=(frame, frame)), \
                    patch("masheart.data.assert_disjoint", return_value={"passed": True}), \
                    patch.object(T4Runtime, "train") as train:
                with self.assertRaisesRegex(ValueError, "previous audit was retained"):
                    run_gpu_experiment(output / "unused-data", output)
                train.assert_not_called()
            self.assertEqual(audit_path.read_bytes(), before)

    def test_incompatible_training_manifest_is_checked_before_model_download(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = output / "logs" / "training_manifest.json"
            manifest.parent.mkdir()
            manifest.write_text('{"previous":"different-training"}\n', encoding="utf-8")
            before = manifest.read_bytes()
            frame = pd.DataFrame([{**SOURCE, "id": "a", "label": "NO"}])
            runtime = T4Runtime(output, stack={"versions": {}})
            with patch.object(runtime, "load_model") as loader:
                with self.assertRaisesRegex(ValueError, "different training protocol"):
                    runtime.train(frame, frame)
                loader.assert_not_called()
            self.assertEqual(manifest.read_bytes(), before)

    def test_training_snapshot_blocks_changed_prompt_before_validation_cache(self):
        import pandas as pd
        import masheart.gpu as gpu

        with tempfile.TemporaryDirectory() as temporary:
            runtime = T4Runtime(Path(temporary), stack={"versions": {}})
            runtime.adapter_dir.mkdir(parents=True)
            adapter = runtime.adapter_dir / "adapter_model.safetensors"
            adapter.write_bytes(b"fixture-only-not-a-model")
            validation = pd.DataFrame([{**SOURCE, "id": "a", "label": "NO"}])
            runtime.trained_adapter_sha256 = file_sha256(adapter)
            runtime.training_signature = {
                "policy": gpu.POLICY_VERSION, "code_sha256": file_sha256(Path(gpu.__file__)),
                "config": gpu.asdict(runtime.config), "analyst_prompt": gpu.ANALYST_SYSTEM,
                "versions": {}, "validation_sha256": fingerprint(validation.to_dict("records")),
            }
            runtime.assert_training_state(validation)
            with patch("masheart.gpu.ANALYST_SYSTEM", "changed after training"):
                with self.assertRaisesRegex(RuntimeError, "analyst_prompt"):
                    runtime.validate_and_select(validation)
            self.assertFalse((Path(temporary) / "logs").exists())

    def test_resume_is_bound_to_ordered_sources_and_protocol(self):
        sources = [{"id": "a", **SOURCE}, {"id": "b", **SOURCE, "text": "Another synthetic source."}]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = TraceCache(directory, "test", "protocol-one", sources)
            first.append("a", {"label": "INVALID", "status": "recorded_failure"})
            resumed = TraceCache(directory, "test", "protocol-one", sources)
            self.assertEqual(resumed.completed["a"]["label"], "INVALID")
            self.assertNotEqual(first.path, TraceCache(directory, "test", "protocol-two", sources).path)
            self.assertNotEqual(first.path, TraceCache(directory, "test", "protocol-one", list(reversed(sources))).path)
            changed = [{**sources[0], "text": "Changed source."}, sources[1]]
            self.assertNotEqual(first.path, TraceCache(directory, "test", "protocol-one", changed).path)
            with self.assertRaisesRegex(ValueError, "already completed"):
                resumed.append("a", {"label": "NO"})

    def test_partial_jsonl_is_retained_and_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = [{"id": "a", **SOURCE}]
            cache = TraceCache(directory, "test", "protocol", sources)
            cache.path.write_text('{"id":', encoding="utf-8")
            before = cache.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "retained"):
                TraceCache(directory, "test", "protocol", sources)
            self.assertEqual(cache.path.read_bytes(), before)

    def test_corrupt_cached_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = [{"id": "a", **SOURCE}]
            cache = TraceCache(directory, "test", "protocol", sources)
            cache.append("a", {"label": "NO"})
            record = json.loads(cache.path.read_text(encoding="utf-8"))
            record["payload_sha256"] = "corrupt"
            cache.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source and protocol"):
                TraceCache(directory, "test", "protocol", sources)

    def test_corrupt_result_is_detected_even_when_json_remains_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = [{"id": "a", **SOURCE}]
            cache = TraceCache(directory, "test", "protocol", sources)
            cache.append("a", {"label": "NO"})
            record = json.loads(cache.path.read_text(encoding="utf-8"))
            record["result"]["label"] = "YS"
            cache.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                TraceCache(directory, "test", "protocol", sources)

    def test_frozen_runtime_rejects_changed_prompts_mode_adapter_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = T4Runtime(Path(temporary), stack={"versions": {}, "gpu": {"name": "unit-test-fixture"}})
            runtime.adapter_dir.mkdir(parents=True)
            adapter = runtime.adapter_dir / "adapter_model.safetensors"
            adapter.write_bytes(b"fixture-only-not-a-model")
            runtime.train_ids, runtime.validation_ids = ["train"], ["val"]
            runtime.freeze_protocol({"fixture_only": True})
            runtime.assert_protocol(check_adapter_file=True)
            with patch("masheart.gpu.ANALYST_SYSTEM", "changed prompt"):
                with self.assertRaisesRegex(RuntimeError, "analyst_system"):
                    runtime.assert_protocol()
            runtime.analyst_use_adapter = True
            with self.assertRaisesRegex(RuntimeError, "analyst_use_adapter"):
                runtime.assert_protocol()
            runtime.analyst_use_adapter = False
            runtime.config = GPUConfig(max_new_tokens=2)
            with self.assertRaisesRegex(RuntimeError, "config"):
                runtime.assert_protocol()
            runtime.config = GPUConfig()
            adapter.write_bytes(b"modified-fixture")
            with self.assertRaisesRegex(RuntimeError, "adapter"):
                runtime.assert_protocol(check_adapter_file=True)

    def test_mas_has_three_calls_and_only_public_source_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = T4Runtime(Path(temporary), stack={"versions": {}, "gpu": {"name": "unit-test-fixture"}})
            runtime.adapter_dir.mkdir(parents=True)
            (runtime.adapter_dir / "adapter_model.safetensors").write_bytes(b"fixture-only-not-a-model")
            runtime.analyst_use_adapter = True
            runtime.freeze_protocol({"fixture_only": True})

            def measured(value):
                return {**generation(value), "input_tokens": 1, "output_tokens": 1,
                        "model_calls": 1, "elapsed_seconds": 0.01}

            review = {"label": "NO", "quote": "not lower blood pressure", "rationale": "Synthetic test rationale."}
            runtime.generate = Mock(side_effect=[measured({"label": "YS"}), measured(review), measured(review)])
            result = runtime.run_mas({**SOURCE, "label": "NO", "group_id": "hidden-group"})
            self.assertEqual(runtime.generate.call_count, 3)
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["label"], "NO")
            self.assertEqual([call.kwargs["use_adapter"] for call in runtime.generate.call_args_list],
                             [True, False, False])
            for call in runtime.generate.call_args_list:
                serialized = json.dumps(call.args[0])
                self.assertNotIn("hidden-group", serialized)
                self.assertNotIn("group_id", serialized)
            first_input = json.loads(runtime.generate.call_args_list[0].args[0][-1]["content"])
            self.assertEqual(first_input, SOURCE)

    def test_no_gpu_fails_before_data_reads_output_or_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-be-created"
            with patch("masheart.gpu._require_gpu_stack", side_effect=GPUUnavailableError("No CUDA GPU")) as guard:
                with self.assertRaisesRegex(GPUUnavailableError, "No CUDA GPU"):
                    run_gpu_experiment(Path(temporary) / "missing-data", output)
            guard.assert_called_once()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
