import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from evaluation.eval.less_protocol import extract_answer, load_eval_data, prompt, score
from evaluation.eval.qwen import merge_predictions, prepare_records


class CharacterTokenizer:
    chat_template = "fake-native-template"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode(errors="ignore")

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


def mmlu_fixture(tmp_path):
    for split in ("dev", "test"):
        directory = tmp_path / "eval" / "mmlu" / split
        directory.mkdir(parents=True)
        (directory / f"test_subject_{split}.csv").write_text("Q,A1,B1,C1,D1,A\n" * (5 if split == "dev" else 2))


def test_legacy_mmlu_prompt_matches_original_formatter(tmp_path):
    mmlu_fixture(tmp_path)
    records, protocol = load_eval_data("mmlu", tmp_path)
    text, shots = prompt(records[0], CharacterTokenizer(), 8192)
    assert text.startswith("The following are multiple choice questions (with answers) about  test subject.\n\n")
    assert text.count("Answer: A") == 5 and text.endswith("Answer:") and shots == 5
    assert protocol["aggregation"].startswith("question micro")
    assert protocol["test_records"] == 2  # legacy does not silently remove overlapping questions


def test_legacy_bbh_extraction_and_punctuation_follow_original_script():
    assert extract_answer("Reasoning. So the answer is (B). More text", "bbh") == "(B)"
    assert score("bbh", "(B)", ["B"])["exact_match"] == 1
    assert score("bbh", "-2", ["2"])["exact_match"] == 1  # legacy ignores punctuation


def test_global_preparation_mmlu_shot_reduction_is_monotone(tmp_path):
    mmlu_fixture(tmp_path)
    records, _ = load_eval_data("mmlu", tmp_path)
    records[0]["question"] = "long" * 430
    prepared = prepare_records(records, CharacterTokenizer(), "mmlu", 8192, 512)
    used = {row["record"]["id"]: row["info"]["effective_shots"] for row in prepared}
    assert used[records[1]["id"]] <= used[records[0]["id"]]


def test_missing_predictions_are_rejected(tmp_path):
    (tmp_path / "predictions.rank0.jsonl").write_text("")
    with pytest.raises(ValueError, match="Missing"):
        merge_predictions(tmp_path, [dict(id="missing", answers=["A"], subtask="s")], 1, "mmlu")


def test_evaluation_writes_legacy_and_clean_scores_separately(tmp_path, monkeypatch):
    import torch
    from evaluation.eval.qwen import run_worker
    import less.model_utils
    mmlu_fixture(tmp_path)
    test = tmp_path / "eval" / "mmlu" / "test" / "test_subject_test.csv"
    test.write_text("Q,A1,B1,C1,D1,A\nNewQ,A1,B1,C1,D1,B\n")
    records, protocol = load_eval_data("mmlu", tmp_path)
    class Batch(dict):
        def to(self, device):
            return {key: value.to(device) for key, value in self.items()}
    class Tokenizer(CharacterTokenizer):
        eos_token_id, pad_token_id = 2, 0
        def get_vocab(self):
            return {}
        def get_chat_template(self):
            return self.chat_template
        def pad(self, values, **kwargs):
            width = max(map(len, values["input_ids"]))
            return Batch({key: torch.tensor([[0] * (width - len(row)) + row for row in rows])
                          for key, rows in values.items()})
    class Model:
        config = SimpleNamespace(model_type="toy", max_position_embeddings=8192)
        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros(*input_ids.shape, 256)
            logits[:, :, ord("A")] = 1
            return SimpleNamespace(logits=logits)
    transformers = ModuleType("transformers")
    transformers.GenerationConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(less.model_utils, "load_model_and_tokenizer", lambda *args, **kwargs: (Model(), Tokenizer()))
    output = tmp_path / "result"
    args = SimpleNamespace(output_dir=output, seed=42, model_path="toy", base_model_path=None,
        dtype="float32", device="cpu", protocol="less", task="mmlu", max_length=8192, max_new_tokens=512, batch_size=2)
    run_worker(args, records, protocol, 0, 1, torch.distributed, torch)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["score"] == 50 and metrics["clean_score"] == 0
    assert metrics["n"] == 2 and metrics["clean_n"] == 1 and metrics["reference_overlap_n"] == 1
    assert (output / "eval_manifest.json").is_file()
