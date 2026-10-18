"""CPU integration of run artifacts and resume; no model downloads or GPU claims."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import pytest

from less.train import online_proxy
from less.train.online_proxy_io import read_jsonl
from test_online_proxy_fd import ScalarLM, record


class SavedScalarLM(ScalarLM):
    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory) / "toy_weights.pt")
        (Path(directory) / "adapter_config.json").write_text("{}")


class ToyTokenizer:
    pad_token_id = 0

    def get_chat_template(self):
        return "toy-test-only"

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_config.json").write_text("{}")


def test_run_outputs_and_resume_match_uninterrupted_training(tmp_path, monkeypatch):
    module = ModuleType("less.data_selection.get_training_dataset")
    module.encode_with_qwen_chat_format = lambda raw, tokenizer, limit: {key: raw[key] for key in
                                                                      ("input_ids", "labels", "attention_mask")}
    monkeypatch.setitem(sys.modules, "less.data_selection.get_training_dataset", module)
    monkeypatch.setattr(online_proxy, "encode_reference_records", lambda *arguments: [record(99)])
    def load_model(args, device):
        model = SavedScalarLM().to(device)
        if args.resume_from:
            model.load_state_dict(torch.load(args.resume_from / "toy_weights.pt", weights_only=True))
        return model, ToyTokenizer()
    monkeypatch.setattr(online_proxy, "load_model", load_model)
    source = tmp_path / "pool.jsonl"
    source.write_text("".join(json.dumps(dict(record(index, conflict=index % 2 == 1),
        messages=[dict(role="assistant", content="synthetic-test-fixture")])) + "\n" for index in range(12)))
    def run(name, stop, resume=None):
        arguments = ["--model-path", "toy-test-only", "--task", "mmlu", "--train-files", str(source),
            "--output-dir", str(tmp_path / name), "--device", "cpu", "--batch-size", "4", "--filter", "fd",
            "--max-batches", str(stop), "--save-every-batches", "1"]
        if resume:
            arguments += ["--resume-from", str(resume)]
        args = online_proxy.parser().parse_args(arguments)
        online_proxy.validate(args)
        online_proxy.execute(args)
        return tmp_path / name
    full = run("full", 2)
    first = run("first", 1)
    continuation = run("continuation", 2, first / "snapshot-batch-1")
    actual = torch.load(continuation / "model" / "toy_weights.pt", weights_only=True)
    expected = torch.load(full / "model" / "toy_weights.pt", weights_only=True)
    assert torch.equal(actual["weight"], expected["weight"])
    assert json.loads((continuation / "status.json").read_text())["status"] == "completed"
    assert len(list(read_jsonl(full / "decisions.jsonl"))) >= 8
    assert json.loads((full / "run.json").read_text())["no_less_execution"]
    assert json.loads((full / "train_results.json").read_text())["full_pool_pass_completed"] is False
    assert json.loads((continuation / "train_results.json").read_text())["initial_batch"] == 1


def test_initialization_failure_still_records_config_and_traceback(tmp_path):
    args = online_proxy.parser().parse_args(["--model-path", "not-loaded", "--task", "mmlu",
        "--train-files", str(tmp_path / "missing.jsonl"), "--output-dir", str(tmp_path / "failed"), "--device", "cpu"])
    online_proxy.validate(args)
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        online_proxy.execute(args)
    run = json.loads((tmp_path / "failed" / "run.json").read_text())
    status = json.loads((tmp_path / "failed" / "status.json").read_text())
    assert run["config"]["task"] == "mmlu" and run["no_less_execution"]
    assert status["status"] == "failed" and "FileNotFoundError" in status["traceback"]


def test_adapter_loading_uses_base_metadata_and_resume_resolves_original_base(tmp_path, monkeypatch):
    import less.model_utils
    class Adapter(ScalarLM):
        peft_config = {"default": SimpleNamespace(r=16, lora_alpha=64)}
        config = SimpleNamespace(use_cache=False)
        def named_parameters(self, *args, **kwargs):
            for name, parameter in super().named_parameters(*args, **kwargs):
                yield "lora_" + name, parameter
    peft = ModuleType("peft")
    peft.PeftModel = Adapter
    peft.LoraConfig = lambda **kwargs: None
    peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft.get_peft_model = lambda *args, **kwargs: Adapter()
    monkeypatch.setitem(sys.modules, "peft", peft)
    calls = []
    def loader(path, **kwargs):
        calls.append(kwargs["base_model_path"])
        return Adapter(), ToyTokenizer()
    monkeypatch.setattr(less.model_utils, "load_model_and_tokenizer", loader)
    initial = tmp_path / "initial-adapter"
    initial.mkdir()
    (initial / "adapter_config.json").write_text(json.dumps(dict(base_model_name_or_path="actual-base")))
    args = SimpleNamespace(model_path=str(initial), base_model_path=None, resume_from=None,
                           dtype="float32", lora_r=16, lora_alpha=64)
    online_proxy.load_model(args, torch.device("cpu"))
    assert calls[-1] is None  # let the shared loader resolve adapter metadata
    args.resume_from = tmp_path / "resume"
    online_proxy.load_model(args, torch.device("cpu"))
    assert calls[-1] == "actual-base"
