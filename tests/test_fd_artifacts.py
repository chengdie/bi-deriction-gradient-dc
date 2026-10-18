import json

import pytest

from less.analysis.fd_fidelity import choose_configuration, fidelity_metrics
from less.analysis.summarize_fd import holm, mcnemar_exact, summarize
from less.data_selection.prepare_fd_pool import prepare
from less.train.online_proxy_io import JsonlPool, RunWriter, inspect_inputs, read_jsonl, write_json


def test_pool_is_reproducible_and_preserves_source_identity(tmp_path):
    files = []
    for source in range(2):
        file = tmp_path / f"source-{source}.jsonl"
        file.write_text("".join(json.dumps(dict(id=f"{source}-{row}", prompt="question", completion="answer")) + "\n"
                                for row in range(12)))
        files.append(file)
    a = prepare(files, tmp_path / "a", size=9)
    b = prepare(files, tmp_path / "b", size=9)
    assert a["pool_sha256"] == b["pool_sha256"] and a["rows"] == b["rows"]
    pool = JsonlPool([tmp_path / "a" / "pool.jsonl"])
    assert len(pool) == 9
    original = [list(read_jsonl(file)) for file in files]
    for mapping in a["rows"]:
        raw, identifiers = pool[mapping["pool_index"]]
        assert raw == original[mapping["source_index"]][mapping["source_row"]]
        assert identifiers["pool_index"] == mapping["pool_index"]
    with pytest.raises(FileExistsError):
        prepare(files, tmp_path / "a", size=9)


def test_lfs_pointer_is_identified_before_json_parsing(tmp_path):
    pointer = tmp_path / "data.jsonl"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 100\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        inspect_inputs([pointer])


def test_fidelity_distinguishes_zero_delta_from_matching_decisions():
    rows = [dict(cos=0.1, dot=1.0, gradient_norm=10.0, delta=0.0, dir_deriv=0.0),
            dict(cos=-0.1, dot=-1.0, gradient_norm=10.0, delta=1e-3, dir_deriv=-1.0)]
    result = fidelity_metrics(rows)
    assert result["sign_agree"] == 0.5
    assert result["decision_agree"] == 1.0
    assert result["zero_delta_rate"] == 0.5
    assert result["roc_auc"] == 1.0


def test_calibration_gates_actual_choice_and_requires_three_points():
    def report(point):
        return dict(point_sha256=point, dtype="bfloat16", grid=[
            dict(epsilon=0.01, scheme="central", sign_agree=0.905, strong_sign_agree=0.98, strong_n=100),
            dict(epsilon=0.01, scheme="forward", sign_agree=0.899, strong_sign_agree=0.98, strong_n=100)])
    choice = choose_configuration([report(str(index)) for index in range(3)])
    assert choice["scheme"] == "forward" and not choice["passed"]
    report_a = report("a")
    report_a["grid"][1]["sign_agree"] = 0.95
    assert not choose_configuration([report_a])["passed"]


def test_multiple_comparison_and_mcnemar():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.05}) == {"a": 0.03, "b": 0.08, "c": 0.08}
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)


def test_raw_artifacts_reconstruct_cost_and_partial_runs_excluded(tmp_path):
    from test_online_proxy_fd import trainer, record
    for mode in ("fd", "cosine"):
        directory = tmp_path / mode
        writer = RunWriter(directory)
        instance = trainer(mode)
        row = instance.process_batch([record(0), record(1, conflict=True)], 0)
        writer.batch(row)
        writer.close()
        config = dict(filter=mode, task="mmlu", seed=3, model_seed=0, batch_size=16, max_length=1024,
            reference_max_length=4096, learning_rate=1e-5, clip_norm=1.0, lora_r=16, lora_alpha=64,
            dtype="bfloat16", epochs=1, max_batches=1, resume_from=None)
        identity = dict(config=dict(config), reference_sha256="ref", chat_template_sha256="template")
        write_json(directory / "run.json", dict(schema_version="online-proxy-fd-v1", config=config,
            identity=identity, inputs=[dict(rows=2, sha256="test-fixture")], world_size=1, environment=dict(gpu=None)))
        write_json(directory / "status.json", dict(status="completed"))
        write_json(directory / "train_results.json", dict(completed=True, batches=1, candidates=2, accepted=1,
            updates=1, refresh_count=1, wall_seconds=10, gpu_hours=0, full_pool_pass_completed=False))
    tables = summarize(tmp_path, tmp_path / "report", repetitions=10, warmup_batches=0, plots=True)
    assert len(tables["runs"]) == 2 and not tables["main_results"]
    assert tables["runs"][0]["accepted"] == 1
    assert tables["decision_count"] == 4
    assert tables["speedups"][0]["wall_speedup"] == 1
    assert (tmp_path / "report" / "tables" / "decisions.csv").is_file()
    assert (tmp_path / "report" / "figures" / "retention.png").stat().st_size > 1000
