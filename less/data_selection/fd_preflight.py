"""Read-only data/artifact audit before GPU runs; stdlib only, no LESS execution."""

import argparse
import json
from pathlib import Path

from less.data_selection.target_data import _raw, load_eval_data as native_eval
from evaluation.eval.less_protocol import load_eval_data as less_eval
from less.train.online_proxy_io import inspect_inputs, sha256_file, utc_now, write_json


def audit(data_dir, artifact_root=None, model_path=None):
    data_dir = Path(data_dir)
    sources, evaluations, artifacts = [], [], []
    for name in ("flan_v2", "cot", "dolly", "oasst1"):
        path = data_dir / "train" / "processed" / name / f"{name}_data.jsonl"
        try:
            sources.append(dict(ready=True, **inspect_inputs([path])[0]))
        except (ValueError, OSError) as exc:
            sources.append(dict(path=str(path.resolve()), ready=False, error=str(exc)))
    for task in ("mmlu", "bbh", "tydiqa"):
        try:
            references, tests = _raw(task, data_dir)
            key = lambda row: " ".join(row["question"].casefold().split())
            questions = {key(row) for row in references}
            overlap = [row["id"] for row in tests if key(row) in questions]
            _, native = native_eval(task, data_dir)
            _, legacy = less_eval(task, data_dir)
            evaluations.append(dict(task=task, ready=legacy["n_subtasks"] == legacy["expected_subtasks"],
                references=len(references), full_test_n=len(tests), reference_overlap_ids=overlap,
                reference_overlap_n=len(overlap), qwen_native=native, less=legacy))
        except (ValueError, OSError, KeyError) as exc:
            evaluations.append(dict(task=task, ready=False, error=str(exc)))
    if artifact_root and Path(artifact_root).is_dir():
        for filename in ("metrics.json", "train_results.json", "checkpoint.json", "selection_meta.json"):
            for path in sorted(Path(artifact_root).rglob(filename)):
                try:
                    content = json.loads(path.read_text())
                    artifacts.append(dict(path=str(path.resolve()), kind=filename, sha256=sha256_file(path),
                        completed=content.get("completed"), score=content.get("score"), task=content.get("task"),
                        note="candidate for reuse; matching configuration/protocol still required"))
                except (OSError, ValueError) as exc:
                    artifacts.append(dict(path=str(path), error=str(exc)))
    return dict(created_at=utc_now(), gpu_checked=False, no_less_execution=True, sources=sources,
        evaluations=evaluations, artifacts=artifacts,
        training_data_ready=all(item["ready"] for item in sources),
        reference_test_zero_overlap=all(item.get("reference_overlap_n") == 0 for item in evaluations),
        model_config_exists=(Path(model_path) / "config.json").is_file() if model_path else None,
        conclusion="audit only; no GPU training or measured model scores are produced")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--artifact-root", type=Path, default=Path("out"))
    p.add_argument("--model-path", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--strict", action="store_true", help="Exit 2 after saving audit if data are missing or dev/test overlap")
    args = p.parse_args()
    report = audit(args.data_dir, args.artifact_root, args.model_path)
    write_json(args.output, report)
    print(json.dumps(dict(training_data_ready=report["training_data_ready"],
        reference_test_zero_overlap=report["reference_test_zero_overlap"],
        available_artifacts=len(report["artifacts"]), output=str(args.output)), indent=2))
    if args.strict and (not report["training_data_ready"] or not report["reference_test_zero_overlap"] or
                       not all(item["ready"] for item in report["evaluations"])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
