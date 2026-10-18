"""Rebuild analysis tables/figures exclusively from saved run artifacts.

No published LESS scores are synthesized or treated as locally measured runs.
Incomplete runs are listed in the audit and excluded from effect estimates.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from less.train.online_proxy_io import fingerprint, read_jsonl, write_json


def csv_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["no_completed_results"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def paired_interval(differences, repetitions=10000, seed=2026):
    if len(differences) < 2:
        return None, None
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)]
    return quantile(means, 0.025), quantile(means, 0.975)


def holm(values):
    """Input {comparison: p}; return monotone Holm-adjusted values."""
    ordered, result, previous = sorted(values.items(), key=lambda item: item[1]), {}, 0.0
    for index, (key, value) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * value))
        result[key] = previous
    return result


def mcnemar_exact(left_only, right_only):
    total = left_only + right_only
    if not total:
        return 1.0
    # Recurrence in log space avoids enormous binomial integers for full MMLU.
    tail = min(left_only, right_only)
    logs = [math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
            - total * math.log(2) for k in range(tail + 1)]
    peak = max(logs)
    return min(1.0, 2 * math.exp(peak) * sum(math.exp(value - peak) for value in logs))


def question_comparison(left_dir, right_dir, task, repetitions=10000, aggregation="macro"):
    paths = [Path(directory) / "eval" / "predictions.jsonl" for directory in (left_dir, right_dir)]
    if not all(path.is_file() for path in paths):
        return None
    left, right = [{row["id"]: row for row in read_jsonl(path)} for path in paths]
    if set(left) != set(right):
        raise ValueError("Paired predictions do not contain identical evaluation IDs")
    metric = {"mmlu": "accuracy", "bbh": "exact_match", "tydiqa": "f1"}[task]
    groups, left_only, right_only = {}, 0, 0
    for rid in left:
        a, b = left[rid], right[rid]
        for key in ("gold", "subtask", "fewshot_ids", "prompt_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Paired evaluation input mismatch: {key}, {rid}")
        av, bv = a["scores"][metric], b["scores"][metric]
        groups.setdefault(a["subtask"], []).append(av - bv)
        left_only += int(av == 100 and bv == 0)
        right_only += int(av == 0 and bv == 100)
    # Resample within each subtask and preserve the actual reported aggregation.
    rng = random.Random(2026)
    def aggregate(samples):
        return (sum(sum(values) for values in samples) / sum(len(values) for values in samples)
                if aggregation == "micro" else statistics.mean(statistics.mean(values) for values in samples))
    boot = [aggregate([rng.choices(values, k=len(values)) for values in groups.values()]) for _ in range(repetitions)]
    return dict(n=len(left), difference=aggregate(list(groups.values())),
                ci_low=quantile(boot, 0.025), ci_high=quantile(boot, 0.975),
                p_mcnemar=mcnemar_exact(left_only, right_only) if task in {"mmlu", "bbh"} else None,
                resampling=f"paired questions within subtask, then {aggregation} mean",
                mcnemar_note="binary micro-level discordance; primary-score CI is reported separately")


def comparable_key(manifest):
    config = manifest["config"]
    keys = ("task", "seed", "model_seed", "batch_size", "max_length", "reference_max_length",
            "learning_rate", "clip_norm", "lora_r", "lora_alpha", "dtype", "epochs", "max_batches",
            "entropy_threshold", "max_cache_updates", "no_retry_stale_empty")
    return fingerprint(dict(config={key: config.get(key) for key in keys},
        sources=[(item["rows"], item["sha256"]) for item in manifest["inputs"]],
        reference=manifest["identity"]["reference_sha256"],
        model=manifest["identity"].get("model"),
        template=manifest["identity"]["chat_template_sha256"]))


def summarize(root, output, repetitions=10000, warmup_batches=20, plots=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents and output.name in {"model", "eval"}:
        raise ValueError("Analysis output must be a dedicated report directory")
    runs, batch_rows, refresh_rows, audit, records, decision_sources = [], [], [], [], [], []
    for manifest_path in sorted(root.rglob("run.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != "online-proxy-fd-v1":
            continue
        directory = manifest_path.parent
        if directory == output or output in directory.parents:
            continue
        results_path = directory / "train_results.json"
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not results_path.exists() or status.get("status") != "completed":
            audit.append(dict(run_dir=str(directory), status=status.get("status", "missing-results"), included=False))
            continue
        results = json.loads(results_path.read_text())
        if not results.get("completed"):
            audit.append(dict(run_dir=str(directory), status="incomplete", included=False))
            continue
        batches = list(read_jsonl(directory / "batches.jsonl"))
        if len(batches) != results["batches"] or len({row["batch_index"] for row in batches}) != len(batches):
            raise ValueError(f"Batch log coverage mismatch: {directory}")
        candidates, accepted = sum(row["n_candidates"] for row in batches), sum(row["accepted"] for row in batches)
        if candidates != results["candidates"] or accepted != results["accepted"]:
            raise ValueError(f"Raw log/result totals disagree: {directory}")
        method, task = manifest["config"]["filter"], manifest["config"]["task"]
        identifier = str(directory.relative_to(root))
        measured = batches[warmup_batches:]
        run = dict(run=identifier, method=method, task=task, seed=manifest["config"]["seed"],
            model_seed=manifest["config"]["model_seed"], candidates=candidates, accepted=accepted,
            retention_rate=accepted / candidates if candidates else None, updates=results["updates"],
            skipped_batches=sum(row["skipped"] for row in batches), refreshes=results["refresh_count"],
            wall_seconds=results["wall_seconds"], gpu_hours=results["gpu_hours"],
            peak_mem_gb=max((row["peak_mem_gb"] for row in batches), default=None),
            measured_batches=len(measured), warmup_batches_excluded=warmup_batches,
            world_size=manifest["world_size"], gpu=manifest["environment"].get("gpu"),
            resumed=bool(manifest["config"].get("resume_from")),
            full_pool_pass_completed=results["full_pool_pass_completed"])
        run["variant"] = fingerprint({key: manifest["config"].get(key) for key in
            ("fd_epsilon", "fd_scheme", "fd_deadzone", "entropy_threshold", "max_cache_updates", "no_retry_stale_empty")})[:12]
        for phase in ("filter", "update", "entropy", "refresh", "batch"):
            run[f"median_{phase}_seconds"] = statistics.median(row["timing"][phase] for row in measured) if measured else None
        metrics_path = directory / "eval" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        if metrics and metrics.get("completed"):
            run.update(score=metrics["score"], metric=metrics["metric"], eval_n=metrics["n"],
                       clean_score=metrics.get("clean_score"), clean_eval_n=metrics.get("clean_n"),
                       eval_reference_overlap_n=metrics.get("reference_overlap_n", 0),
                       eval_protocol_sha256=fingerprint(metrics["protocol"]))
        else:
            run.update(score=None, metric=None, eval_n=None, eval_protocol_sha256=None)
        runs.append(run)
        records.append((manifest, directory, run, metrics))
        audit.append(dict(run_dir=str(directory), status="completed", included=True, evaluated=run["score"] is not None))
        batch_rows.extend(dict(run=identifier, **{key: value for key, value in row.items() if key not in {"decisions", "attempts"}})
                          for row in batches)
        refresh_rows.extend(dict(run=identifier, **row) for row in read_jsonl(directory / "refreshes.jsonl"))
        decision_sources.append((identifier, directory / "decisions.jsonl"))
    main_groups = {}
    for run in runs:
        if run["score"] is not None and run["full_pool_pass_completed"] and not run["resumed"]:
            # Include protocol and data/config identity so incompatible runs cannot silently mix.
            manifest = next(item[0] for item in records if item[2] is run)
            configuration = dict(manifest["identity"])
            configuration.pop("random_match_sha256", None)
            configuration["config"] = {key: value for key, value in configuration["config"].items()
                                       if key not in {"seed", "random_seed"}}
            key = (run["task"], run["method"], run["eval_protocol_sha256"], fingerprint(configuration))
            main_groups.setdefault(key, []).append(run)
    main = []
    for (task, method, protocol, configuration), group in main_groups.items():
        seeds = [run["seed"] for run in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds in the same result group; choose one run per seed")
        scores = [run["score"] for run in group]
        main.append(dict(task=task, method=method, n_seeds=len(scores), seeds=seeds,
            mean=statistics.mean(scores), std=statistics.stdev(scores) if len(scores) > 1 else None,
            mean_gpu_hours=statistics.mean(run["gpu_hours"] for run in group),
            eval_protocol_sha256=protocol, configuration_sha256=configuration))
    paired, questions, speedups = [], [], []
    lookup = {}
    for item in records:
        if item[2]["method"] in {"fd", "random"}:
            continue
        key = (comparable_key(item[0]), item[2]["method"])
        if key in lookup:
            raise ValueError("Duplicate comparable runs; use a narrower --root")
        lookup[key] = item
    for manifest, directory, run, metrics in records:
        if run["method"] != "fd":
            continue
        for baseline in ("cosine", "random", "none"):
            if baseline == "random":
                matches = [item for item in records if item[2]["method"] == "random" and
                    comparable_key(item[0]) == comparable_key(manifest) and
                    item[0]["config"].get("random_match_log") and
                    Path(item[0]["config"]["random_match_log"]).resolve() == (directory / "batches.jsonl").resolve()]
                if len(matches) > 1:
                    raise ValueError("Multiple random-matched runs for one FD trajectory")
                other = matches[0] if matches else None
            else:
                other = lookup.get((comparable_key(manifest), baseline))
            if not other:
                continue
            _, other_dir, other_run, other_metrics = other
            if baseline == "cosine" and not run["resumed"] and not other_run["resumed"] and run["candidates"] == other_run["candidates"]:
                same_hardware = run["world_size"] == other_run["world_size"] and run["gpu"] == other_run["gpu"]
                speedups.append(dict(task=run["task"], seed=run["seed"], same_hardware=same_hardware,
                    variant=run["variant"],
                    wall_speedup=other_run["wall_seconds"] / run["wall_seconds"] if same_hardware else None,
                    fd_run=run["run"], cosine_run=other_run["run"]))
            if not metrics or not other_metrics or not metrics.get("completed") or not other_metrics.get("completed"):
                continue
            if metrics["protocol"] != other_metrics["protocol"] or metrics["n"] != other_metrics["n"]:
                raise ValueError("Paired evaluation protocols differ")
            if not run["full_pool_pass_completed"] or not other_run["full_pool_pass_completed"] or run["resumed"] or other_run["resumed"]:
                continue
            paired.append(dict(task=run["task"], seed=run["seed"], comparison=f"fd-{baseline}",
                               variant=run["variant"],
                               difference=run["score"] - other_run["score"]))
            aggregation = "micro" if "micro" in metrics["protocol"].get("aggregation", "") else "macro"
            question = question_comparison(directory, other_dir, run["task"], repetitions, aggregation)
            if question:
                questions.append(dict(task=run["task"], seed=run["seed"], variant=run["variant"], comparison=f"fd-{baseline}", **question))
    paired_summary = []
    for task, variant in sorted({(row["task"], row["variant"]) for row in paired}):
        for comparison in ("fd-cosine", "fd-random", "fd-none"):
            group = [row for row in paired if row["task"] == task and row["variant"] == variant and row["comparison"] == comparison]
            if not group:
                continue
            values = [row["difference"] for row in group]
            low, high = paired_interval(values, repetitions)
            paired_summary.append(dict(task=task, variant=variant, comparison=comparison, n_seeds=len(values),
                mean_difference=statistics.mean(values), ci_low=low, ci_high=high,
                noninferior=low > -1.0 if comparison == "fd-cosine" and low is not None and len(values) >= 3 else None,
                note="seed bootstrap; three seeds have limited power; no cross-task IID pooling"))
    for task, seed, variant in {(row["task"], row["seed"], row["variant"]) for row in questions}:
        group = [row for row in questions if row["task"] == task and row["seed"] == seed and
                 row["variant"] == variant and row["p_mcnemar"] is not None]
        adjusted = holm({row["comparison"]: row["p_mcnemar"] for row in group})
        for row in group:
            row["p_mcnemar_holm"] = adjusted[row["comparison"]]
    tables = dict(runs=runs, main_results=main, batches=batch_rows,
                  refreshes=refresh_rows, paired_differences=paired, paired_summary=paired_summary,
                  question_comparisons=questions, speedups=speedups, audit=audit)
    for name, rows in tables.items():
        csv_table(output / "tables" / f"{name}.csv", rows)
    # Stream per-sample decisions: full multi-seed runs can contain millions of rows.
    decision_count = 0
    fields = ("run", "schema_version", "batch_index", "epoch", "update_index", "attempt", "final_attempt",
              "id", "pool_index", "position", "source", "source_row", "accepted", "reason", "cos", "dot",
              "gradient_norm", "delta", "dir_deriv", "loss_base", "loss_perturbed", "loss_plus")
    with (output / "tables" / "decisions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for identifier, path in decision_sources:
            for row in read_jsonl(path):
                writer.writerow(dict(run=identifier, **row))
                decision_count += 1
    fidelity_grid, fidelity_buckets, fidelity_batches, fidelity_sources = [], [], [], []
    for path in sorted(root.rglob("fidelity.json")):
        if output in path.parents:
            continue
        report = json.loads(path.read_text())
        if not report.get("completed"):
            continue
        fidelity_sources.append(path.parent / "fidelity.csv")
        for row in report["grid"]:
            identifiers = dict(point=report["point_sha256"], task=report["task"], dtype=report["dtype"],
                               model_path=report["model_path"], snapshot_batch=report.get("snapshot_batch", 0),
                               epsilon=row["epsilon"], scheme=row["scheme"])
            fidelity_grid.append(dict(identifiers, **{key: value for key, value in row.items()
                                 if key not in {"buckets", "per_batch", "epsilon", "scheme"}}))
            fidelity_buckets.extend(dict(identifiers, **bucket) for bucket in row["buckets"])
            fidelity_batches.extend(dict(identifiers, **{key: value for key, value in batch.items() if key != "buckets"})
                                    for batch in row.get("per_batch", []))
    csv_table(output / "tables" / "fidelity_grid.csv", fidelity_grid)
    csv_table(output / "tables" / "fidelity_buckets.csv", fidelity_buckets)
    csv_table(output / "tables" / "fidelity_batches.csv", fidelity_batches)
    tables.update(fidelity_grid=fidelity_grid, fidelity_buckets=fidelity_buckets, fidelity_batches=fidelity_batches,
                  decision_count=decision_count)
    write_json(output / "summary.json", dict(tables={key: value for key, value in tables.items() if key not in {"batches", "refreshes"}},
                raw_table_rows=dict(batches=len(batch_rows), refreshes=len(refresh_rows), decisions=decision_count),
                complete_runs=len(runs), audit=audit))
    if plots and (runs or fidelity_grid):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (output / "figures").mkdir(exist_ok=True)
        for field, source, xfield, filename, ylabel in (
            ("accepted", batch_rows, "batch_index", "retention.png", "Retained fraction"),
            ("reference_loss", refresh_rows, "update_index", "reference_loss.png", "Reference loss")):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for run in runs:
                rows = [row for row in source if row["run"] == run["run"]]
                values = [row[field] / row["n_candidates"] if field == "accepted" else row[field] for row in rows]
                ax.plot([row[xfield] for row in rows], values, label=f"{run['method']} {run['task']} s{run['seed']}", alpha=0.8)
            ax.set(xlabel=xfield, ylabel=ylabel)
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(output / "figures" / filename, dpi=160)
            plt.close(fig)
        if fidelity_grid:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for task, scheme in sorted({(row["task"], row["scheme"]) for row in fidelity_grid}):
                rows = [row for row in fidelity_grid if row["task"] == task and row["scheme"] == scheme]
                epsilons = sorted({row["epsilon"] for row in rows})
                ax.plot(epsilons, [statistics.mean(row["sign_agree"] for row in rows if row["epsilon"] == epsilon)
                                   for epsilon in epsilons], marker="o", label=f"{task} {scheme}")
            ax.set(xscale="log", xlabel="epsilon (LoRA L2 distance)", ylabel="Mean sign agreement")
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(output / "figures" / "epsilon_fidelity.png", dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for task, scheme, epsilon in sorted({(row["task"], row["scheme"], row["epsilon"]) for row in fidelity_grid}):
                rows = sorted([row for row in fidelity_grid if (row["task"], row["scheme"], row["epsilon"]) ==
                               (task, scheme, epsilon)], key=lambda row: row["snapshot_batch"])
                ax.plot([row["snapshot_batch"] for row in rows], [row["decision_agree"] for row in rows],
                        marker="o", label=f"{task} {scheme} eps={epsilon:g}")
            ax.set(xlabel="Snapshot batch", ylabel="Decision agreement")
            ax.legend(fontsize="x-small", ncol=2)
            fig.tight_layout()
            fig.savefig(output / "figures" / "fidelity_over_training.png", dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for path in fidelity_sources:
                with path.open(newline="") as stream:
                    rows = [row for row in csv.DictReader(stream) if row.get("cos") and row.get("dir_deriv")]
                if rows:
                    first = (rows[0]["epsilon"], rows[0]["scheme"])
                    rows = [row for row in rows if (row["epsilon"], row["scheme"]) == first][:4096]
                    ax.scatter([float(row["cos"]) for row in rows], [float(row["dir_deriv"]) for row in rows],
                               s=5, alpha=0.35, label=str(path.parent.relative_to(root)))
            ax.set(xlabel="Exact gradient cosine", ylabel="FD directional derivative", yscale="symlog")
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.axvline(0, color="gray", linewidth=0.6)
            ax.legend(fontsize="x-small")
            fig.tight_layout()
            fig.savefig(output / "figures" / "cos_vs_direction.png", dpi=160)
            plt.close(fig)
    (output / "README.md").write_text(
        "# FD analysis\n\nAll numbers are recomputed from run.json, batches.jsonl and evaluation outputs.\n\n"
        f"Completed execution segments: {len(runs)}. Excluded/incomplete runs are listed in tables/audit.csv.\n\n"
        "Runs without evaluations remain in cost tables but do not enter effect estimates. "
        "Smoke/partial-pool runs and resume segments are excluded from main outcome estimates. "
        "Resume segments must be joined/audited as a trajectory before reporting a full-run score or cost.\n\n"
        "No LESS experiments are launched. Published scores require source and model/protocol annotations. "
        "Question intervals preserve subtask macro averaging; seed intervals with three seeds are exploratory.\n",
        encoding="utf-8")
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--warmup-batches", type=int, default=20)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if args.bootstrap <= 0 or args.warmup_batches < 0:
        p.error("Invalid bootstrap/warmup count")
    tables = summarize(args.root, args.output, args.bootstrap, args.warmup_batches, not args.no_plots)
    print(json.dumps(dict(complete_runs=len(tables["runs"]), evaluated_groups=len(tables["main_results"]))))


if __name__ == "__main__":
    main()
