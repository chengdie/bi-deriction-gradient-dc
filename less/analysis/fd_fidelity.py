"""Compare FD with exact gradients at the SAME parameters and cached direction.

This tool never trains. Logged future batches supply candidate identities only;
their historical cosine scores are not reused as fixed-snapshot ground truth.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from less.train.online_proxy_io import (JsonlPool, fingerprint, inspect_inputs, read_jsonl,
                                       sha256_file, utc_now, write_json)


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result, start = [0.0] * len(values), 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            result[index] = (start + stop - 1) / 2
        start = stop
    return result


def correlation(left, right):
    if len(left) < 2:
        return None
    a, b = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - a) * (y - b) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - a) ** 2 for x in left) * sum((y - b) ** 2 for y in right))
    return numerator / denominator if denominator else None


def fidelity_metrics(rows):
    all_scored = [row for row in rows if row.get("delta") is not None]
    zero_gradient_n = sum(row.get("gradient_norm") == 0 for row in all_scored)
    rows = [row for row in rows if row.get("gradient_norm", 0) and row.get("delta") is not None]
    if not rows:
        raise ValueError("No valid, nonzero-gradient candidates for fidelity analysis")
    sign = lambda value: (value > 0) - (value < 0)
    agrees = [sign(row["delta"]) == -sign(row["cos"]) for row in rows]
    decisions = [(row["delta"] <= 0) == (row["cos"] >= 0) for row in rows]
    strong = [row for row in rows if abs(row["cos"]) >= 0.05]
    strong_agree = statistics.mean(sign(row["delta"]) == -sign(row["cos"]) for row in strong) if strong else None
    relative = [abs(row["dir_deriv"] - row["dot"]) / abs(row["dot"]) for row in rows if row["dot"]]
    labels = [row["cos"] >= 0 for row in rows]
    predictions = [row["delta"] <= 0 for row in rows]
    positives, predicted = sum(labels), sum(predictions)
    expected = (positives * predicted + (len(rows) - positives) * (len(rows) - predicted)) / len(rows) ** 2
    agreement = statistics.mean(decisions)
    scores = [-row["delta"] for row in rows]
    ranked = ranks(scores)
    auc = ((sum(rank + 1 for rank, label in zip(ranked, labels) if label) - positives * (positives + 1) / 2) /
           (positives * (len(rows) - positives))) if 0 < positives < len(rows) else None
    buckets = []
    for lower, upper in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 1.0000001)):
        subset = [row for row in rows if lower <= abs(row["cos"]) < upper]
        buckets.append(dict(lower=lower, upper=min(upper, 1.0), n=len(subset),
            decision_agree=statistics.mean((row["delta"] <= 0) == (row["cos"] >= 0) for row in subset) if subset else None,
            sign_agree=statistics.mean(sign(row["delta"]) == -sign(row["cos"]) for row in subset) if subset else None))
    exact_set = {index for index, keep in enumerate(labels) if keep}
    fd_set = {index for index, keep in enumerate(predictions) if keep}
    return dict(n=len(rows), n_total_scored=len(all_scored), zero_gradient_n=zero_gradient_n,
        decision_agree_including_zero=statistics.mean((row["delta"] <= 0) == bool(row.get("gradient_norm") and row["dot"] >= 0)
                                                     for row in all_scored),
        sign_agree=statistics.mean(agrees), decision_agree=agreement,
        strong_n=len(strong), strong_sign_agree=strong_agree,
        zero_delta_rate=statistics.mean(row["delta"] == 0 for row in rows),
        spearman=correlation(ranks([row["dot"] for row in rows]), ranks([row["dir_deriv"] for row in rows])),
        median_relative_error=statistics.median(relative) if relative else None,
        cohen_kappa=(agreement - expected) / (1 - expected) if expected < 1 else None,
        roc_auc=auc, retention_difference=(predicted - positives) / len(rows),
        jaccard=len(exact_set & fd_set) / len(exact_set | fd_set) if exact_set | fd_set else 1.0, buckets=buckets)


def choose_configuration(reports):
    if not reports:
        raise ValueError("Provide at least one fidelity report")
    points = [report["point_sha256"] for report in reports]
    if len(set(points)) != len(points):
        raise ValueError("Duplicate calibration parameter points")
    if len({report.get("task") for report in reports}) != 1:
        raise ValueError("Calibrate each task separately; do not silently mix task-specific grids")
    input_signatures = {fingerprint([(source["rows"], source["sha256"]) for source in report.get("inputs", [])])
                        for report in reports}
    if len(input_signatures) != 1:
        raise ValueError("Calibration points use different candidate pools")
    common = None
    for report in reports:
        current = {(row["epsilon"], row["scheme"], report["dtype"]) for row in report["grid"]}
        common = current if common is None else common & current
    scores = []
    for epsilon, scheme, dtype in common:
        entries = [next(row for row in report["grid"] if row["epsilon"] == epsilon and row["scheme"] == scheme)
                   for report in reports]
        strong_total = sum(row["strong_n"] for row in entries)
        strong = (sum((row["strong_sign_agree"] or 0) * row["strong_n"] for row in entries) / strong_total
                  if strong_total else None)
        scores.append(dict(epsilon=epsilon, scheme=scheme, dtype=dtype,
            sign_agree=statistics.mean(row["sign_agree"] for row in entries), strong_sign_agree=strong,
            strong_n=strong_total))
    if not scores:
        raise ValueError("Reports have no common epsilon/scheme/dtype grid")
    best_score = max(row["sign_agree"] for row in scores)
    tied = [row for row in scores if best_score - row["sign_agree"] < 0.005]
    chosen = max(tied, key=lambda row: (row["epsilon"], row["sign_agree"], row["scheme"] == "forward"))
    if chosen["scheme"] == "central":
        forward = [row for row in scores if row["scheme"] == "forward" and
                   chosen["sign_agree"] - row["sign_agree"] < 0.01]
        if forward:
            top = max(row["sign_agree"] for row in forward)
            chosen = max((row for row in forward if top - row["sign_agree"] < 0.005), key=lambda row: row["epsilon"])
    # Gate the ACTUALLY chosen config, not a better central config discarded for cost.
    return dict(chosen, passed=(chosen["sign_agree"] >= 0.90 and chosen["strong_sign_agree"] is not None
                               and chosen["strong_sign_agree"] >= 0.97 and len(points) >= 3),
                calibration_points=len(points), point_sha256=points, grid=scores, created_at=utc_now())


def run(args):
    import torch
    from less.train.online_proxy import load_model
    from less.train.online_proxy_core import OnlineProxyTrainer, ProxyConfig, valid_record
    from less.train.online_proxy_fd import FiniteDifferenceProxyTrainer
    from less.train.online_proxy_io import encode_reference_records
    from less.data_selection.get_training_dataset import encode_with_qwen_chat_format
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.model_seed)
    args.resume_from = None
    args.gradient_checkpointing = False
    device = torch.device(args.device)
    model, tokenizer = load_model(args, device)
    references = encode_reference_records(args.task, args.data_dir, tokenizer, args.reference_max_length)
    pool = JsonlPool(args.train_files)
    if args.batches_log:
        batches = [row for row in read_jsonl(args.batches_log) if row["batch_index"] >= args.after_batch][:args.num_batches]
        if len(batches) != args.num_batches:
            raise ValueError("Not enough subsequent candidate batches")
        indices = [index for row in batches for index in row["candidates"]]
        batch_for_index = {index: row["batch_index"] for row in batches for index in row["candidates"]}
    elif args.candidate_indices:
        indices = json.loads(args.candidate_indices.read_text())
    else:
        indices = random.Random(args.seed).sample(range(len(pool)), min(args.sample_size, len(pool)))
    if not args.batches_log:
        batch_for_index = {}
    if len(set(indices)) != len(indices) or any(not isinstance(index, int) or not 0 <= index < len(pool) for index in indices):
        raise ValueError("Candidate indices must be unique, valid pool indices")
    records = []
    for index in indices:
        raw, identity = pool[index]
        records.append(dict(encode_with_qwen_chat_format(raw, tokenizer, args.max_length), **identity))
    config = ProxyConfig(filter_mode="fd", fd_batch_size=args.fd_batch_size)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5)
    trainer = FiniteDifferenceProxyTrainer(model, optimizer, references, config, tokenizer.pad_token_id)
    state_hash = None
    if args.reference_mode == "cached":
        path = Path(args.model_path) / "online_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"]["reference_sha256"] != fingerprint(references):
            raise ValueError("Snapshot cached reference does not match this task/tokenization")
        if state["trainer"]["reference_gradient"] is None:
            raise ValueError("Snapshot has no cached reference gradient")
        trainer.reference_gradient = state["trainer"]["reference_gradient"].to(device)
        if not torch.isfinite(trainer.reference_gradient).all() or not trainer.reference_gradient.norm() > 0:
            raise ValueError("Snapshot has an invalid cached reference gradient")
        state_hash = sha256_file(path)
    else:
        trainer.refresh_reference("fidelity_fresh")
    # Distinct directories/optimizer states do not prove distinct parameters.
    point_digest = hashlib.sha256()
    for parameter in trainer.params:
        point_digest.update(str(tuple(parameter.shape)).encode())
        point_digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    point_hash = point_digest.hexdigest()
    exact, _ = OnlineProxyTrainer._filter(trainer, records, -1)
    # A repeat of an identical microbatch measures nondeterminism, not quantization error.
    noise = 0.0
    with torch.no_grad():
        valid = [record for record in records if valid_record(record)]
        for start in range(0, len(valid), args.fd_batch_size):
            batch = valid[start:start + args.fd_batch_size]
            first, second = trainer.forward_values(batch), trainer.forward_values(batch)
            noise = max(noise, float((first - second).abs().max()))
    rows, grid = [], []
    for scheme in args.scheme:
        for epsilon in args.epsilon:
            config.scheme, config.epsilon = scheme, epsilon
            decisions, _ = trainer._filter(records, -1)
            entries = []
            for truth, decision in zip(exact, decisions):
                row = dict(decision, cos=truth["cos"], dot=truth["dot"], gradient_norm=truth["gradient_norm"],
                           epsilon=epsilon, scheme=scheme, point_sha256=point_hash,
                           source_batch_index=batch_for_index.get(decision["pool_index"]))
                entries.append(row)
                rows.append(row)
            per_batch = []
            for batch_index in sorted(set(batch_for_index.values())):
                subset = [row for row in entries if row["source_batch_index"] == batch_index]
                if any(row.get("gradient_norm") for row in subset):
                    per_batch.append(dict(batch_index=batch_index, **fidelity_metrics(subset)))
            grid.append(dict(epsilon=epsilon, scheme=scheme, per_batch=per_batch, **fidelity_metrics(entries)))
    columns = ("point_sha256", "pool_index", "id", "epsilon", "scheme", "cos", "dot", "gradient_norm",
               "delta", "dir_deriv", "loss_base", "loss_perturbed", "loss_plus", "accepted", "reason", "source_batch_index")
    with (output / "fidelity.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = Path(args.model_path) / "checkpoint.json"
    report = dict(completed=True, point_sha256=point_hash, task=args.task, dtype=args.dtype,
        snapshot_batch=json.loads(checkpoint.read_text())["next_batch"] if checkpoint.is_file() else 0,
        snapshot_state_sha256=state_hash,
        model_path=args.model_path, reference_mode=args.reference_mode,
        inputs=inspect_inputs(args.train_files), candidate_indices=indices, noise_floor_max=noise, grid=grid,
        comparison="both methods recomputed at identical fixed parameters and reference direction")
    write_json(output / "fidelity.json", report)
    print(json.dumps(dict(point_sha256=point_hash, noise_floor_max=noise, grid=grid), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--select-from", nargs="+", type=Path, help="Three or more fidelity.json files; writes gated choice.json")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-path")
    p.add_argument("--base-model-path")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--train-files", nargs="+", type=Path)
    p.add_argument("--task", choices=("mmlu", "bbh", "tydiqa"), default="mmlu")
    p.add_argument("--candidate-indices", type=Path)
    p.add_argument("--batches-log", type=Path)
    p.add_argument("--after-batch", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=32)
    p.add_argument("--sample-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--epsilon", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    p.add_argument("--scheme", nargs="+", choices=("forward", "central"), default=["forward", "central"])
    p.add_argument("--reference-mode", choices=("fresh", "cached"), default="fresh")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--reference-max-length", type=int, default=4096)
    p.add_argument("--fd-batch-size", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.select_from:
        choice = choose_configuration([json.loads(path.read_text()) for path in args.select_from])
        if (args.output_dir / "choice.json").exists():
            raise FileExistsError("Refusing to overwrite a calibration choice")
        write_json(args.output_dir / "choice.json", choice)
        print(json.dumps(choice, indent=2))
        return
    if not args.model_path or not args.train_files:
        p.error("Provide --model-path and --train-files (or --select-from)")
    if min(args.epsilon) <= 0 or min(args.sample_size, args.fd_batch_size, args.num_batches) <= 0:
        p.error("Epsilon and sample/microbatch sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
