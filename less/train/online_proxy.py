"""Online LoRA training: exact cosine, finite differences, no filter, or matched random.

python -m less.train.online_proxy --help
torchrun --standalone --nproc_per_node=2 -m less.train.online_proxy ...
"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback

from less.train.online_proxy_io import (SCHEMA_VERSION, JsonlPool, RunWriter, code_provenance,
    encode_reference_records, environment_info, fingerprint, inspect_inputs, read_jsonl,
    sha256_file, utc_now, write_json)

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-model-path")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--train-files", nargs="+", type=Path)
    p.add_argument("--task", choices=("mmlu", "bbh", "tydiqa"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--filter", choices=("cosine", "fd", "none", "random"), default="cosine")
    p.add_argument("--fd-epsilon", type=float, default=1e-3)
    p.add_argument("--fd-scheme", choices=("forward", "central"), default="forward")
    p.add_argument("--fd-deadzone", type=float, default=0)
    p.add_argument("--fd-batch-size", type=int, default=2, help="Per-rank scoring/update microbatch; independent of global batch")
    p.add_argument("--token-chunk-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16, help="GLOBAL candidate batch, independent of world size")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--reference-max-length", type=int, default=4096)
    p.add_argument("--entropy-threshold", type=float, default=0.05)
    p.add_argument("--max-cache-updates", type=int, default=20)
    p.add_argument("--no-retry-stale-empty", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--model-seed", type=int, default=0, help="Shared LoRA initialization for every order seed/method")
    p.add_argument("--random-seed", type=int)
    p.add_argument("--random-match-log", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--save-every-batches", type=int, default=500)
    p.add_argument("--resume-from", type=Path, help="Snapshot; writes continuation to a NEW empty output directory")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fd-choice", type=Path, help="Read epsilon/scheme/dtype from a gated calibration choice.json")
    p.add_argument("--dry-run", action="store_true", help="Print configuration without loading torch/models or writing files")
    return p


def validate(args):
    if min(args.batch_size, args.fd_batch_size, args.token_chunk_size, args.max_length,
           args.reference_max_length, args.epochs, args.lora_r, args.lora_alpha) <= 0:
        raise ValueError("Sizes, lengths, epochs and LoRA dimensions must be positive")
    if args.max_batches is not None and args.max_batches <= 0 or args.save_every_batches < 0:
        raise ValueError("Invalid batch limit/save interval")
    if args.filter == "random" and not args.random_match_log:
        raise ValueError("--filter random requires --random-match-log")
    if args.fd_choice:
        choice = json.loads(args.fd_choice.read_text())
        if not choice.get("passed"):
            raise ValueError("Calibration choice has not passed the fidelity gates")
        args.fd_epsilon, args.fd_scheme = choice["epsilon"], choice["scheme"]
        args.dtype = choice.get("dtype", args.dtype)
    if args.random_seed is None:
        args.random_seed = 1000 + args.seed
    if args.train_files is None:
        args.train_files = [args.data_dir / "train" / "processed" / name / f"{name}_data.jsonl"
                            for name in ("flan_v2", "cot", "dolly", "oasst1")]


def load_model(args, device):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from less.model_utils import load_model_and_tokenizer, resolve_lora_targets, resolve_checkpoint, QWEN_PROJECTIONS
    model_path = str(args.resume_from) if args.resume_from else args.model_path
    base_override = args.base_model_path
    if args.resume_from and base_override is None:
        base_override = resolve_checkpoint(args.model_path)[0]
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_path=base_override,
                                               dtype=args.dtype, device=device, trainable=True)
    if not isinstance(model, PeftModel):
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=resolve_lora_targets(model, QWEN_PROJECTIONS)))
    for adapter in model.peft_config.values():
        if adapter.r != args.lora_r or adapter.lora_alpha != args.lora_alpha:
            raise ValueError("Loaded adapter rank/alpha differ from the requested run configuration")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora" not in name.lower():
                raise ValueError(f"Only LoRA parameters may be trainable: {name}")
            parameter.data = parameter.data.float()
    model.eval()
    # Explicitly disable runtime caches on the text configuration as well.
    model.config.use_cache = False
    if hasattr(model.config, "get_text_config"):
        model.config.get_text_config().use_cache = False
    return model, tokenizer


def serializable_args(args):
    return {key: str(value.resolve()) if isinstance(value, Path) else
            [str(item.resolve()) if isinstance(item, Path) else item for item in value] if isinstance(value, list) else value
            for key, value in vars(args).items()}


def training_identity(args, inputs, references, template_hash):
    fields = ("filter", "task", "batch_size", "max_length", "reference_max_length", "learning_rate",
              "clip_norm", "lora_r", "lora_alpha", "seed", "model_seed", "dtype", "fd_epsilon", "fd_scheme",
              "fd_deadzone", "entropy_threshold", "max_cache_updates", "no_retry_stale_empty", "random_seed")
    from less.model_utils import resolve_checkpoint
    base, adapter = resolve_checkpoint(args.model_path, args.base_model_path)
    base = Path(base).resolve()
    model = dict(path=str(base), config_sha256=sha256_file(base / "config.json") if (base / "config.json").is_file() else None,
        weights=[dict(name=path.name, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                 for path in sorted(base.glob("*.safetensors"))],
        initial_adapter_sha256={path.name: sha256_file(path) for path in sorted(Path(adapter).glob("adapter_model.*"))}
                               if adapter else None)
    return dict(config={key: getattr(args, key) for key in fields}, model=model,
                random_match_sha256=sha256_file(args.random_match_log) if args.filter == "random" and args.random_match_log else None,
                sources=[dict(rows=item["rows"], sha256=item["sha256"]) for item in inputs],
                reference_sha256=fingerprint(references), chat_template_sha256=template_hash)


def save_checkpoint(args, trainer, tokenizer, directory, next_batch, identity, collective, writer):
    import torch
    with collective.guard():
        if collective.rank == 0:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            trainer.model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            state = dict(trainer=trainer.state_dict(), next_batch=next_batch, identity=identity,
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(trainer.device) if trainer.device.type == "cuda" else None,
                         created_at=utc_now())
            temporary = directory / "online_state.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(directory / "online_state.pt")
            write_json(directory / "checkpoint.json", dict(completed=True, next_batch=next_batch,
                       update_index=trainer.updates, state_sha256=sha256_file(directory / "online_state.pt"),
                       run_dir=str(args.output_dir.resolve()), identity_sha256=fingerprint(identity)))
            writer.append("checkpoints", dict(path=str(directory.resolve()), next_batch=next_batch,
                                              update_index=trainer.updates, created_at=utc_now()))


def execute(args):
    import torch
    import torch.distributed as dist
    from less.train.online_proxy_core import Collective, OnlineProxyTrainer, ProxyConfig
    from less.train.online_proxy_fd import FiniteDifferenceProxyTrainer

    world_size, local_rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 and args.device.startswith("cuda") else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --dry-run for planning or --device cpu for diagnostics")
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    collective = Collective(device)
    writer = None
    started = time.perf_counter()
    try:
        with collective.guard():
            if collective.rank == 0:
                writer = RunWriter(args.output_dir)
                write_json(args.output_dir / "status.json", dict(status="initializing", completed=False, started_at=utc_now()))
                write_json(args.output_dir / "run.json", dict(schema_version=SCHEMA_VERSION,
                    protocol_version=SCHEMA_VERSION, config=serializable_args(args), created_at=utc_now(),
                    code=code_provenance(ROOT), no_less_execution=True, initialization_completed=False))
                logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout),
                    logging.FileHandler(args.output_dir / "console.log", encoding="utf-8")], force=True)
            inputs = inspect_inputs(args.train_files)
            if any(item["rows"] <= 10 for item in inputs):
                raise ValueError("Training input contains placeholder-sized files; prepare the complete LESS sources first")
        input_signatures = collective.gather(fingerprint(inputs))
        if len(set(input_signatures)) != 1:
            raise ValueError("Workers see different training inputs")
        torch.manual_seed(args.model_seed)
        random.seed(args.model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.model_seed)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
        with collective.guard():
            model, tokenizer = load_model(args, device)
            references = encode_reference_records(args.task, args.data_dir, tokenizer, args.reference_max_length)
            pool = JsonlPool(args.train_files)
        config = ProxyConfig(filter_mode=args.filter, batch_size=args.batch_size, fd_batch_size=args.fd_batch_size,
            token_chunk_size=args.token_chunk_size, epsilon=args.fd_epsilon, scheme=args.fd_scheme,
            deadzone=args.fd_deadzone, entropy_threshold=args.entropy_threshold, max_cache_updates=args.max_cache_updates,
            retry_stale_empty=not args.no_retry_stale_empty, clip_norm=args.clip_norm, random_seed=args.random_seed)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                      lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        matched = {}
        template_hash = fingerprint(tokenizer.get_chat_template())
        identity = training_identity(args, inputs, references, template_hash)
        signatures = collective.gather(fingerprint(identity))
        if len(set(signatures)) != 1:
            raise ValueError("Workers see different reference data, model identity or training settings")
        if args.random_match_log:
            with collective.guard():
                matched = {row["batch_index"]: row for row in read_jsonl(args.random_match_log)}
                manifest = json.loads((args.random_match_log.parent / "run.json").read_text())
                status = json.loads((args.random_match_log.parent / "status.json").read_text())
                if status.get("status") != "completed":
                    raise ValueError("Random replay requires a completed FD run")
                match_config = manifest["config"]
                for key in ("seed", "model_seed", "batch_size", "task", "epochs", "max_length", "reference_max_length"):
                    if match_config[key] != getattr(args, key):
                        raise ValueError(f"Random replay configuration mismatch: {key}")
                if [item["sha256"] for item in manifest["inputs"]] != [item["sha256"] for item in inputs]:
                    raise ValueError("Random replay uses a different candidate pool")
                if manifest["identity"].get("model") != identity["model"]:
                    raise ValueError("Random replay uses a different base model or initialization adapter")
        cls = FiniteDifferenceProxyTrainer if args.filter == "fd" else OnlineProxyTrainer
        trainer = cls(model, optimizer, references, config, tokenizer.pad_token_id, matched)
        next_batch = 0
        if args.resume_from:
            with collective.guard():
                state = torch.load(args.resume_from / "online_state.pt", map_location="cpu", weights_only=True)
                if state["identity"] != identity:
                    raise ValueError("Resume input/configuration does not match the saved trajectory")
                trainer.load_state_dict(state["trainer"])
                next_batch = state["next_batch"]
                torch.set_rng_state(state["torch_rng"])
                if device.type == "cuda" and state.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state(state["cuda_rng"], device)
        pool_manifests = []
        for item in inputs:
            path = Path(item["path"]).parent / "pool_manifest.json"
            if path.is_file():
                saved_pool = json.loads(path.read_text())
                if saved_pool.get("pool_sha256") != item["sha256"]:
                    raise ValueError("Pool JSONL does not match its source mapping manifest")
                pool_manifests.append(dict(path=str(path), sha256=sha256_file(path), sources=saved_pool["sources"]))
        manifest = dict(schema_version=SCHEMA_VERSION, protocol_version=SCHEMA_VERSION, initialization_completed=True,
            created_at=utc_now(), config=serializable_args(args), trainer_config=asdict(config), inputs=inputs,
            pool_size=len(pool), pool_manifests=pool_manifests, reference_count=len(references),
            reference_ids=[record["id"] for record in references], identity=identity,
            identity_sha256=fingerprint(identity), code=code_provenance(ROOT),
            environment=environment_info(torch, device), world_size=collective.world_size,
            initial_batch=next_batch, no_less_execution=True,
            reference_policy="cached-gradient+per-update-entropy" if trainer.uses_reference else "not-needed",
            loss_reduction="mean of per-example supervised-token means", training_mode="eval-with-autograd",
            comparison_note="Qwen results require matching evaluation manifests; published LESS numbers are external references")
        with collective.guard():
            if writer:
                write_json(args.output_dir / "run.json", manifest)
                write_json(args.output_dir / "status.json", dict(status="running", completed=False, started_at=manifest["created_at"]))
        from less.data_selection.get_training_dataset import encode_with_qwen_chat_format
        batches_per_epoch = (len(pool) + args.batch_size - 1) // args.batch_size
        total_batches = batches_per_epoch * args.epochs
        stop = min(total_batches, args.max_batches) if args.max_batches else total_batches
        if next_batch >= stop:
            raise ValueError("Resume cursor is at/after requested batch limit")
        segment_accepted = segment_candidates = segment_skipped = 0
        order, previous_epoch = None, None
        for batch_index in range(next_batch, stop):
            epoch, local_batch = divmod(batch_index, batches_per_epoch)
            if epoch != previous_epoch:
                order = list(range(len(pool)))
                random.Random(args.seed + epoch).shuffle(order)
                previous_epoch = epoch
            indices = order[local_batch * args.batch_size:(local_batch + 1) * args.batch_size]
            with collective.guard():
                records = []
                for index in indices:
                    raw, identifiers = pool[index]
                    records.append(dict(encode_with_qwen_chat_format(raw, tokenizer, args.max_length), **identifiers))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row = trainer.process_batch(records, batch_index, epoch)
            with collective.guard():
                if writer:
                    writer.batch(row)
                    logging.info("batch=%d accepted=%d/%d updates=%d seconds=%.3f", batch_index,
                                 row["accepted"], len(records), trainer.updates, row["timing"]["batch"])
            segment_accepted += row["accepted"]
            segment_candidates += len(records)
            segment_skipped += int(row["skipped"])
            if args.save_every_batches and (batch_index + 1) % args.save_every_batches == 0:
                save_checkpoint(args, trainer, tokenizer, args.output_dir / f"snapshot-batch-{batch_index + 1}",
                                batch_index + 1, identity, collective, writer)
        save_checkpoint(args, trainer, tokenizer, args.output_dir / "model", stop, identity, collective, writer)
        duration = time.perf_counter() - started
        results = dict(completed=True, scope="this execution segment", initial_batch=next_batch, next_batch=stop,
            full_pool_pass_completed=stop == total_batches, batches=stop - next_batch, updates=trainer.updates,
            candidates=segment_candidates, accepted=segment_accepted, skipped_batches=segment_skipped,
            retention_rate=segment_accepted / segment_candidates, wall_seconds=duration,
            gpu_hours=duration * collective.world_size / 3600 if device.type == "cuda" else 0.0,
            refresh_count=trainer.refresh_count, model=str((args.output_dir / "model").resolve()), finished_at=utc_now())
        with collective.guard():
            if writer:
                write_json(args.output_dir / "train_results.json", results)
                write_json(args.output_dir / "status.json", dict(status="completed", **results))
    except BaseException as exc:
        if writer:
            write_json(args.output_dir / "status.json", dict(status="failed", completed=False,
                       error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), failed_at=utc_now()))
        raise
    finally:
        if writer:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    args = parser().parse_args()
    validate(args)
    if args.dry_run:
        print(json.dumps(dict(config=serializable_args(args), schema_version=SCHEMA_VERSION,
                             global_batch_size=args.batch_size, no_less_execution=True), ensure_ascii=False, indent=2))
        return
    execute(args)


if __name__ == "__main__":
    main()
