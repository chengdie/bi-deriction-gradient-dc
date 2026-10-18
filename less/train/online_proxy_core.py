"""Exact online conflict screening and shared, explicitly synchronized updates.

Each worker owns a full model. We reduce gradient *sums* and accepted counts,
never rank means; ranks with no retained samples still enter every collective.
No DDP forward/backward hooks are used, because local batch sizes can differ.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math
import random
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class ProxyConfig:
    filter_mode: str = "cosine"
    batch_size: int = 16
    fd_batch_size: int = 2
    token_chunk_size: int = 64
    epsilon: float = 1e-3
    scheme: str = "forward"
    deadzone: float = 0.0
    entropy_threshold: float = 0.05
    max_cache_updates: int = 20
    retry_stale_empty: bool = True
    clip_norm: float = 1.0
    random_seed: int = 1003
    check_restore: bool = True

    def __post_init__(self):
        if self.filter_mode not in {"cosine", "fd", "none", "random"}:
            raise ValueError("Unknown filter mode")
        if self.scheme not in {"forward", "central"}:
            raise ValueError("Unknown finite difference scheme")
        if min(self.batch_size, self.fd_batch_size, self.token_chunk_size, self.max_cache_updates) <= 0:
            raise ValueError("Batch, token chunk and cache sizes must be positive")
        if not all(math.isfinite(x) for x in (self.epsilon, self.deadzone, self.entropy_threshold, self.clip_norm)):
            raise ValueError("Configuration must be finite")
        if self.epsilon <= 0 or self.deadzone < 0 or self.entropy_threshold < 0 or self.clip_norm <= 0:
            raise ValueError("Invalid perturbation, threshold or clipping value")


class Collective:
    def __init__(self, device):
        self.device = device
        self.active = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.active else 0
        self.world_size = dist.get_world_size() if self.active else 1

    def sum(self, value):
        if self.active:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def maximum(self, value):
        if self.active:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value

    def gather(self, value):
        if not self.active:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    @contextmanager
    def guard(self):
        """Propagate a rank-local error before any subsequent collective."""
        error = None
        try:
            yield
        except Exception:
            error = f"rank {self.rank}:\n{traceback.format_exc()}"
        errors = [item for item in self.gather(error) if item]
        if errors:
            raise RuntimeError("; ".join(errors))


def valid_record(record):
    return any(label != -100 for label in record["labels"][1:])


class OnlineProxyTrainer:
    def __init__(self, model, optimizer, reference_records, config=None,
                 pad_token_id=0, matched_batches=None):
        self.model, self.optimizer = model, optimizer
        self.config = config or ProxyConfig()
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("No trainable parameters")
        self.device = self.params[0].device
        self.collective = Collective(self.device)
        self.references = reference_records
        self.pad_token_id = pad_token_id
        self.matched_batches = matched_batches or {}
        self.model.eval()
        self.dimension = sum(p.numel() for p in self.params)
        self.reference_gradient = None
        self.reference_entropy = None
        self.cache_age = 0
        self.updates = 0
        self.refresh_count = 0
        self.refresh_events = []
        self.phase = "filter"
        self.counts = {key: [0, 0] for key in ("filter", "update", "entropy", "refresh")}
        if self.collective.active:
            layouts = self.collective.gather([(tuple(parameter.shape), str(parameter.dtype)) for parameter in self.params])
            if any(layout != layouts[0] for layout in layouts):
                raise ValueError("Ranks have different trainable parameter layouts")
            for parameter in self.params:
                dist.broadcast(parameter.data, src=0)

    @property
    def uses_reference(self):
        return self.config.filter_mode in {"cosine", "fd"}

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def timed(self, timing, phase):
        previous = self.phase
        self.phase = phase
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            timing[phase] = timing.get(phase, 0.0) + time.perf_counter() - started
            self.phase = previous

    def zero_vector(self):
        return torch.zeros(self.dimension, device=self.device, dtype=torch.float32)

    def flatten(self, gradients):
        return torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32)).reshape(-1)
                          for p, g in zip(self.params, gradients)])

    def collate(self, records):
        width = max(len(record["input_ids"]) for record in records)
        batch = {}
        for key, padding in (("input_ids", self.pad_token_id), ("labels", -100), ("attention_mask", 0)):
            values = [list(record.get(key, [1] * len(record["input_ids"]))) for record in records]
            batch[key] = torch.tensor([value + [padding] * (width - len(value)) for value in values],
                                      dtype=torch.long, device=self.device)
        return batch

    def forward_values(self, records, entropy=False):
        """Per-example means; CE is chunked over supervised positions only."""
        batch = self.collate(records)
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        logits = outputs.logits
        self.counts[self.phase][0] += len(records)
        values = []
        for index in range(len(records)):
            labels = batch["labels"][index, 1:]
            positions = (labels != -100).nonzero().flatten()
            if not positions.numel():
                raise ValueError("Loss requested for a sample without supervised tokens")
            value = logits.new_zeros((), dtype=torch.float32)
            for chunk in positions.split(self.config.token_chunk_size):
                scores = logits[index, chunk, :].float()
                if entropy:
                    logp = scores.log_softmax(-1)
                    value = value - (logp.exp() * logp).sum() / math.log(scores.shape[-1])
                else:
                    value = value + F.cross_entropy(scores, labels[chunk], reduction="sum")
            value = value / positions.numel()
            if not torch.isfinite(value):
                raise ValueError(f"Non-finite {'entropy' if entropy else 'loss'}: {records[index].get('id')}")
            values.append(value)
        return torch.stack(values)

    def gradients(self, loss, count):
        self.counts[self.phase][1] += count
        result = self.flatten(torch.autograd.grad(loss, self.params, allow_unused=True))
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite gradient")
        return result

    def entropies(self):
        local = []
        with self.collective.guard(), torch.no_grad():
            for position in range(self.collective.rank, len(self.references), self.collective.world_size):
                record = self.references[position]
                if not valid_record(record):
                    raise ValueError(f"Invalid reference: {record.get('id')}")
                local.append((position, float(self.forward_values([record], entropy=True)[0])))
        merged = sorted(item for group in self.collective.gather(local) for item in group)
        if [position for position, _ in merged] != list(range(len(self.references))):
            raise ValueError("Reference entropy coverage mismatch")
        return torch.tensor([value for _, value in merged], dtype=torch.float64, device=self.device)

    def refresh_reference(self, reason, batch_index=-1, current_entropy=None):
        vector, local_loss, count = self.zero_vector(), 0.0, 0
        with self.collective.guard():
            if not self.references:
                raise ValueError("Reference set is empty")
            for position in range(self.collective.rank, len(self.references), self.collective.world_size):
                record = self.references[position]
                if not valid_record(record):
                    raise ValueError(f"Invalid reference: {record.get('id')}")
                loss = self.forward_values([record])[0]
                vector.add_(self.gradients(loss, 1))
                local_loss += float(loss.detach())
                count += 1
        totals = torch.tensor([local_loss, count], dtype=torch.float64, device=self.device)
        self.collective.sum(vector)
        self.collective.sum(totals)
        if int(totals[1]) != len(self.references):
            raise ValueError("Reference gradient coverage mismatch")
        vector /= int(totals[1])
        norm = float(torch.linalg.vector_norm(vector.double()))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Reference gradient is zero or non-finite; experiment stopped")
        self.reference_gradient = vector
        self.reference_entropy = current_entropy.clone() if current_entropy is not None else self.entropies()
        self.cache_age = 0
        self.refresh_count += 1
        self.refresh_events.append(dict(refresh_index=self.refresh_count, reason=reason, batch_index=batch_index,
                                        update_index=self.updates, reference_loss=float(totals[0] / totals[1]),
                                        reference_gradient_norm=norm, mean_entropy=float(self.reference_entropy.mean())))

    def decision(self, record, position):
        return dict(id=str(record.get("id", record.get("pool_index", position))),
                    pool_index=record.get("pool_index", position), position=position,
                    source=record.get("source"), source_row=record.get("source_row"),
                    accepted=False, reason="no_supervised_tokens", cos=None, dot=None,
                    gradient_norm=None, delta=None, dir_deriv=None, loss_base=None,
                    loss_perturbed=None, loss_plus=None)

    def _filter(self, records, batch_index):
        decisions, gradient_sum = [], self.zero_vector()
        direction = self.reference_gradient / torch.linalg.vector_norm(self.reference_gradient.double()).float()
        with self.collective.guard():
            for position in range(self.collective.rank, len(records), self.collective.world_size):
                record = records[position]
                row = self.decision(record, position)
                if valid_record(record):
                    loss = self.forward_values([record])[0]
                    gradient = self.gradients(loss, 1)
                    norm = float(torch.linalg.vector_norm(gradient.double()))
                    dot = float(torch.dot(gradient.double(), direction.double()))
                    cosine = max(-1.0, min(1.0, dot / norm)) if norm else None
                    row.update(loss_base=float(loss.detach()), gradient_norm=norm, dot=dot, cos=cosine,
                               accepted=bool(norm and dot >= 0), reason="zero_gradient" if not norm else
                               ("aligned" if dot >= 0 else "conflict"))
                    if row["accepted"]:
                        gradient_sum.add_(gradient)
                decisions.append(row)
        return decisions, gradient_sum

    def _unfiltered(self, records, batch_index):
        eligible = [position for position, record in enumerate(records) if valid_record(record)]
        keep = set(eligible)
        with self.collective.guard():
            if self.config.filter_mode == "random":
                matched = self.matched_batches.get(batch_index)
                if matched is None or matched["candidates"] != [record["pool_index"] for record in records]:
                    raise ValueError(f"Random replay candidate/order mismatch at batch {batch_index}")
                count = matched["accepted"]
                if not 0 <= count <= len(eligible):
                    raise ValueError("Random replay requested an impossible retained count")
                keep = set(random.Random(self.config.random_seed + batch_index).sample(eligible, count))
            decisions = []
            for position in range(self.collective.rank, len(records), self.collective.world_size):
                row = self.decision(records[position], position)
                if position in eligible:
                    row.update(accepted=position in keep,
                               reason="unfiltered" if self.config.filter_mode == "none" else
                               ("random_keep" if position in keep else "random_drop"))
                decisions.append(row)
        return decisions, None

    def update_gradient(self, records, decisions):
        retained = [records[row["position"]] for row in decisions if row["accepted"]]
        vector = self.zero_vector()
        with self.collective.guard():
            for start in range(0, len(retained), self.config.fd_batch_size):
                batch = retained[start:start + self.config.fd_batch_size]
                vector.add_(self.gradients(self.forward_values(batch).sum(), len(batch)))
        return vector

    def process_batch(self, records, batch_index, epoch=0):
        self.model.eval()
        self.counts = {key: [0, 0] for key in self.counts}
        timing, attempts = {}, []
        event_start = len(self.refresh_events)
        self.synchronize()
        batch_started = time.perf_counter()
        if self.uses_reference and self.reference_gradient is None:
            with self.timed(timing, "refresh"):
                self.refresh_reference("initial", batch_index)
        for attempt in range(2):
            with self.timed(timing, "filter"):
                local, vector = self._filter(records, batch_index) if self.uses_reference else self._unfiltered(records, batch_index)
                merged = sorted((row for group in self.collective.gather(local) for row in group), key=lambda row: row["position"])
                if [row["position"] for row in merged] != list(range(len(records))):
                    raise ValueError("Candidate decision coverage mismatch")
            accepted = sum(row["accepted"] for row in merged)
            attempts.append(dict(attempt=attempt, cache_age=self.cache_age, accepted=accepted, decisions=merged))
            if (accepted == 0 and self.uses_reference and self.cache_age > 0
                    and self.config.retry_stale_empty and attempt == 0):
                with self.timed(timing, "refresh"):
                    self.refresh_reference("stale_empty", batch_index)
                continue
            break
        displacement = projection = before_clip = None
        drift = None
        if accepted:
            with self.timed(timing, "update"):
                if vector is None:
                    vector = self.update_gradient(records, local)
                # Every rank enters these reductions, including empty local selections.
                self.collective.sum(vector)
                count = torch.tensor(sum(row["accepted"] for row in local), device=self.device, dtype=torch.long)
                self.collective.sum(count)
                if int(count) != accepted:
                    raise ValueError("Global accepted-count mismatch")
                vector /= accepted
                before = [parameter.detach().clone() for parameter in self.params]
                self.optimizer.zero_grad(set_to_none=True)
                offset = 0
                for parameter in self.params:
                    parameter.grad = vector[offset:offset + parameter.numel()].reshape_as(parameter).to(parameter.dtype).clone()
                    offset += parameter.numel()
                before_clip = float(torch.nn.utils.clip_grad_norm_(self.params, self.config.clip_norm, error_if_nonfinite=True))
                with self.collective.guard():
                    self.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in self.params):
                        raise ValueError("Non-finite parameters after optimizer update")
                change = torch.cat([(p.detach().double() - previous.double()).flatten()
                                    for p, previous in zip(self.params, before)])
                displacement = float(torch.linalg.vector_norm(change))
                projection = float(torch.dot(change, self.reference_gradient.double())) if self.uses_reference else None
                self.updates += 1
                self.cache_age += 1
            if self.uses_reference:
                with self.timed(timing, "entropy"):
                    entropy = self.entropies()
                    drift = float((entropy - self.reference_entropy).abs().mean() /
                                  self.reference_entropy.abs().mean().clamp_min(1e-3))
                reason = "entropy_drift" if drift > self.config.entropy_threshold else (
                    "max_cache_updates" if self.cache_age >= self.config.max_cache_updates else None)
                if reason:
                    with self.timed(timing, "refresh"):
                        self.refresh_reference(reason, batch_index, entropy)
        self.synchronize()
        timing["batch"] = time.perf_counter() - batch_started
        keys = ("filter", "update", "entropy", "refresh", "batch")
        times = torch.tensor([timing.get(key, 0.0) for key in keys], dtype=torch.float64, device=self.device)
        self.collective.maximum(times)
        counters = torch.tensor([value for pair in self.counts.values() for value in pair], dtype=torch.long, device=self.device)
        self.collective.sum(counters)
        count_dict = {key: dict(forward=int(counters[2 * i]), backward=int(counters[2 * i + 1]))
                      for i, key in enumerate(self.counts)}
        memory = torch.tensor(torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0,
                              device=self.device, dtype=torch.float64)
        self.collective.maximum(memory)
        return dict(batch_index=batch_index, epoch=epoch, update_index=self.updates,
                    filter_mode=self.config.filter_mode, candidates=[record["pool_index"] for record in records],
                    candidate_ids=[str(record["id"]) for record in records], n_candidates=len(records),
                    accepted=accepted, rejected=len(records) - accepted, skipped=not bool(accepted),
                    decisions=merged, attempts=attempts, retried=len(attempts) > 1,
                    timing={key: float(value) for key, value in zip(keys, times)}, counters=count_dict,
                    n_forward_examples=sum(value["forward"] for value in count_dict.values()),
                    n_backward_examples=sum(value["backward"] for value in count_dict.values()),
                    peak_mem_gb=float(memory), update_displacement_norm=displacement,
                    reference_dot_update=projection, gradient_norm_before_clip=before_clip,
                    entropy_drift=drift, cache_age=self.cache_age, refresh_count=self.refresh_count,
                    refreshes=self.refresh_events[event_start:])

    def state_dict(self):
        return dict(config=asdict(self.config), optimizer=self.optimizer.state_dict(), updates=self.updates,
                    cache_age=self.cache_age, refresh_count=self.refresh_count,
                    reference_gradient=self.reference_gradient.cpu() if self.reference_gradient is not None else None,
                    reference_entropy=self.reference_entropy.cpu() if self.reference_entropy is not None else None)

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates, self.cache_age, self.refresh_count = state["updates"], state["cache_age"], state["refresh_count"]
        for name in ("reference_gradient", "reference_entropy"):
            value = state[name]
            setattr(self, name, value.to(self.device) if value is not None else None)
