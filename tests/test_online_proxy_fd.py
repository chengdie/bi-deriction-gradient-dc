import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from less.train.online_proxy_core import OnlineProxyTrainer, ProxyConfig
from less.train.online_proxy_fd import FiniteDifferenceProxyTrainer


class ScalarLM(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.1, 0.05], dtype=dtype))
        self.steps = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        signs = torch.where(input_ids == 1, -1.0, 1.0).to(self.weight.dtype)
        return SimpleNamespace(logits=signs.unsqueeze(-1) * self.weight)


def record(index, conflict=False, invalid=False, length=2):
    return dict(id=f"sample-{index}", pool_index=index,
                input_ids=[int(conflict)] + [2] * (length - 1),
                labels=[-100] + ([-100] if invalid else [0]) + [-100] * (length - 2),
                attention_mask=[1] * length)


def trainer(mode="fd", scheme="forward", dtype=torch.float32, **kwargs):
    model = ScalarLM(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    config = ProxyConfig(filter_mode=mode, scheme=scheme, epsilon=1e-3, entropy_threshold=1e9,
                         max_cache_updates=100, **kwargs)
    cls = FiniteDifferenceProxyTrainer if mode == "fd" else OnlineProxyTrainer
    return cls(model, optimizer, [record(99)], config)


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_fd_direction_and_exact_restore(scheme):
    instance = trainer(scheme=scheme, dtype=torch.float64)
    instance.refresh_reference("test")
    before = [parameter.clone() for parameter in instance.params]
    records = [record(0), record(1, conflict=True)]
    exact, _ = OnlineProxyTrainer._filter(instance, records, 0)
    rows, _ = instance._filter(records, 0)
    assert [row["accepted"] for row in rows] == [True, False]
    for truth, row in zip(exact, rows):
        assert abs(row["dir_deriv"] - truth["dot"]) / abs(truth["dot"]) < 0.01
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(instance.params, before))


def test_restore_on_failed_forward():
    instance = trainer()
    instance.refresh_reference("test")
    before = instance.params[0].clone()
    with pytest.raises(ValueError):
        with instance.perturbed(-1e-2):
            raise ValueError("simulated OOM")
    assert torch.equal(instance.params[0], before)


def test_per_example_loss_batch_and_padding():
    instance = trainer()
    records = [record(0, length=2), record(1, conflict=True, length=4), record(2, length=7), record(3, length=3)]
    together = instance.forward_values(records)
    separate = torch.stack([instance.forward_values([item])[0] for item in records])
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=0)


def test_update_weights_examples_equally_despite_different_answer_lengths():
    instance = trainer("none", fd_batch_size=3)
    records = [record(0, length=2), record(1, length=5), record(2, conflict=True, length=3)]
    records[1]["labels"] = [-100, 0, 1, 2, 1]
    records[2]["labels"] = [-100, 1, 0]
    expected = sum((instance.gradients(instance.forward_values([item])[0], 1) for item in records),
                   instance.zero_vector())
    decisions = [dict(position=index, accepted=True) for index in range(len(records))]
    actual = instance.update_gradient(records, decisions)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_fd_and_exact_same_retained_update():
    exact, fd = trainer("cosine"), trainer("fd")
    records = [record(0), record(1, conflict=True), record(2), record(3, invalid=True)]
    a, b = exact.process_batch(records, 0), fd.process_batch(records, 0)
    assert a["accepted"] == b["accepted"] == 2
    assert b["decisions"][-1]["reason"] == "no_supervised_tokens"
    torch.testing.assert_close(exact.params[0], fd.params[0], atol=1e-6, rtol=0)
    assert a["n_backward_examples"] == 4  # three candidates plus initial reference
    assert b["n_backward_examples"] == 3  # two retained plus initial reference


def test_stale_empty_retries_then_skips_without_optimizer_step():
    instance = trainer()
    instance.refresh_reference("initial")
    instance.cache_age = 1
    before = instance.params[0].clone()
    row = instance.process_batch([record(0, conflict=True)], 0)
    assert row["retried"] and row["skipped"] and len(row["attempts"]) == 2
    assert row["refreshes"][0]["reason"] == "stale_empty"
    assert instance.updates == 0 and not instance.optimizer.state
    assert torch.equal(before, instance.params[0])


def test_refresh_at_configured_effective_update_limit():
    instance = trainer()
    instance.config.max_cache_updates = 1
    row = instance.process_batch([record(0)], 0)
    assert [event["reason"] for event in row["refreshes"]] == ["initial", "max_cache_updates"]
    assert instance.cache_age == 0


def test_matched_random_replays_count_and_seed():
    records = [record(index) for index in range(8)]
    choices = []
    for _ in range(2):
        instance = trainer("random")
        instance.matched_batches = {0: dict(candidates=list(range(8)), accepted=3)}
        row = instance.process_batch(records, 0)
        assert row["accepted"] == 3 and row["n_backward_examples"] == 3
        choices.append([decision["pool_index"] for decision in row["decisions"] if decision["accepted"]])
    assert choices[0] == choices[1]
    instance.matched_batches[1] = dict(candidates=[99], accepted=2)
    with pytest.raises(RuntimeError, match="mismatch"):
        instance.process_batch(records, 1)


def test_checkpoint_restores_optimizer_and_cached_direction():
    original = trainer()
    original.process_batch([record(0)], 0)
    saved_model = copy.deepcopy(original.model.state_dict())
    saved_state = copy.deepcopy(original.state_dict())
    resumed = trainer()
    resumed.model.load_state_dict(saved_model)
    resumed.load_state_dict(saved_state)
    a = original.process_batch([record(1), record(2, conflict=True)], 1)
    b = resumed.process_batch([record(1), record(2, conflict=True)], 1)
    assert a["accepted"] == b["accepted"]
    assert original.updates == resumed.updates
    assert torch.equal(original.params[0], resumed.params[0])


def distributed_worker(rank, world_size, rendezvous, output, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=45))
    try:
        instance = trainer(mode)
        if mode == "random":
            instance.matched_batches = {0: dict(candidates=[0, 1, 2], accepted=2)}
        # Rank 1 has zero retained candidates, and zero references when m=1.
        row = instance.process_batch([record(0), record(1, conflict=True), record(2)], 0)
        torch.save(dict(parameters=instance.params[0].detach(), accepted=row["accepted"],
                        gradient=instance.params[0].grad, decisions=row["decisions"]), Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["cosine", "fd", "none", "random"])
def test_two_ranks_match_single_rank_with_empty_local_selection(tmp_path, mode):
    single = trainer(mode)
    if mode == "random":
        single.matched_batches = {0: dict(candidates=[0, 1, 2], accepted=2)}
    expected = single.process_batch([record(0), record(1, conflict=True), record(2)], 0)
    mp.spawn(distributed_worker, args=(2, str(tmp_path / "store"), str(tmp_path), mode), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert result["accepted"] == expected["accepted"]
        assert result["decisions"] == expected["decisions"]
        torch.testing.assert_close(result["parameters"], single.params[0], atol=1e-6, rtol=0)
        torch.testing.assert_close(result["gradient"], single.params[0].grad, atol=1e-6, rtol=0)


def distributed_failure_worker(rank, rendezvous, output):
    from less.train.online_proxy_core import Collective
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        collective = Collective(torch.device("cpu"))
        try:
            with collective.guard():
                if rank == 1:
                    raise ValueError("simulated worker-local forward failure")
        except RuntimeError as exc:
            (Path(output) / f"error-{rank}.txt").write_text(str(exc))
    finally:
        dist.destroy_process_group()


def test_rank_local_failure_propagates_to_every_worker(tmp_path):
    mp.spawn(distributed_failure_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert "simulated worker-local forward failure" in (tmp_path / f"error-{rank}.txt").read_text()
