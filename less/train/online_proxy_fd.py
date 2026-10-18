"""Finite differences along the cached, normalized reference gradient."""

from contextlib import contextmanager
import torch

from less.train.online_proxy_core import OnlineProxyTrainer, valid_record


class FiniteDifferenceProxyTrainer(OnlineProxyTrainer):
    @contextmanager
    def perturbed(self, distance):
        backups = [parameter.detach().clone() for parameter in self.params]
        norm = torch.linalg.vector_norm(self.reference_gradient.double())
        direction = self.reference_gradient / norm.float()
        try:
            with torch.no_grad():
                offset = 0
                for parameter in self.params:
                    step = direction[offset:offset + parameter.numel()].reshape_as(parameter)
                    parameter.add_(step.to(parameter.dtype), alpha=distance)
                    offset += parameter.numel()
            yield
        finally:
            with torch.no_grad():
                for parameter, backup in zip(self.params, backups):
                    parameter.copy_(backup)
            if self.config.check_restore and any(not torch.equal(p, b) for p, b in zip(self.params, backups)):
                raise RuntimeError("Finite difference perturbation was not restored exactly")

    def score_records(self, records):
        with torch.no_grad():
            base = self.forward_values(records)
            with self.perturbed(-self.config.epsilon):
                minus = self.forward_values(records)
            plus = None
            if self.config.scheme == "central":
                with self.perturbed(self.config.epsilon):
                    plus = self.forward_values(records)
                delta = (minus - plus) / 2
            else:
                delta = minus - base
            return base, minus, plus, delta

    def _filter(self, records, batch_index):
        decisions = [self.decision(records[position], position)
                     for position in range(self.collective.rank, len(records), self.collective.world_size)]
        eligible = [row for row in decisions if valid_record(records[row["position"]])]
        with self.collective.guard():
            for start in range(0, len(eligible), self.config.fd_batch_size):
                rows = eligible[start:start + self.config.fd_batch_size]
                batch = [records[row["position"]] for row in rows]
                base, minus, plus, delta = self.score_records(batch)
                for index, row in enumerate(rows):
                    difference = float(delta[index])
                    keep = difference <= self.config.deadzone
                    row.update(delta=difference, dir_deriv=-difference / self.config.epsilon,
                               loss_base=float(base[index]), loss_perturbed=float(minus[index]),
                               loss_plus=float(plus[index]) if plus is not None else None,
                               accepted=keep, reason="aligned_or_deadzone" if keep else "conflict")
        return decisions, None
