from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rogii.models.emission_features import EmissionInput, EmissionSequence
from rogii.models.sequence_tcn import ResidualBlock


class CandidateEmissionTCN(nn.Module):
    """Shared temporal scorer applied identically to every candidate offset."""

    def __init__(self, n_costs: int, n_row_features: int, offsets: np.ndarray, config: dict[str, Any]):
        super().__init__()
        channels = int(config.get("channels", 32))
        blocks = int(config.get("blocks", 4))
        kernel = int(config.get("kernel_size", 5))
        dropout = float(config.get("dropout", 0.10))
        self.register_buffer("offsets", torch.as_tensor(offsets / max(np.max(np.abs(offsets)), 1.0)))
        self.input = nn.Conv1d(n_costs + n_row_features + 1, channels, 1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel, 2**index, dropout) for index in range(blocks)]
        )
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, costs: torch.Tensor, row_features: torch.Tensor) -> torch.Tensor:
        # costs: B,T,C,S. Treat each state as an item so weights are shared across offsets.
        batch, steps, _, states = costs.shape
        cost_x = costs.permute(0, 3, 2, 1)
        row_x = row_features.permute(0, 2, 1)[:, None].expand(-1, states, -1, -1)
        offset_x = self.offsets[None, :, None, None].expand(batch, -1, 1, steps)
        values = torch.cat([cost_x, row_x, offset_x], dim=2).reshape(batch * states, -1, steps)
        logits = self.head(self.blocks(self.input(values))).reshape(batch, states, steps)
        return logits.permute(0, 2, 1)


class _SequenceDataset(Dataset):
    def __init__(self, sequences: list[EmissionSequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> EmissionSequence:
        return self.sequences[index]


def collate_emissions(batch: list[EmissionSequence]):
    steps = max(len(item.target_state) for item in batch)
    costs = np.zeros((len(batch), steps, batch[0].costs.shape[1], batch[0].costs.shape[2]), np.float32)
    rows = np.zeros((len(batch), steps, batch[0].row_features.shape[1]), np.float32)
    target = np.full((len(batch), steps), -100, np.int64)
    valid = np.zeros((len(batch), steps), bool)
    for index, item in enumerate(batch):
        length = len(item.target_state)
        costs[index, :length] = item.costs
        rows[index, :length] = item.row_features
        target[index, :length] = item.target_state
        valid[index, :length] = item.valid
    target[~valid] = -100
    return tuple(torch.from_numpy(value) for value in (costs, rows, target, valid))


def _loss(
    logits: torch.Tensor,
    costs: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=-100,
        label_smoothing=float(config.get("label_smoothing", 0.02)),
    )
    valid = target >= 0
    if not valid.any() or float(config.get("hard_negative_weight", 0.20)) <= 0:
        return ce
    raw = costs[:, :, 0, :].clone()
    raw.scatter_(-1, target.clamp_min(0).unsqueeze(-1), float("inf"))
    hard = raw.argmin(dim=-1)
    true_logit = logits.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    hard_logit = logits.gather(-1, hard.unsqueeze(-1)).squeeze(-1)
    margin = nn.functional.relu(float(config.get("hard_negative_margin", 0.5)) - true_logit + hard_logit)
    return ce + float(config.get("hard_negative_weight", 0.20)) * margin[valid].mean()


def predict_emissions(
    model: CandidateEmissionTCN,
    sequences: list[EmissionInput],
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for item in sequences:
            costs = torch.from_numpy(item.costs.astype(np.float32))[None].to(device)
            rows = torch.from_numpy(item.row_features)[None].to(device)
            output.append(model(costs, rows).squeeze(0).float().cpu().numpy())
    return output


def train_emission_fold(
    train: list[EmissionSequence],
    valid: list[EmissionSequence],
    offsets: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[CandidateEmissionTCN, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _SequenceDataset(train),
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_emissions,
        generator=generator,
    )
    model = CandidateEmissionTCN(train[0].costs.shape[1], train[0].row_features.shape[1], offsets, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 8e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = None
    best_nll = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(config.get("epochs", 8))):
        model.train()
        running, batches = 0.0, 0
        for costs, rows, target, _ in loader:
            costs, rows, target = costs.to(device), rows.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = _loss(model(costs, rows), costs, target, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            batches += 1
        predictions = predict_emissions(model, valid, device)
        nll_values = []
        top10_values = []
        for item, logits in zip(valid, predictions, strict=True):
            mask = item.valid
            if not mask.any():
                continue
            shifted = logits[mask] - logits[mask].max(axis=1, keepdims=True)
            logsumexp = np.log(np.exp(shifted).sum(axis=1))
            nll_values.extend(logsumexp - shifted[np.arange(mask.sum()), item.target_state[mask]])
            order = np.argsort(-logits[mask], axis=1)
            top10_values.extend(np.any(order[:, :10] == item.target_state[mask, None], axis=1))
        valid_nll = float(np.mean(nll_values))
        history.append(
            {"epoch": float(epoch + 1), "train_loss": running / max(batches, 1),
             "valid_nll": valid_nll, "valid_top10": float(np.mean(top10_values))}
        )
        if np.isfinite(valid_nll) and valid_nll < best_nll - 1e-5:
            best_nll = valid_nll
            best_state = deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(config.get("patience", 2)):
                break
    if best_state is None:
        raise RuntimeError("Stage 12B training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return model, history
