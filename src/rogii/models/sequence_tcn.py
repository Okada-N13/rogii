from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rogii.models.sequence_features import SequenceWell, feature_standardizer


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + 0.5 * self.network(values)


class ResidualTCN(nn.Module):
    def __init__(self, n_features: int, config: dict[str, Any]):
        super().__init__()
        channels = int(config.get("channels", 32))
        blocks = int(config.get("blocks", 4))
        kernel_size = int(config.get("kernel_size", 5))
        dropout = float(config.get("dropout", 0.15))
        self.input = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(channels, kernel_size, 2**index, dropout)
                for index in range(blocks)
            ]
        )
        self.head = nn.Conv1d(channels, 1, 1)
        self.output_cap = float(config.get("output_cap", 40.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.blocks(self.input(values))).squeeze(1)
        return self.output_cap * torch.tanh(raw / self.output_cap)


class _WellDataset(Dataset):
    def __init__(self, wells: list[SequenceWell], mean: np.ndarray, scale: np.ndarray, stride: int):
        self.wells = wells
        self.mean = mean
        self.scale = scale
        self.stride = int(stride)

    def __len__(self) -> int:
        return len(self.wells)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        well = self.wells[index]
        positions = np.arange(0, len(well.features), self.stride, dtype=int)
        if positions[-1] != len(well.features) - 1:
            positions = np.append(positions, len(well.features) - 1)
        x = (well.features[positions] - self.mean) / self.scale
        y = well.residual_target[positions]
        return x.astype(np.float32), y.astype(np.float32)


def _collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(item[0]) for item in batch)
    features = batch[0][0].shape[1]
    x = np.zeros((len(batch), features, length), dtype=np.float32)
    y = np.zeros((len(batch), length), dtype=np.float32)
    mask = np.zeros((len(batch), length), dtype=bool)
    for index, (values, target) in enumerate(batch):
        n = len(values)
        x[index, :, :n] = values.T
        y[index, :n] = target
        mask[index, :n] = True
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def predict_tcn(
    model: ResidualTCN,
    wells: list[SequenceWell],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for well in wells:
            values = (well.features - mean) / scale
            tensor = torch.from_numpy(values.T[None].astype(np.float32)).to(device)
            prediction = model(tensor).cpu().numpy().reshape(-1)
            predictions.append(prediction.astype(np.float32))
    return predictions


def train_tcn_fold(
    train_wells: list[SequenceWell],
    valid_wells: list[SequenceWell],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[ResidualTCN, np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    mean, scale = feature_standardizer(train_wells)
    dataset = _WellDataset(train_wells, mean, scale, int(config.get("training_stride", 4)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
    )
    model = ResidualTCN(train_wells[0].features.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(int(config.get("epochs", 8))):
        model.train()
        total_squared = 0.0
        total_rows = 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(x)
                loss = torch.square(prediction[mask] - y[mask]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            total_squared += float(loss.detach().cpu()) * int(mask.sum())
            total_rows += int(mask.sum())
        validation = predict_tcn(model, valid_wells, mean, scale, device)
        valid_error = np.concatenate(
            [prediction - well.residual_target for prediction, well in zip(validation, valid_wells, strict=True)]
        )
        valid_loss = float(np.mean(np.square(valid_error)))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_residual_rmse": float(np.sqrt(total_squared / max(total_rows, 1))),
                "valid_residual_rmse": float(np.sqrt(valid_loss)),
            }
        )
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.get("patience", 2)):
                break
    if best_state is None:
        raise RuntimeError("TCN training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return model, mean, scale, history
