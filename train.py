from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from genome import Genome
from model import build_model


class CharDataset(Dataset):
    def __init__(self, text: str, char_to_idx: dict[str, int], seq_length: int):
        self.encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.encoded) - self.seq_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded[idx : idx + self.seq_length + 1]
        return chunk[:-1], chunk[1:]


@dataclass
class TrainResult:
    model: nn.Module
    loss: float
    train_loss: float
    valid_loss: float | None
    valid_ppl: float | None
    valid_bits_per_char: float | None
    valid_bits_per_byte: float | None
    steps: int
    seconds: float


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(set(text)) < 2:
        raise ValueError("Training text must contain at least two unique characters.")
    return text


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def train_short(
    genome: Genome,
    text: str,
    char_to_idx: dict[str, int],
    max_steps: int = 40,
    device: str = "cpu",
    valid_text: str | None = None,
) -> TrainResult:
    dataset = CharDataset(text, char_to_idx, genome.seq_length)
    if len(dataset) == 0:
        raise ValueError("Training text is too short for the selected sequence length.")

    loader = DataLoader(dataset, batch_size=genome.batch_size, shuffle=True, drop_last=False)
    model = build_model(len(char_to_idx), genome).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=genome.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses: list[float] = []
    steps = 0
    started = time.time()
    while steps < max_steps:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            steps += 1
            if steps >= max_steps:
                break
    train_loss = sum(losses[-10:]) / max(1, len(losses[-10:]))
    valid_loss, valid_bits_per_byte = (
        evaluate_loss(model, valid_text, char_to_idx, genome, device) if valid_text else (None, None)
    )
    selected_loss = valid_loss if valid_loss is not None else train_loss
    valid_ppl = math.exp(valid_loss) if valid_loss is not None else None
    valid_bits_per_char = valid_loss / math.log(2) if valid_loss is not None else None
    return TrainResult(
        model=model,
        loss=selected_loss,
        train_loss=train_loss,
        valid_loss=valid_loss,
        valid_ppl=valid_ppl,
        valid_bits_per_char=valid_bits_per_char,
        valid_bits_per_byte=valid_bits_per_byte,
        steps=steps,
        seconds=time.time() - started,
    )


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    text: str,
    char_to_idx: dict[str, int],
    genome: Genome,
    device: str = "cpu",
) -> tuple[float, float]:
    dataset = CharDataset(text, char_to_idx, genome.seq_length)
    if len(dataset) == 0:
        raise ValueError("Validation text is too short for the selected sequence length.")

    loader = DataLoader(dataset, batch_size=genome.batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    byte_lengths = torch.zeros(len(char_to_idx), dtype=torch.float)
    for ch, idx in char_to_idx.items():
        byte_lengths[idx] = len(ch.encode("utf-8"))
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss_value = float(loss.item())
        token_count = y.numel()
        total_nll += loss_value * token_count
        total_tokens += token_count
        total_bytes += float(byte_lengths[y.cpu()].sum().item())
    mean_loss = total_nll / max(1, total_tokens)
    bits_per_byte = (total_nll / math.log(2)) / max(1.0, total_bytes)
    return mean_loss, bits_per_byte
