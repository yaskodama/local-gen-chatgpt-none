from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from genome import Genome


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, genome: Genome):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, genome.embed_dim)
        self.lstm = nn.LSTM(
            input_size=genome.embed_dim,
            hidden_size=genome.hidden_dim,
            num_layers=genome.num_layers,
            batch_first=True,
            dropout=genome.dropout if genome.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(genome.hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.output(out)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size: int, genome: Genome):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, genome.embed_dim)
        self.position = PositionalEncoding(genome.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=genome.embed_dim,
            nhead=genome.num_heads,
            dim_feedforward=genome.hidden_dim,
            dropout=genome.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=genome.num_layers)
        self.output = nn.Linear(genome.embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.position(x)
        out = self.encoder(x, mask=mask)
        return self.output(out)


def build_model(vocab_size: int, genome: Genome) -> nn.Module:
    if genome.model_type == "lstm":
        return CharLSTM(vocab_size, genome)
    if genome.model_type == "transformer":
        return CharTransformer(vocab_size, genome)
    raise ValueError(f"Unknown model_type: {genome.model_type}")


@torch.no_grad()
def sample_text(
    model: nn.Module,
    start_text: str,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    length: int = 120,
    temperature: float = 0.8,
    device: str = "cpu",
) -> str:
    model.eval()
    if not start_text:
        start_text = random_start(char_to_idx)
    text = start_text
    for _ in range(length):
        encoded = [char_to_idx.get(ch, 0) for ch in text[-128:]]
        x = torch.tensor([encoded], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        text += idx_to_char[next_idx]
    return text


def random_start(char_to_idx: dict[str, int]) -> str:
    return "\n" if "\n" in char_to_idx else next(iter(char_to_idx))
