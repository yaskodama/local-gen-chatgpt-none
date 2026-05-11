from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any


MODEL_TYPES = ("lstm", "transformer")
EMBED_DIMS = (16, 24, 32, 48, 64)
HIDDEN_DIMS = (32, 48, 64, 96, 128)
NUM_LAYERS = (1, 2, 3)
NUM_HEADS = (1, 2, 4)
DROPOUTS = (0.0, 0.1, 0.2)
LEARNING_RATES = (1e-3, 2e-3, 3e-3, 5e-3)
BATCH_SIZES = (8, 16)
SEQ_LENGTHS = (24, 32, 48, 64)


@dataclass(frozen=True)
class Genome:
    model_type: str
    embed_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    learning_rate: float
    batch_size: int
    seq_length: int

    @staticmethod
    def random() -> "Genome":
        embed_dim = random.choice(EMBED_DIMS)
        return Genome(
            model_type=random.choice(MODEL_TYPES),
            embed_dim=embed_dim,
            hidden_dim=random.choice(HIDDEN_DIMS),
            num_layers=random.choice(NUM_LAYERS),
            num_heads=random.choice([h for h in NUM_HEADS if embed_dim % h == 0]),
            dropout=random.choice(DROPOUTS),
            learning_rate=random.choice(LEARNING_RATES),
            batch_size=random.choice(BATCH_SIZES),
            seq_length=random.choice(SEQ_LENGTHS),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Genome":
        return Genome(
            model_type=str(data["model_type"]),
            embed_dim=int(data["embed_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            num_layers=int(data["num_layers"]),
            num_heads=int(data["num_heads"]),
            dropout=float(data["dropout"]),
            learning_rate=float(data["learning_rate"]),
            batch_size=int(data["batch_size"]),
            seq_length=int(data["seq_length"]),
        )


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    values: dict[str, Any] = {}
    for key in parent_a.to_dict():
        values[key] = random.choice((getattr(parent_a, key), getattr(parent_b, key)))
    if values["embed_dim"] % values["num_heads"] != 0:
        values["num_heads"] = random.choice([h for h in NUM_HEADS if values["embed_dim"] % h == 0])
    return Genome.from_dict(values)


def mutate(genome: Genome, mutation_rate: float = 0.2) -> Genome:
    values = genome.to_dict()
    choices: dict[str, tuple[Any, ...]] = {
        "model_type": MODEL_TYPES,
        "embed_dim": EMBED_DIMS,
        "hidden_dim": HIDDEN_DIMS,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUTS,
        "learning_rate": LEARNING_RATES,
        "batch_size": BATCH_SIZES,
        "seq_length": SEQ_LENGTHS,
    }
    for key, options in choices.items():
        if random.random() < mutation_rate:
            values[key] = random.choice(options)
    if values["embed_dim"] % values["num_heads"] != 0:
        values["num_heads"] = random.choice([h for h in NUM_HEADS if values["embed_dim"] % h == 0])
    return Genome.from_dict(values)
