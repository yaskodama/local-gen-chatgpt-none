from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import torch

from evaluate import evaluate_text
from genome import Genome, crossover, mutate
from model import sample_text
from train import build_vocab, load_text, train_short


@dataclass
class IndividualResult:
    genome: Genome
    score: float
    loss: float
    train_loss: float
    valid_loss: float | None
    valid_ppl: float | None
    valid_bits_per_char: float | None
    valid_bits_per_byte: float | None
    generated: str
    metrics: dict[str, float]
    checkpoint_path: str


def run_ga(
    data_path: str = "data/train.txt",
    valid_path: str | None = "data/valid.txt",
    results_dir: str = "results",
    generations: int = 3,
    population_size: int = 6,
    elite_size: int = 2,
    max_steps: int = 35,
    seed: int = 42,
    device: str = "cpu",
    resume: bool = False,
    mutation_rate: float = 0.25,
) -> IndividualResult:
    random.seed(seed)
    torch.manual_seed(seed)
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)

    text = load_text(data_path)
    valid_text = load_text(valid_path) if valid_path and Path(valid_path).exists() else None
    vocab_text = text + (valid_text or "")
    char_to_idx, idx_to_char = build_vocab(vocab_text)
    vocab_payload = {"char_to_idx": char_to_idx, "idx_to_char": {str(k): v for k, v in idx_to_char.items()}}
    (results / "vocab.json").write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    population = initial_population(results, population_size, resume=resume)
    best: IndividualResult | None = None
    best_valid: IndividualResult | None = None
    log_path = results / "log.jsonl"
    start_generation = next_generation_index(log_path) if resume else 0
    if not resume:
        log_path.write_text("", encoding="utf-8")

    for offset in range(generations):
        generation = start_generation + offset
        evaluated = [
            evaluate_individual(
                genome=genome,
                generation=generation,
                index=index,
                text=text,
                valid_text=valid_text,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                results_dir=results,
                max_steps=max_steps,
                device=device,
            )
            for index, genome in enumerate(population)
        ]
        evaluated.sort(key=lambda item: item.score, reverse=True)
        if best is None or evaluated[0].score > best.score:
            best = evaluated[0]
            save_best(best, char_to_idx, idx_to_char, results)
        valid_candidates = [item for item in evaluated if item.valid_loss is not None]
        if valid_candidates:
            generation_best_valid = min(valid_candidates, key=lambda item: item.valid_loss or float("inf"))
            if best_valid is None or (generation_best_valid.valid_loss or float("inf")) < (
                best_valid.valid_loss or float("inf")
            ):
                best_valid = generation_best_valid
                save_named_best(best_valid, char_to_idx, idx_to_char, results, "best_valid")

        with log_path.open("a", encoding="utf-8") as f:
            for rank, item in enumerate(evaluated):
                f.write(
                    json.dumps(
                        {
                            "generation": generation,
                            "rank": rank,
                            "score": item.score,
                            "loss": item.loss,
                            "train_loss": item.train_loss,
                            "valid_loss": item.valid_loss,
                            "valid_ppl": item.valid_ppl,
                            "valid_bits_per_char": item.valid_bits_per_char,
                            "valid_bits_per_byte": item.valid_bits_per_byte,
                            "checkpoint_path": item.checkpoint_path,
                            "genome": item.genome.to_dict(),
                            "metrics": item.metrics,
                            "generated": item.generated,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        elites = [item.genome for item in evaluated[:elite_size]]
        population = elites[:]
        while len(population) < population_size:
            if len(elites) >= 2:
                parent_a, parent_b = random.sample(elites, k=2)
            else:
                parent_a = parent_b = elites[0]
            population.append(mutate(crossover(parent_a, parent_b), mutation_rate=mutation_rate))

        print(
            f"generation={generation} best_score={evaluated[0].score:.4f} "
            f"loss={evaluated[0].loss:.4f} valid_ppl={evaluated[0].valid_ppl} "
            f"genome={evaluated[0].genome.to_dict()}"
        )

    if best is None:
        raise RuntimeError("GA did not evaluate any individual.")
    return best


def initial_population(results_dir: Path, population_size: int, resume: bool) -> list[Genome]:
    population: list[Genome] = []
    best_path = results_dir / "best_valid_genome.json"
    if not best_path.exists():
        best_path = results_dir / "best_genome.json"
    if resume and best_path.exists():
        best = Genome.from_dict(json.loads(best_path.read_text(encoding="utf-8")))
        population.append(best)
        while len(population) < population_size:
            population.append(mutate(best, mutation_rate=0.45))
    else:
        population = [Genome.random() for _ in range(population_size)]
    return population


def next_generation_index(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    last_generation = -1
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            last_generation = max(last_generation, int(payload.get("generation", -1)))
    return last_generation + 1


def evaluate_individual(
    genome: Genome,
    generation: int,
    index: int,
    text: str,
    valid_text: str | None,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    results_dir: Path,
    max_steps: int,
    device: str,
) -> IndividualResult:
    train_result = train_short(
        genome,
        text,
        char_to_idx,
        max_steps=max_steps,
        device=device,
        valid_text=valid_text,
    )
    generated = sample_text(
        train_result.model,
        start_text=text[: min(12, len(text))],
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        length=120,
        temperature=0.8,
        device=device,
    )
    metrics = evaluate_text(generated, text, train_result.loss)
    checkpoint_path = results_dir / f"gen{generation:03d}_idx{index:03d}.pt"
    torch.save(
        {
            "genome": genome.to_dict(),
            "model_state": train_result.model.state_dict(),
            "loss": train_result.loss,
            "train_loss": train_result.train_loss,
            "valid_loss": train_result.valid_loss,
            "valid_ppl": train_result.valid_ppl,
            "valid_bits_per_char": train_result.valid_bits_per_char,
            "valid_bits_per_byte": train_result.valid_bits_per_byte,
            "score": metrics["score"],
        },
        checkpoint_path,
    )
    return IndividualResult(
        genome=genome,
        score=metrics["score"],
        loss=train_result.loss,
        train_loss=train_result.train_loss,
        valid_loss=train_result.valid_loss,
        valid_ppl=train_result.valid_ppl,
        valid_bits_per_char=train_result.valid_bits_per_char,
        valid_bits_per_byte=train_result.valid_bits_per_byte,
        generated=generated,
        metrics=metrics,
        checkpoint_path=str(checkpoint_path),
    )


def save_best(
    result: IndividualResult,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    results_dir: Path,
) -> None:
    save_named_best(result, char_to_idx, idx_to_char, results_dir, "best")


def save_named_best(
    result: IndividualResult,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    results_dir: Path,
    name: str,
) -> None:
    source = torch.load(result.checkpoint_path, map_location="cpu")
    source["char_to_idx"] = char_to_idx
    source["idx_to_char"] = idx_to_char
    torch.save(source, results_dir / f"{name}_model.pt")
    (results_dir / f"{name}_genome.json").write_text(
        json.dumps(result.genome.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_dir / f"{name}_sample.txt").write_text(result.generated, encoding="utf-8")
