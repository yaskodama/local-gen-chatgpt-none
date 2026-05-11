from __future__ import annotations

import argparse

from ga import run_ga


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve tiny character-level generators with GA.")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--valid", default="data/valid.txt")
    parser.add_argument("--results", default="results")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Continue from results/best_genome.json when present.")
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    args = parser.parse_args()

    best = run_ga(
        data_path=args.data,
        valid_path=args.valid,
        results_dir=args.results,
        generations=args.generations,
        population_size=args.population,
        elite_size=args.elite,
        max_steps=args.steps,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        mutation_rate=args.mutation_rate,
    )
    print("\nBest individual")
    print(f"score: {best.score:.4f}")
    print(f"loss: {best.loss:.4f}")
    if best.valid_loss is not None:
        print(f"train_loss: {best.train_loss:.4f}")
        print(f"valid_loss: {best.valid_loss:.4f}")
        print(f"valid_ppl: {best.valid_ppl:.4f}")
        print(f"valid_bits_per_char: {best.valid_bits_per_char:.4f}")
        print(f"valid_bits_per_byte: {best.valid_bits_per_byte:.4f}")
    print(f"genome: {best.genome.to_dict()}")
    print("saved: results/best_model.pt")


if __name__ == "__main__":
    main()
