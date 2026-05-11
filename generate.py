from __future__ import annotations

import argparse

import torch

from genome import Genome
from model import build_model, sample_text


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    genome = Genome.from_dict(checkpoint["genome"])
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = {int(k): v for k, v in checkpoint["idx_to_char"].items()}
    model = build_model(len(char_to_idx), genome).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, char_to_idx, idx_to_char


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from the best evolved char model.")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, char_to_idx, idx_to_char = load_checkpoint(args.checkpoint, args.device)
    print(
        sample_text(
            model,
            start_text=args.prompt,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            length=args.length,
            temperature=args.temperature,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
