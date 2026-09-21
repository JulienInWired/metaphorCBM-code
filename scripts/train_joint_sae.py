"""Train paired image and text sparse autoencoders from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from metaphorcbm.training import SAETrainer


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("The training configuration must be a JSON object.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train paired image and text sparse autoencoders."
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration file")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint from which to resume training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = SAETrainer(load_config(args.config))
    resume_from = str(args.resume_from) if args.resume_from is not None else None
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
