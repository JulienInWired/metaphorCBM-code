"""CLI entry point for the cross-modal coupling pipeline."""

from __future__ import annotations

from metaphorcbm.config import parse_args_with_config
from metaphorcbm.coupling.config import build_arg_parser, config_from_args
from metaphorcbm.coupling.pipeline import CrossModalCouplingTrainer


def main() -> None:
    parser = build_arg_parser()
    args = parse_args_with_config(parser)
    config = config_from_args(args)
    trainer = CrossModalCouplingTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
