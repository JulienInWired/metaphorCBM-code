"""JSON-backed command-line configuration helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def load_json_object(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    return payload


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Load parser defaults from JSON before applying command-line overrides."""

    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", type=str, default=None, help="JSON configuration file")

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=str, default=None)
    known, _ = probe.parse_known_args(argv)

    if known.config:
        payload = load_json_object(known.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(payload) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown configuration fields in {known.config}: "
                + ", ".join(unknown_keys)
            )
        parser.set_defaults(**payload)

    return parser.parse_args(argv)


__all__ = ["load_json_object", "parse_args_with_config"]
