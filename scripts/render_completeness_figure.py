#!/usr/bin/env python3
"""Render the paper completeness figure from saved few-shot records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHOT_KEYS = ("1", "2", "4", "8", "16", "full")
SHOT_LABELS = ("1", "2", "4", "8", "16", "All")

DATASETS = (
    {
        "name": "CIFAR-100",
        "results": REPOSITORY_ROOT / "results/completeness/cifar100_rn50/results_raw.json",
        "metaphor_key": "MetaphorCBM-PA",
        "grid": (0, slice(0, 3)),
    },
    {
        "name": "CIFAR-10",
        "results": REPOSITORY_ROOT / "results/completeness/cifar10_rn50/results_raw.json",
        "metaphor_key": "MetaphorCBM-PA",
        "grid": (0, slice(3, 6)),
    },
    {
        "name": "Tiny-ImageNet",
        "results": REPOSITORY_ROOT / "results/completeness/tiny_imagenet_rn50/results_raw.json",
        "metaphor_key": "SAE-CBM",
        "grid": (1, slice(0, 2)),
    },
    {
        "name": "CUB-200",
        "results": REPOSITORY_ROOT / "results/completeness/cub200_rn50/results_raw.json",
        "metaphor_key": "SAE-CBM",
        "grid": (1, slice(2, 4)),
    },
    {
        "name": "MS-COCO",
        "results": REPOSITORY_ROOT / "results/completeness/coco_rn50/results_raw.json",
        "metaphor_key": "MetaphorCBM-PA",
        "grid": (1, slice(4, 6)),
    },
)

LP_COLOR = "#244A73"
METAPHOR_COLOR = "#C84C35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the five-panel completeness figure from saved few-shot JSON records."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for completeness_fewshot.pdf and its PNG preview.",
    )
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def load_means(path: Path, method_key: str) -> List[float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if method_key not in payload:
        raise KeyError(f"Method {method_key!r} is absent from {path}")

    means: List[float] = []
    for shot in SHOT_KEYS:
        values: Sequence[float] = payload[method_key][shot]
        if not values:
            raise ValueError(f"No results for method={method_key!r}, shot={shot!r} in {path}")
        means.append(sum(float(value) for value in values) / len(values))
    return means


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def padded_limits(values: Sequence[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    span = max(upper - lower, 1.0)
    margin = max(0.045 * span, 0.75)
    return max(0.0, lower - margin), min(100.0, upper + margin)


def render(output_dir: Path, png_dpi: int) -> tuple[Path, Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.25, 4.55), facecolor="white")
    grid = figure.add_gridspec(2, 6)
    x_values = list(range(len(SHOT_KEYS)))

    for dataset in DATASETS:
        axis = figure.add_subplot(grid[dataset["grid"]])
        results_path = Path(dataset["results"])
        lp_means = load_means(results_path, "LP")
        metaphor_means = load_means(results_path, str(dataset["metaphor_key"]))

        axis.plot(
            x_values,
            lp_means,
            color=LP_COLOR,
            linestyle=(0, (4.2, 2.4)),
            linewidth=1.65,
            marker="o",
            markersize=4.5,
            markerfacecolor=LP_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
        axis.plot(
            x_values,
            metaphor_means,
            color=METAPHOR_COLOR,
            linestyle="-",
            linewidth=1.8,
            marker="D",
            markersize=4.5,
            markerfacecolor=METAPHOR_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=4,
        )

        axis.set_title(str(dataset["name"]), loc="left", fontweight="bold", pad=4.0)
        axis.set_xticks(x_values, SHOT_LABELS)
        axis.set_xlim(-0.16, len(SHOT_KEYS) - 0.84)
        axis.set_ylim(*padded_limits(lp_means + metaphor_means))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.grid(axis="y", color="#D5D8DC", linewidth=0.55, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#70757A")
        axis.spines["bottom"].set_color("#70757A")
        axis.spines["left"].set_linewidth(0.65)
        axis.spines["bottom"].set_linewidth(0.65)
        axis.tick_params(axis="both", width=0.65, length=3.0, color="#70757A")

    legend_handles = (
        Line2D(
            [0],
            [0],
            color=LP_COLOR,
            linestyle=(0, (4.2, 2.4)),
            linewidth=1.65,
            marker="o",
            markersize=4.8,
            markerfacecolor=LP_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label="Black-box LP",
        ),
        Line2D(
            [0],
            [0],
            color=METAPHOR_COLOR,
            linestyle="-",
            linewidth=1.8,
            marker="D",
            markersize=4.8,
            markerfacecolor=METAPHOR_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label="MetaphorCBM",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    figure.supxlabel("Labeled samples per class", y=0.035, fontsize=8.9)
    figure.supylabel("Top-1 accuracy (%)", x=0.018, fontsize=8.9)
    figure.subplots_adjust(
        left=0.078,
        right=0.992,
        bottom=0.125,
        top=0.895,
        wspace=0.72,
        hspace=0.48,
    )

    pdf_path = output_dir / "completeness_fewshot.pdf"
    png_path = output_dir / "completeness_fewshot.png"
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": "MetaphorCBM completeness evaluation"},
    )
    figure.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    pdf_path, png_path = render(args.output_dir, args.png_dpi)
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
