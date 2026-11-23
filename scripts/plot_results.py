"""Figures for the report from results/*/sweep.csv and real.json.

    python scripts/plot_results.py

Writes results/figures/{equity_vs_P,access_vs_P}.{png,pdf}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"

# Reference palette (dataviz skill), light mode; first three categorical slots.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BASELINE = "#3d3c39"
SERIES = [  # (results folder, label, colour) — colour follows the instance
    ("bornova_R2", "Bornova, R = 2 km", "#2a78d6"),
    ("bornova_R1p5", "Bornova, R = 1.5 km", "#eb6834"),
    ("balcova_R1p5", "Balçova–Narlıdere, R = 1.5 km", "#1baf7a"),
]
REAL = [  # current schedule, one point per area (independent of R for equity)
    ("bornova_R2", "Bornova"),
    ("balcova_R1p5", "Balçova–N."),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _axes(title: str, subtitle: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    fig.subplots_adjust(left=0.09, right=0.70, top=0.82, bottom=0.13)
    fig.text(0.1, 0.94, title, fontsize=12.5, fontweight="bold", color=TEXT)
    fig.text(0.1, 0.885, subtitle, fontsize=9.5, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Pharmacies on duty per day, P")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    return fig, ax


def _plot(metric: str, real_key: str, title: str, subtitle: str, ylabel: str,
          fname: str, ylim=None, fmt="{:.2f}", below: tuple[str, ...] = ()) -> None:
    """``below``: series whose end label goes under the last point instead of right of it."""
    fig, ax = _axes(title, subtitle, ylabel)
    for folder, label, colour in SERIES:
        df = pd.read_csv(RES / folder / "sweep.csv").dropna(subset=[metric])
        ax.plot(df["P"], df[metric], color=colour, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, zorder=3)
        last = df.iloc[-1]
        under = folder in below
        ax.annotate(f"{label}\n{fmt.format(last[metric])}", (last["P"], last[metric]),
                    xytext=(0, -12) if under else (10, 0), textcoords="offset points",
                    ha="center" if under else "left", va="top" if under else "center",
                    fontsize=8.5, color=TEXT, annotation_clip=False)
    for folder, label in REAL:
        real = json.loads((RES / folder / "real.json").read_text())
        x, y = real["P_mean"], real[real_key]
        ax.plot([x], [y], marker="D", markersize=8, markerfacecolor=SURFACE,
                markeredgecolor=BASELINE, markeredgewidth=1.8, linestyle="none",
                zorder=4, label="Current plan (40 days)" if folder == REAL[0][0] else None)
        ax.annotate(f"{label}\n{fmt.format(y)}", (x, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=TEXT_2)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(2.4, 8.6)
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), frameon=False,
              fontsize=8.5, labelcolor=TEXT_2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200)
    plt.close(fig)
    print("->", OUT / f"{fname}.png")


def main() -> None:
    _plot("rho", "rho",
          "Equity: how regular the rotation is",
          "ρ = shortest gap between duties ÷ average gap (1 = perfectly regular)",
          "ρ", "equity_vs_P", ylim=(0, 1.05))
    _plot("access_mean_km", "access_mean_km",
          "Access: distance to the nearest on-duty pharmacy",
          "Mean over all pharmacies and days; model values are Stage-2 incumbents (≤13% from optimal)",
          "Mean distance (km)", "access_vs_P", ylim=(0, 0.95), below=("balcova_R1p5",))


if __name__ == "__main__":
    main()
