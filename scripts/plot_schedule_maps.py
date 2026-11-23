"""One map per duty day, collected in a multi-page PDF.

Each page shows every pharmacy of the instance, the day's on-duty pharmacies with their
service radius R, a thin line from each pharmacy to its nearest on-duty pharmacy, and
(as a cross) any pharmacy with no on-duty pharmacy within R. All pages share the same
extent, so days can be compared by flipping pages.

    python scripts/plot_schedule_maps.py results/bornova_R2 --P 4       # model schedule
    python scripts/plot_schedule_maps.py results/bornova_R2 --P 4 5 6   # several
    python scripts/plot_schedule_maps.py results/bornova_R2 --real      # current plan
    python scripts/plot_schedule_maps.py results/bornova_R2 --all       # real + every P

The instance (regions, R) is read from <result dir>/meta.json, written by run_sweep.py.
Output: <result dir>/maps_P<P>.pdf or maps_real.pdf.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmacy.evaluate import evaluate  # noqa: E402
from pharmacy.instance import Instance, build_instance, load_real_schedule  # noqa: E402

# Reference palette (dataviz skill), light mode.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
ON_DUTY = "#2a78d6"        # categorical slot 1
CRITICAL = "#d03b3b"       # status: uncovered (always with a cross marker + legend label)
WEEKDAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def to_km(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection in km around the instance centre (circles stay round)."""
    lat, lon = inst.pharmacies["lat"].to_numpy(), inst.pharmacies["lon"].to_numpy()
    lat0, lon0 = lat.mean(), lon.mean()
    return (lon - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 110.57


def short(name: str) -> str:
    return name.replace("ECZANESİ", "").replace("ECZANESI", "").strip().title()


def load_model_schedule(inst: Instance, path: Path) -> np.ndarray:
    sched = pd.read_csv(path, dtype={"pharmacy_id": str}).set_index("pharmacy_id")
    days = [c for c in sched.columns if c.startswith("d")]
    return sched.loc[inst.pharmacies["pharmacy_id"], days].to_numpy(dtype=int)


def draw_day(ax, inst: Instance, x: np.ndarray, y: np.ndarray, on: np.ndarray,
             lines: bool) -> tuple[int, float]:
    R = inst.R
    for j in on:
        ax.add_patch(Circle((x[j], y[j]), R, facecolor=ON_DUTY, alpha=0.07,
                            edgecolor=ON_DUTY, linewidth=0.8, zorder=1))
    nearest_idx = on[np.argmin(inst.dist[:, on], axis=1)]
    nearest = inst.dist[np.arange(inst.n), nearest_idx]
    uncovered = nearest > R + 1e-9
    if lines:
        for i in range(inst.n):
            j = nearest_idx[i]
            if i != j:
                ax.plot([x[i], x[j]], [y[i], y[j]], color=GRID, linewidth=0.6, zorder=2)
    ax.scatter(x, y, s=9, color=MUTED, linewidths=0, zorder=3)
    ax.scatter(x[uncovered], y[uncovered], s=42, marker="x", color=CRITICAL,
               linewidths=1.6, zorder=4)
    ax.scatter(x[on], y[on], s=70, color=ON_DUTY, edgecolors=SURFACE, linewidths=2,
               zorder=5)
    names = inst.pharmacies["name"]
    for j in on:
        ax.annotate(short(names[j]), (x[j], y[j]), xytext=(6, 5), textcoords="offset points",
                    fontsize=8, color=TEXT, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    return int(uncovered.sum()), float(nearest.mean())


def scale_bar(ax, km: float = 1.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.04 * (y1 - y0)
    ax.plot([bx, bx + km], [by, by], color=TEXT_2, linewidth=2, solid_capstyle="butt")
    ax.text(bx + km / 2, by + 0.015 * (y1 - y0), f"{km:g} km", ha="center", va="bottom",
            fontsize=8, color=TEXT_2)


def legend(ax, R: float, show_uncovered: bool) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=ON_DUTY,
               markeredgecolor=SURFACE, label="Nöbetçi eczane"),
        Line2D([], [], marker="o", linestyle="none", markersize=4, markerfacecolor=MUTED,
               markeredgecolor=MUTED, label="Diğer eczaneler"),
        Line2D([], [], color=ON_DUTY, linewidth=0.8, label=f"Hizmet yarıçapı R = {R:g} km"),
        Line2D([], [], color=GRID, linewidth=1.2, label="En yakın nöbetçiye bağlantı"),
    ]
    if show_uncovered:
        handles.append(Line2D([], [], marker="x", linestyle="none", markersize=7,
                              markeredgecolor=CRITICAL, markeredgewidth=1.6,
                              label="R içinde nöbetçisi yok"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=len(handles), fontsize=8.5, frameon=False, labelcolor=TEXT_2,
              handletextpad=0.4, columnspacing=1.6)


def available_P(result_dir: Path) -> list[int]:
    return sorted(int(f.stem.removeprefix("schedule_P"))
                  for f in result_dir.glob("schedule_P*.csv"))


def schedule_job(inst: Instance, result_dir: Path, P: int | None):
    """(X, page labels, title kind, output path, plan summary) for the real plan (P=None)
    or for the model schedule with daily count P."""
    if P is None:
        X, dates, _ = load_real_schedule(inst)
        labels = [f"{d}  {WEEKDAYS_TR[dt.date.fromisoformat(d).weekday()]}" for d in dates]
        return X, labels, "Mevcut plan", result_dir / "maps_real.pdf", evaluate(inst, X, False)
    path = result_dir / f"schedule_P{P}.csv"
    if not path.exists():
        sys.exit(f"No model schedule for P={P} in {result_dir}. "
                 f"Available P: {available_P(result_dir) or 'none'} (run_sweep.py starts at "
                 f"the smallest P for which every pharmacy can be on duty).")
    X = load_model_schedule(inst, path)
    T = X.shape[1]
    labels = [f"Döngü günü {t + 1} / {T}" for t in range(T)]
    return X, labels, f"Model, P = {P}", result_dir / f"maps_P{P}.pdf", evaluate(inst, X, True)


def write_pdf(inst: Instance, regions: list[str], X: np.ndarray, labels: list[str],
              kind: str, out: Path, summary: dict, lines: bool, days: int | None) -> None:
    n_days = min(days or X.shape[1], X.shape[1])
    x, y = to_km(inst)
    pad = inst.R * 0.6
    xlim, ylim = (x.min() - pad, x.max() + pad), (y.min() - pad, y.max() + pad)
    region_txt = ", ".join(regions)

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})
    with PdfPages(out) as pdf:
        for t in range(n_days):
            on = np.flatnonzero(X[:, t])
            fig, ax = plt.subplots(figsize=(11.69, 8.27))        # A4 landscape
            fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.07)
            ax.set_facecolor(SURFACE)
            if len(on):
                n_unc, acc = draw_day(ax, inst, x, y, on, lines)
            else:
                ax.scatter(x, y, s=9, color=MUTED, linewidths=0)
                n_unc, acc = inst.n, float("nan")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            scale_bar(ax)
            legend(ax, inst.R, n_unc > 0)
            fig.text(0.03, 0.95, f"{labels[t]}  ·  {kind}", fontsize=14, fontweight="bold",
                     color=TEXT)
            fig.text(0.03, 0.915,
                     f"Bu gün: {len(on)} nöbetçi  ·  R içinde nöbetçisi olmayan eczane: {n_unc}"
                     f"  ·  en yakın nöbetçiye ortalama {acc:.2f} km",
                     fontsize=10, color=TEXT)
            fig.text(0.03, 0.885,
                     f"{region_txt} ({inst.n} eczane)  ·  tüm plan: en kısa nöbet aralığı "
                     f"{summary['G_min']} gün, ρ = {summary['rho']}, "
                     f"bir eczanenin en fazla nöbeti {summary['load_max']}",
                     fontsize=9, color=TEXT_2)
            pdf.savefig(fig)
            plt.close(fig)
        d = pdf.infodict()
        d["Title"] = f"Nöbet haritaları — {kind} — {region_txt}"
    print(f"{n_days} pages -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("result_dir", type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--P", type=int, nargs="+", help="model schedule(s) schedule_P<P>.csv")
    src.add_argument("--real", action="store_true", help="the current (real) schedule")
    src.add_argument("--all", action="store_true",
                     help="the real schedule and every model schedule in the folder")
    p.add_argument("--no-lines", action="store_true", help="omit nearest-on-duty lines")
    p.add_argument("--days", type=int, help="only the first N days")
    a = p.parse_args()

    meta = json.loads((a.result_dir / "meta.json").read_text())
    inst = build_instance(meta["regions"], meta["R"], meta["name"])
    if a.all:
        Ps = [None] + available_P(a.result_dir)
    elif a.real:
        Ps = [None]
    else:
        Ps = a.P
    for P in Ps:
        X, labels, kind, out, summary = schedule_job(inst, a.result_dir, P)
        write_pdf(inst, meta["regions"], X, labels, kind, out, summary,
                  not a.no_lines, a.days)


if __name__ == "__main__":
    main()
