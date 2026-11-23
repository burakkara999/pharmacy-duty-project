"""Metrics of a duty schedule X (|I| x days), for model output and the real schedule alike."""

from __future__ import annotations

import numpy as np

from .instance import Instance


def gaps(row: np.ndarray, cyclic: bool) -> list[int]:
    """Gaps between consecutive duties of one pharmacy.

    Cyclic: the schedule repeats, so the last->first gap across cycles is included and
    the gaps sum to len(row). Linear: only gaps observed inside the horizon.
    """
    days = np.flatnonzero(row)
    if len(days) == 0:
        return []
    g = list(np.diff(days))
    if cyclic:
        g.append(days[0] + len(row) - days[-1])
    return [int(v) for v in g]


def evaluate(inst: Instance, X: np.ndarray, cyclic: bool) -> dict:
    n, T = X.shape
    exempt = set(inst.always_needed)
    counts = X.sum(axis=0)
    loads = X.sum(axis=1)

    uncovered, access, access_max = [], [], []
    for t in range(T):
        on = np.flatnonzero(X[:, t])
        if len(on) == 0:
            uncovered.append(n)
            continue
        nearest = inst.dist[:, on].min(axis=1)
        uncovered.append(int((nearest > inst.R + 1e-9).sum()))
        access.append(nearest.mean())
        access_max.append(nearest.max())

    all_gaps = [g for i in range(n) if i not in exempt for g in gaps(X[i], cyclic)]
    g_min = min(all_gaps) if all_gaps else None
    P_mean = counts.mean()
    ideal_gap = n / P_mean            # average gap over all duties
    return {
        "days": T,
        "P_min": int(counts.min()), "P_max": int(counts.max()), "P_mean": round(float(P_mean), 2),
        "ideal_gap": round(ideal_gap, 2),
        "G_min": g_min,
        "rho": round(g_min / ideal_gap, 3) if g_min else None,
        "gaps_le_3": sum(g <= 3 for g in all_gaps),
        "load_min": int(loads.min()), "load_max": int(loads.max()),
        "never_on_duty": int((loads == 0).sum()),
        "uncovered_pharmacy_days": int(sum(uncovered)),
        "days_with_uncovered": int(sum(u > 0 for u in uncovered)),
        "access_mean_km": round(float(np.mean(access)), 3) if access else None,
        "access_worst_km": round(float(np.max(access_max)), 3) if access_max else None,
    }
