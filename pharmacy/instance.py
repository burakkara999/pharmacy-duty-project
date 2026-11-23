"""Problem instances: pharmacies, distances, neighbourhoods N_i, and the real schedule.

Notation follows Section 3 of proposal_pharmacy.tex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EARTH_KM = 6371.0


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distances in km between all pairs of points (symmetric)."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    h = (np.sin((la.T - la) / 2) ** 2
         + np.cos(la) * np.cos(la.T) * np.sin((lo.T - lo) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@dataclass
class Instance:
    name: str
    pharmacies: pd.DataFrame          # one row per i, index 0..n-1
    dist: np.ndarray                  # d_ij in km
    R: float                          # service radius in km
    N: list[list[int]] = field(init=False)   # N_i, sorted by distance (i first)
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.N = [list(np.argsort(row, kind="stable")[: int((row <= self.R).sum())])
                  for row in self.dist]

    @property
    def n(self) -> int:
        return len(self.pharmacies)

    @property
    def always_needed(self) -> list[int]:
        """Pharmacies with N_i = {i}: on duty every day, exempt from the gap constraints."""
        return [i for i, Ni in enumerate(self.N) if len(Ni) == 1]


def result_dir_name(name: str, R: float) -> str:
    """Results folder name, e.g. ("bornova", 1.5) -> "bornova_R1p5" (no dots in paths)."""
    return f"{name}_R{R:g}".replace(".", "p")


def latest_pharmacy_file() -> Path:
    files = sorted(RAW.glob("izmir_pharmacies_*.csv"))
    if not files:
        raise FileNotFoundError("run scripts/fetch_izmir_data.py pharmacies first")
    return files[-1]


def build_instance(regions: list[str], R: float, name: str | None = None,
                   outlier_km: float = 6.0, path: Path | None = None) -> Instance:
    """Pharmacies of the given İzmir duty regions.

    Pharmacies farther than ``outlier_km`` from the cluster's median location are
    treated as geocoding errors and dropped (reported in ``Instance.dropped``).
    """
    df = pd.read_csv(path or latest_pharmacy_file(), dtype={"pharmacy_id": str})
    df["region"] = df["region"].str.strip()
    df = df[df["region"].isin(regions) & df["lat"].notna() & df["lon"].notna()]
    missing = set(regions) - set(df["region"])
    if missing:
        raise ValueError(f"unknown regions: {sorted(missing)}")
    centre = haversine_matrix(
        np.r_[df["lat"].median(), df["lat"].to_numpy()],
        np.r_[df["lon"].median(), df["lon"].to_numpy()])[0, 1:]
    keep = centre <= outlier_km
    ph = df[keep].reset_index(drop=True)
    dist = haversine_matrix(ph["lat"].to_numpy(), ph["lon"].to_numpy())
    return Instance(name or "+".join(regions), ph, dist, R, dropped=df[~keep])


_norm_re = re.compile(r"[^A-ZÇĞİÖŞÜ0-9]")


def _norm(s: str) -> str:
    return _norm_re.sub("", str(s).upper().replace("ECZANESİ", ""))


def load_real_schedule(inst: Instance, path: Path | None = None
                       ) -> tuple[np.ndarray, list[str], int]:
    """Real duty schedule restricted to the instance's pharmacies.

    Returns ``(X, days, n_unmatched)`` where ``X[i, t] = 1`` if pharmacy i was on duty
    on ``days[t]``. A duty row is matched to an instance pharmacy by normalised name
    within 300 m, or by location alone within 50 m. Rows of the instance's regions that
    cannot be matched (e.g. a dropped outlier) are counted in ``n_unmatched``.
    """
    duty = pd.read_csv(path or RAW / "izmir_duty.csv")
    days = sorted(duty["date"].unique())
    day_idx = {d: t for t, d in enumerate(days)}
    ph = inst.pharmacies
    names = ph["name"].map(_norm).to_numpy()
    regions = set(ph["region"])
    X = np.zeros((inst.n, len(days)), dtype=int)
    unmatched = 0
    for r in duty.itertuples():
        d = haversine_matrix(np.r_[r.lat, ph["lat"].to_numpy()],
                             np.r_[r.lon, ph["lon"].to_numpy()])[0, 1:]
        same = names == _norm(r.name)
        cand = np.where(same & (d <= 0.3))[0]
        if not len(cand):
            cand = np.where(d <= 0.05)[0]
        if len(cand):
            X[cand[np.argmin(d[cand])], day_idx[r.date]] = 1
        elif r.region.strip() in regions:
            unmatched += 1
    return X, days, unmatched
