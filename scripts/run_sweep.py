"""Solve the cyclic duty model for P = P*, P*+1, ... and compare with the real schedule.

Example:
    python scripts/run_sweep.py --regions "BORNOVA 1,BORNOVA 2,BORNOVA 3,BORNOVA 4" \
        --R 2.0 --extra 2 --name bornova

Writes results/<name>_R<R>/{meta.json, sweep.csv, real.json, schedule_P<P>.csv}, with the
decimal point of R written as "p" (R = 1.5 -> results/<name>_R1p5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmacy.evaluate import evaluate  # noqa: E402
from pharmacy.instance import (ROOT, build_instance, load_real_schedule,  # noqa: E402
                               result_dir_name)
from pharmacy.model import (min_cover, min_cover_containing,  # noqa: E402
                            stage1_search, stage2)


def _py(d: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--regions", required=True, help="comma-separated İzmir duty regions")
    p.add_argument("--R", type=float, required=True, help="service radius in km")
    p.add_argument("--name", required=True)
    p.add_argument("--P", type=int, nargs="*", help="explicit P values (default P*..P*+extra)")
    p.add_argument("--extra", type=int, default=2)
    p.add_argument("--t1", type=float, default=60, help="time limit per Stage-1 feasibility solve (s)")
    p.add_argument("--t2", type=float, default=300, help="time limit for Stage 2 (s)")
    p.add_argument("--no-stage2", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    inst = build_instance([r.strip() for r in a.regions.split(",")], a.R, a.name)
    out = ROOT / "results" / result_dir_name(a.name, a.R)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"name": a.name, "regions": [r.strip() for r in a.regions.split(",")], "R": a.R},
        ensure_ascii=False, indent=2))
    print(f"{inst.name}: n={inst.n}, R={a.R} km, dropped outliers={list(inst.dropped['name'])}, "
          f"always needed={inst.always_needed}")

    X_real, days, unmatched = load_real_schedule(inst)
    real = _py(evaluate(inst, X_real, cyclic=False))
    real["unmatched_rows"] = unmatched
    (out / "real.json").write_text(json.dumps(real, indent=2))
    print("real schedule:", real)

    P_star, _ = min_cover(inst)
    P_i = min_cover_containing(inst)
    P_once = max(P_i)
    Ps = a.P or list(range(P_once, P_once + a.extra + 1))
    print(f"P* = {P_star}; every pharmacy can be on duty only if P >= max_i P_i = {P_once} "
          f"({sum(v > P_star for v in P_i)} pharmacies are in no minimum cover); solving P = {Ps}")

    rows = []
    for P in Ps:
        if P < P_once:
            print(f"P={P} < {P_once}: infeasible, skipped")
            continue
        s1 = stage1_search(inst, P, time_limit=a.t1, verbose=a.verbose)
        T = s1.T
        print(f"P={P}: T={T} G*={s1.G} ({s1.seconds:.0f}s) {' '.join(s1.log)}")
        if s1.G is None:
            rows.append({"P": P, "T": T or None, "G_star": None})
            continue
        X, s2_info = s1.X, {}
        if not a.no_stage2:
            s2 = stage2(inst, P, T, s1.G, start=s1.X, time_limit=a.t2, verbose=a.verbose)
            print(f"   stage 2: access={s2.access} km, gap={s2.gap} ({s2.seconds:.0f}s)")
            if s2.X is not None:
                X = s2.X
            s2_info = {"stage2_mip_gap": s2.gap, "stage2_seconds": round(s2.seconds)}
        ev = _py(evaluate(inst, X, cyclic=True))
        rows.append({"P": P, "T": T, "P_star": P_star, "G_star": s1.G, "G_proven": s1.proven, "stage1_seconds": round(s1.seconds),
                     **s2_info, **ev})
        sched = pd.DataFrame(X, columns=[f"d{t + 1}" for t in range(T)])
        sched.insert(0, "name", inst.pharmacies["name"])
        sched.insert(0, "pharmacy_id", inst.pharmacies["pharmacy_id"])
        sched.to_csv(out / f"schedule_P{P}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.csv", index=False)
    cols = [c for c in ["P", "T", "G_star", "G_proven", "ideal_gap", "rho", "load_max", "gaps_le_3",
                        "uncovered_pharmacy_days", "access_mean_km", "access_worst_km",
                        "stage2_mip_gap"] if c in df]
    print(df[cols].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
