"""Lexicographic cyclic duty model of Section 3 (sec:model) in proposal_pharmacy.tex.

Stage 0  P* by static location set covering                  (eq:pstar)
Stage 1  max G^min, solved by bisection over g with the hard
         window constraint sum_{s in W_g(t)} x_is <= 1          (Remark "Solving Stage 1")
Stage 2  min access distance f_2 subject to G^min >= G*(P)    (eq:f2, eq:m-assign, eq:m-open)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .instance import Instance


def _model(name: str, time_limit: float | None, verbose: bool) -> gp.Model:
    m = gp.Model(name)
    m.Params.OutputFlag = int(verbose)
    if time_limit:
        m.Params.TimeLimit = time_limit
    return m


def min_cover(inst: Instance, verbose: bool = False) -> tuple[int, list[int]]:
    """Stage 0: P* and one minimum cover (eq:pstar)."""
    m = _model("pstar", None, verbose)
    u = m.addVars(inst.n, vtype=GRB.BINARY, name="u")
    m.addConstrs((gp.quicksum(u[j] for j in inst.N[i]) >= 1 for i in range(inst.n)), name="cover")
    m.setObjective(u.sum(), GRB.MINIMIZE)
    m.optimize()
    return round(m.ObjVal), [j for j in range(inst.n) if u[j].X > 0.5]


def min_cover_containing(inst: Instance, verbose: bool = False) -> list[int]:
    """P_i = size of the smallest cover that contains pharmacy i.

    Because every pharmacy must be on duty at least once (eq:m-once), P must be at least
    max_i P_i, which can exceed P*.
    """
    m = _model("pi", None, verbose)
    u = m.addVars(inst.n, vtype=GRB.BINARY, name="u")
    m.addConstrs((gp.quicksum(u[j] for j in inst.N[i]) >= 1 for i in range(inst.n)), name="cover")
    m.setObjective(u.sum(), GRB.MINIMIZE)
    out = []
    for i in range(inst.n):
        u[i].LB = 1
        m.optimize()
        out.append(round(m.ObjVal))
        u[i].LB = 0
    return out


def gap_upper_bound(inst: Instance) -> int:
    """G^min <= min_i |N_i|.

    If every gap is at least g, each pharmacy is on duty at most once in any window of g
    days, but pharmacy i needs an on-duty pharmacy in N_i on each of those g days, so
    |N_i| >= g. Pharmacies whose N_i contains an always-needed pharmacy are covered
    every day anyway and give no bound.
    """
    always = set(inst.always_needed)
    sizes = [len(Ni) for Ni in inst.N if not always.intersection(Ni)]
    return min(sizes) if sizes else 10**9


def cycle_lengths(n: int, P: int, factor: float = 3.0):
    """Candidate |T| >= n/P, coprime to 7 (weekday rotation remark), shortest first."""
    T0 = math.ceil(n / P)
    return [T for T in range(T0, int(factor * T0) + 1) if math.gcd(T, 7) == 1]


def shortest_feasible_cycle(inst: Instance, P: int, time_limit: float = 30,
                            verbose: bool = False) -> tuple[int | None, list[str]]:
    """Smallest |T| for which eq:m-cover, eq:m-count and eq:m-once are feasible."""
    log = []
    for T in cycle_lengths(inst.n, P):
        ok, _ = feasible(inst, P, T, 1, time_limit, verbose)
        log.append(f"T={T}:{ok}")
        if ok:
            return T, log
    return None, log


def _base(inst: Instance, P: int, T: int, name: str, time_limit: float | None,
          verbose: bool, fix_rotation: bool = True):
    """Model with x and constraints eq:m-cover, eq:m-count, eq:m-once."""
    m = _model(name, time_limit, verbose)
    x = m.addVars(inst.n, T, vtype=GRB.BINARY, name="x")
    m.addConstrs((gp.quicksum(x[j, t] for j in inst.N[i]) >= 1
                  for i in range(inst.n) for t in range(T)), name="cover")
    m.addConstrs((x.sum("*", t) == P for t in range(T)), name="count")
    m.addConstrs((x.sum(i, "*") >= 1 for i in range(inst.n)), name="once")
    for i in inst.always_needed:
        for t in range(T):
            x[i, t].LB = 1
    if fix_rotation:
        # Rotational symmetry of a cyclic plan without exceptions: some duty of the
        # most constrained pharmacy can be moved to day 1.
        i0 = min(range(inst.n), key=lambda i: len(inst.N[i]))
        x[i0, 0].LB = 1
    return m, x


def _add_windows(m: gp.Model, x, inst: Instance, T: int, g: int) -> None:
    """Hard version of eq:m-mingap: every gap >= g (always-needed pharmacies exempt)."""
    if g <= 1:
        return
    exempt = set(inst.always_needed)
    m.addConstrs((gp.quicksum(x[i, (t + s) % T] for s in range(g)) <= 1
                  for i in range(inst.n) if i not in exempt for t in range(T)),
                 name=f"win{g}")


def _solution(x, n: int, T: int) -> np.ndarray:
    return np.array([[round(x[i, t].X) for t in range(T)] for i in range(n)], dtype=int)


@dataclass
class StageResult:
    P: int
    T: int
    G: int | None = None
    X: np.ndarray | None = None
    access: float | None = None
    gap: float | None = None
    log: list[str] = field(default_factory=list)
    seconds: float = 0.0
    proven: bool = True


def feasible(inst: Instance, P: int, T: int, g: int, time_limit: float, verbose=False,
             start: np.ndarray | None = None) -> tuple[bool | None, np.ndarray | None]:
    """Is there a schedule with every gap >= g? None = unknown (time limit)."""
    m, x = _base(inst, P, T, f"feas_g{g}", time_limit, verbose)
    _add_windows(m, x, inst, T, g)
    m.Params.SolutionLimit = 1
    m.Params.MIPFocus = 1
    if start is not None:
        for (i, t), v in x.items():
            v.Start = start[i, t]
    m.optimize()
    if m.SolCount:
        return True, _solution(x, inst.n, T)
    if m.Status == GRB.INFEASIBLE:
        return False, None
    return None, None


def stage1_search(inst: Instance, P: int, time_limit: float = 60, verbose=False
                  ) -> StageResult:
    """Stage 1 with the cycle length chosen as well: G*(P) = max over |T| of G*(P, |T|).

    Upper bounds: G^min <= n/P (average gap), G^min <= min_i |N_i| (gap_upper_bound), and
    G^min <= floor(|T|/2) whenever P|T| > n, because a pharmacy with two duties per cycle
    has a gap <= |T|/2. Cycles of 2n/P days or more, where some pharmacy needs three
    duties, are not tried.

    Start: bisection over g at the shortest feasible cycle. Then bisection over g between
    that value and the bound, where "g is achievable" means: some cycle |T| with
    floor(|T|/2) >= g admits a schedule with all gaps >= g. Cycles are scanned from the
    shortest, so the result uses the shortest cycle (fewest extra duties) attaining G*.
    ``StageResult.proven`` is False if a solve hit the time limit (then G* is a lower
    bound).
    """
    t0 = time.time()
    log = []
    T_min, tlog = shortest_feasible_cycle(inst, P, time_limit, verbose)
    log += tlog
    if T_min is None:
        res = StageResult(P, 0, log=log)
        res.seconds = time.time() - t0
        return res
    best = stage1(inst, P, T_min, time_limit, verbose)
    proven = best.proven
    bound = min(inst.n // P, gap_upper_bound(inst))
    log.append(f"T={T_min}: G*={best.G} by bisection; bound={bound}")
    cands = [T for T in cycle_lengths(inst.n, P) if T_min < T < 2 * inst.n / P]
    lo, hi = best.G, bound
    while lo < hi:
        g = (lo + hi + 1) // 2
        found = None
        for T in (T for T in cands if T // 2 >= g):
            ok, X = feasible(inst, P, T, g, time_limit, verbose)
            if ok is None:
                proven = False
            if ok:
                found = StageResult(P, T, g, X)
                break
        log.append(f"g={g}:{'T=' + str(found.T) if found else 'none'}")
        if found:
            lo, best = g, found
        else:
            hi = g - 1
    best.proven = proven
    best.log = log
    best.seconds = time.time() - t0
    return best


def stage1(inst: Instance, P: int, T: int, time_limit: float = 60, verbose=False
           ) -> StageResult:
    """Largest g such that a schedule with all gaps >= g exists (bisection), fixed |T|."""
    t0 = time.time()
    res = StageResult(P, T)
    ok, X = feasible(inst, P, T, 1, time_limit, verbose)
    res.log.append(f"g=1: {ok}")
    if not ok:
        res.seconds = time.time() - t0
        return res
    lo, hi, best = 1, min(T, inst.n // P), X     # G^min <= |I|/P
    while lo < hi:
        g = (lo + hi + 1) // 2
        ok, X = feasible(inst, P, T, g, time_limit, verbose)
        res.log.append(f"g={g}: {ok}")
        if ok is None:
            res.proven = False
        if ok:
            lo, best = g, X
        else:          # infeasible, or unknown within the time limit (conservative)
            hi = g - 1
    res.G, res.X = lo, best
    res.seconds = time.time() - t0
    return res


def stage2(inst: Instance, P: int, T: int, G: int, start: np.ndarray | None = None,
           time_limit: float = 300, mip_gap: float = 0.01, verbose=False) -> StageResult:
    """Minimise total distance to the nearest on-duty pharmacy s.t. all gaps >= G."""
    t0 = time.time()
    m, x = _base(inst, P, T, "access", time_limit, verbose)
    _add_windows(m, x, inst, T, G)
    m.Params.MIPGap = mip_gap
    idx = [(i, int(j), t) for i, Ni in enumerate(inst.N) for j in Ni for t in range(T)]
    w = m.addVars(idx, lb=0.0, ub=1.0, name="w")
    m.addConstrs((gp.quicksum(w[i, j, t] for j in inst.N[i]) == 1
                  for i in range(inst.n) for t in range(T)), name="assign")
    m.addConstrs((w[i, j, t] <= x[j, t] for (i, j, t) in idx), name="open")
    m.setObjective(gp.quicksum(inst.dist[i, j] * w[i, j, t] for (i, j, t) in idx),
                   GRB.MINIMIZE)
    if start is not None:
        for (i, t), v in x.items():
            v.Start = start[i, t]
    m.optimize()
    res = StageResult(P, T, G)
    if m.SolCount:
        res.X = _solution(x, inst.n, T)
        res.access = m.ObjVal / (inst.n * T)       # mean km per pharmacy-day
        res.gap = m.MIPGap
    res.log.append(f"status={m.Status} sols={m.SolCount} gap={getattr(m, 'MIPGap', None)}")
    res.seconds = time.time() - t0
    return res
