"""
test_lbvh.py — Randomised validation of LBVHTree against a brute-force reference.

Each scenario builds a brute-force ground truth (all pairs i<j whose inflated AABBs
overlap) and checks that the LBVH:
  1. Contains every ground-truth pair  (no false negatives)
  2. Reports no pair where i == j      (no self-pairs)
  3. Reports no duplicate pair         (each pair at most once)

The AABB overlap condition with alert is symmetric:
    |x_i - x_j| <= r_i + r_j + alert   AND
    |y_i - y_j| <= r_i + r_j + alert

This is the exact predicate the kernel uses (inflate one AABB, test against the other).
"""

import sys
import numpy as np
from lbvh_wrapper import LBVHTree


# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------

def naive_pairs(x, y, r, alert):
    """Return a set of frozensets {i, j} for all overlapping pairs."""
    n = len(x)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (abs(x[i] - x[j]) <= r[i] + r[j] + alert and
                    abs(y[i] - y[j]) <= r[i] + r[j] + alert):
                pairs.add(frozenset((i, j)))
    return pairs


def lbvh_pairs(tree, x, y, r, alert):
    """Build the tree and return a set of frozensets {i, j}."""
    tree.build(x, y, r, alert)
    raw = tree.query()                      # shape (k, 2)
    return {frozenset(row) for row in raw}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def check_no_self_pairs(raw, label):
    bad = [(a, b) for a, b in raw if a == b]
    if bad:
        print(f"  FAIL [{label}] self-pairs: {bad}")
        return len(bad)
    return 0


def check_no_duplicates(raw, label):
    seen = set()
    dups = []
    for row in raw:
        key = frozenset(row)
        if key in seen:
            dups.append(tuple(row))
        seen.add(key)
    if dups:
        print(f"  FAIL [{label}] duplicate pairs: {dups}")
        return len(dups)
    return 0


def check_no_false_negatives(lbvh_set, naive_set, label):
    missing = naive_set - lbvh_set
    if missing:
        for p in missing:
            i, j = sorted(p)
            print(f"  FAIL [{label}] missing pair ({i}, {j})")
        return len(missing)
    return 0


def run_scenario(label, x, y, r, alert, max_pairs=None):
    """
    Run one scenario. Returns number of failures (0 = pass).
    max_pairs defaults to n*(n-1)//2 + 1 to avoid buffer overflow in tests.
    """
    n = len(x)
    if max_pairs is None:
        max_pairs = max(n * (n - 1) // 2 + 1, 16)

    tree = LBVHTree(n, max_pairs)
    tree.build(x, y, r, alert)
    raw_pairs = tree.query()    # ndarray (k, 2)

    ref   = naive_pairs(x, y, r, alert)
    found = {frozenset(row) for row in raw_pairs}

    fail  = 0
    fail += check_no_self_pairs(raw_pairs, label)
    fail += check_no_duplicates(raw_pairs, label)
    fail += check_no_false_negatives(found, ref, label)

    status = "PASS" if fail == 0 else "FAIL"
    print(f"  {status}  n={n:4d}  alert={alert:.3f}  "
          f"naive={len(ref):4d} pairs  lbvh={len(found):4d} pairs  [{label}]")
    return fail


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_uniform_random(rng, n, alert, label):
    """Particles uniformly distributed in [0, 10]², equal radii."""
    x = rng.uniform(0, 10, n).astype(np.float32)
    y = rng.uniform(0, 10, n).astype(np.float32)
    r = np.full(n, 0.5, dtype=np.float32)
    return run_scenario(label, x, y, r, alert)


def test_variable_radii(rng, n, alert, label):
    """Particles with heterogeneous radii in [0.1, 1.5]."""
    x = rng.uniform(0, 20, n).astype(np.float32)
    y = rng.uniform(0, 20, n).astype(np.float32)
    r = rng.uniform(0.1, 1.5, n).astype(np.float32)
    return run_scenario(label, x, y, r, alert)


def test_dense_cluster(rng, n, alert, label):
    """
    All particles within a tiny region → many duplicate or near-duplicate
    Morton codes.  Stresses the delta() fallback (32 + clz(i ^ j)).
    """
    x = rng.uniform(0, 0.1, n).astype(np.float32)
    y = rng.uniform(0, 0.1, n).astype(np.float32)
    r = np.full(n, 1.0, dtype=np.float32)   # all overlap every other
    return run_scenario(label, x, y, r, alert)


def test_sparse(rng, n, alert, label):
    """Particles far apart — almost no contacts, tree mostly empty."""
    x = (np.arange(n) * 10.0).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    r = np.full(n, 0.4, dtype=np.float32)
    return run_scenario(label, x, y, r, alert)


def test_grid(n_side, alert, label):
    """
    Regular grid of n_side × n_side particles with r=0.6.
    Each particle overlaps its 4 axis-aligned neighbours.
    Predictable structure exercises the tree's spatial locality.
    """
    coords = np.arange(n_side, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    x = gx.ravel()
    y = gy.ravel()
    r = np.full(len(x), 0.6, dtype=np.float32)
    return run_scenario(label, x, y, r, alert)


def test_collinear(n, alert, label):
    """All particles on y=0 — degenerate scene AABB, tests epsilon guard."""
    x = np.linspace(0, n - 1, n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    r = np.full(n, 0.6, dtype=np.float32)
    return run_scenario(label, x, y, r, alert)


def test_two_clusters(rng, n_each, gap, alert, label):
    """
    Two well-separated clusters. No cross-cluster pairs should appear
    (unless alert is large enough to bridge the gap).
    """
    x1 = rng.uniform(0,  2, n_each).astype(np.float32)
    y1 = rng.uniform(0,  2, n_each).astype(np.float32)
    x2 = rng.uniform(gap, gap + 2, n_each).astype(np.float32)
    y2 = rng.uniform(0,  2, n_each).astype(np.float32)
    x  = np.concatenate([x1, x2])
    y  = np.concatenate([y1, y2])
    r  = np.full(2 * n_each, 0.5, dtype=np.float32)
    return run_scenario(label, x, y, r, alert)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_fail = 0
    seeds = [0, 42, 137, 999, 31415]

    print("=" * 65)
    print("Uniform random — sparse (n=50, alert=0.2)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_uniform_random(np.random.default_rng(s), 50,  0.2,
                                          f"seed={s}")

    print()
    print("=" * 65)
    print("Uniform random — medium density (n=200, alert=0.3)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_uniform_random(np.random.default_rng(s), 200, 0.3,
                                          f"seed={s}")

    print()
    print("=" * 65)
    print("Uniform random — large (n=1000, alert=0.2)")
    print("=" * 65)
    for s in seeds[:3]:
        total_fail += test_uniform_random(np.random.default_rng(s), 1000, 0.2,
                                          f"seed={s}")

    print()
    print("=" * 65)
    print("Variable radii (n=100, alert=0.1)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_variable_radii(np.random.default_rng(s), 100, 0.1,
                                          f"seed={s}")

    print()
    print("=" * 65)
    print("Dense cluster — duplicate Morton codes (n=30, alert=0)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_dense_cluster(np.random.default_rng(s), 30, 0.0,
                                         f"seed={s}")

    print()
    print("=" * 65)
    print("Sparse grid — almost no contacts (n=60, alert=0)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_sparse(np.random.default_rng(s), 60, 0.0,
                                  f"seed={s}")

    print()
    print("=" * 65)
    print("Regular grid 8×8 — known neighbourhood (alert=0.1)")
    print("=" * 65)
    total_fail += test_grid(8, 0.1, "8x8 grid")
    total_fail += test_grid(10, 0.05, "10x10 grid")

    print()
    print("=" * 65)
    print("Collinear — degenerate y-extent (n=20, alert=0.1)")
    print("=" * 65)
    total_fail += test_collinear(20, 0.1, "collinear n=20")
    total_fail += test_collinear(50, 0.0, "collinear n=50 alert=0")

    print()
    print("=" * 65)
    print("Two clusters — no cross-cluster pairs (gap=20, alert=0.3)")
    print("=" * 65)
    for s in seeds:
        total_fail += test_two_clusters(np.random.default_rng(s),
                                        40, 20.0, 0.3, f"seed={s}")

    print()
    print("=" * 65)
    print("Alert sensitivity — pairs appear only above threshold")
    print("=" * 65)
    # Two particles 1.0 apart, r=0.4 each → gap=0.2; threshold alert=0.2
    for alert in [0.0, 0.1, 0.19, 0.20, 0.21, 0.5]:
        x = np.array([0.0, 1.0], dtype=np.float32)
        y = np.array([0.0, 0.0], dtype=np.float32)
        r = np.array([0.4, 0.4], dtype=np.float32)
        total_fail += run_scenario(f"alert={alert:.2f}", x, y, r,
                                   float(alert), max_pairs=4)

    print()
    print("=" * 65)
    result = "ALL TESTS PASSED" if total_fail == 0 else f"{total_fail} FAILURE(S)"
    print(f"  {result}")
    print("=" * 65)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
