"""
benchmark_lbvh.py — Wall-clock comparison of LBVH vs naive O(n²) contact detection.

For each particle count n, both methods are run REPEATS times and the median
wall-clock time is reported.  Three spatial distributions are tested:
  - uniform:  particles scattered uniformly in a box
  - clustered: particles grouped in a few dense clusters
  - grid:     particles on a regular lattice
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from lbvh_wrapper import LBVHTree


REPEATS   = 7       # runs per (n, distribution) combination (median taken)
ALERT     = 0.0     # pure AABB overlap, no inflation
MAX_PAIRS_FACTOR = 20  # max_pairs = n * MAX_PAIRS_FACTOR


# ---------------------------------------------------------------------------
# Naive O(n²) reference
# ---------------------------------------------------------------------------

def naive_contact_detection(x, y, r, alert):
    """Return list of (i,j) pairs whose inflated AABBs overlap."""
    pairs = []
    threshold = r[:, None] + r[None, :] + alert   # (n, n)
    dx = np.abs(x[:, None] - x[None, :])           # (n, n)
    dy = np.abs(y[:, None] - y[None, :])
    overlap = (dx <= threshold) & (dy <= threshold)
    ii, jj = np.where(overlap)
    for i, j in zip(ii, jj):
        if i < j:
            pairs.append((int(i), int(j)))
    return pairs


# ---------------------------------------------------------------------------
# Particle generators
# ---------------------------------------------------------------------------

def make_uniform(rng, n):
    side = np.sqrt(n)                          # box scales with n so density stays constant
    x = rng.uniform(0, side, n).astype(np.float32)
    y = rng.uniform(0, side, n).astype(np.float32)
    r = np.full(n, 0.4, dtype=np.float32)
    return x, y, r


def make_clustered(rng, n, n_clusters=8):
    centres_x = rng.uniform(2, 18, n_clusters)
    centres_y = rng.uniform(2, 18, n_clusters)
    x, y = [], []
    per_cluster = (n + n_clusters - 1) // n_clusters  # ceiling so total >= n
    for cx, cy in zip(centres_x, centres_y):
        x.append(rng.normal(cx, 0.5, per_cluster))
        y.append(rng.normal(cy, 0.5, per_cluster))
    x = np.concatenate(x)[:n].astype(np.float32)
    y = np.concatenate(y)[:n].astype(np.float32)
    r = np.full(n, 0.3, dtype=np.float32)
    return x, y, r


def make_grid(_, n):
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side, dtype=np.float32),
                          np.arange(side, dtype=np.float32))
    x = gx.ravel()[:n]
    y = gy.ravel()[:n]
    r = np.full(n, 0.4, dtype=np.float32)
    return x, y, r


DISTRIBUTIONS = {
    "uniform":   lambda rng, n: make_uniform(rng, n),
    "clustered": lambda rng, n: make_clustered(rng, n),
    "grid":      lambda rng, n: make_grid(rng, n),
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def benchmark_one(n, dist_name, dist_fn, rng):
    x, y, r = dist_fn(rng, n)
    max_pairs = n * MAX_PAIRS_FACTOR

    # --- LBVH (includes build + query) ---
    tree = LBVHTree(n, max_pairs)

    def run_lbvh():
        tree.build(x, y, r, ALERT)
        tree.query()

    t_lbvh = median_time(run_lbvh, REPEATS)

    # count pairs once for reporting
    tree.build(x, y, r, ALERT)
    n_pairs_lbvh = len(tree.query())

    # --- Naive ---
    def run_naive():
        naive_contact_detection(x, y, r, ALERT)

    t_naive = median_time(run_naive, REPEATS)
    n_pairs_naive = len(naive_contact_detection(x, y, r, ALERT))

    speedup = t_naive / t_lbvh if t_lbvh > 0 else float("inf")

    print(f"  n={n:6d}  {dist_name:10s}  "
          f"naive={t_naive*1e3:8.2f} ms  lbvh={t_lbvh*1e3:8.2f} ms  "
          f"speedup={speedup:6.2f}x  "
          f"pairs(naive/lbvh)={n_pairs_naive}/{n_pairs_lbvh}")

    return t_naive, t_lbvh, speedup, n_pairs_naive


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    ns = [50, 100, 250, 500, 1000, 2500, 5000, 10000]

    results = {name: {"ns": [], "t_naive": [], "t_lbvh": [], "speedup": []}
               for name in DISTRIBUTIONS}

    for dist_name, dist_fn in DISTRIBUTIONS.items():
        print()
        print("=" * 80)
        print(f"Distribution: {dist_name}")
        print("=" * 80)
        for n in ns:
            t_naive, t_lbvh, speedup, _ = benchmark_one(
                n, dist_name, dist_fn, rng)
            results[dist_name]["ns"].append(n)
            results[dist_name]["t_naive"].append(t_naive * 1e3)
            results[dist_name]["t_lbvh"].append(t_lbvh * 1e3)
            results[dist_name]["speedup"].append(speedup)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"uniform": "steelblue", "clustered": "tomato", "grid": "seagreen"}

    # Left: absolute times
    ax = axes[0]
    for dist_name, data in results.items():
        c = colors[dist_name]
        ax.plot(data["ns"], data["t_naive"], "--", color=c, alpha=0.6,
                label=f"naive {dist_name}")
        ax.plot(data["ns"], data["t_lbvh"],  "-o", color=c,
                label=f"LBVH {dist_name}")
    ax.set_xlabel("Number of particles n")
    ax.set_ylabel("Median wall-clock time (ms)")
    ax.set_title("Contact detection: LBVH vs naive")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # Right: speedup
    ax = axes[1]
    for dist_name, data in results.items():
        ax.plot(data["ns"], data["speedup"], "-o", color=colors[dist_name],
                label=dist_name)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="break-even")
    ax.set_xlabel("Number of particles n")
    ax.set_ylabel("Speedup  (naive / LBVH)")
    ax.set_title("LBVH speedup over naive")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/benchmark_lbvh.png", dpi=300)
    print("\nPlot saved to benchmark_lbvh.png")
    plt.show()


if __name__ == "__main__":
    main()
