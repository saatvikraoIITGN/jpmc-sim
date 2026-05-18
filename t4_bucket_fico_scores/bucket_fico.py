import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "Task 3 and 4_Loan_Data.csv"
NUM_BUCKETS = 10
MIN_BUCKET_RECORDS = 100


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.sort_values('fico_score').reset_index(drop=True)
    return df


def aggregate_by_fico(df):
    """
    Group data by unique FICO score to reduce DP state space from ~10k rows
    to ~550 unique scores.
    """
    grouped = df.groupby('fico_score')['default'].agg(['sum', 'count']).reset_index()
    grouped = grouped.sort_values('fico_score').reset_index(drop=True)
    scores = grouped['fico_score'].values.astype(int)
    counts = grouped['count'].values.astype(int)
    defaults = grouped['sum'].values.astype(int)
    return scores, counts, defaults


# ---------------------------------------------------------------------------
# MSE-based dynamic programming quantization
# ---------------------------------------------------------------------------

def dp_mse_quantize(scores, counts, defaults, num_buckets=NUM_BUCKETS,
                    min_records=MIN_BUCKET_RECORDS):
    """
    Find bucket boundaries minimizing total MSE.

    For a bucket with n records and k defaults (default rate p = k/n):
        MSE contribution = sum (y_i - p)^2 = k(1-p)^2 + (n-k)p^2 = k - k^2/n

    A minimum-records constraint avoids degenerate tiny buckets.
    """
    m = len(scores)
    r = num_buckets

    pre_n = np.zeros(m + 1, dtype=np.float64)
    pre_k = np.zeros(m + 1, dtype=np.float64)
    for i in range(m):
        pre_n[i + 1] = pre_n[i] + counts[i]
        pre_k[i + 1] = pre_k[i] + defaults[i]

    INF = float('inf')

    def range_cost(i, j):
        n = pre_n[j + 1] - pre_n[i]
        k = pre_k[j + 1] - pre_k[i]
        if n < min_records:
            return INF
        return k - k * k / n

    dp = np.full((r + 1, m + 1), INF)
    split = np.zeros((r + 1, m + 1), dtype=int)
    dp[0][0] = 0.0

    for b in range(1, r + 1):
        for i in range(b, m + 1):
            for j in range(b - 1, i):
                cost = dp[b - 1][j] + range_cost(j, i - 1)
                if cost < dp[b][i]:
                    dp[b][i] = cost
                    split[b][i] = j

    boundaries = []
    i = m
    for b in range(r, 0, -1):
        j = split[b][i]
        if b > 1:
            boundaries.append(scores[j])
        i = j
    boundaries.sort()
    return boundaries, dp[r][m]


# ---------------------------------------------------------------------------
# Log-likelihood-based dynamic programming quantization
# ---------------------------------------------------------------------------

def dp_ll_quantize(scores, counts, defaults, num_buckets=NUM_BUCKETS,
                   min_records=MIN_BUCKET_RECORDS):
    """
    Find bucket boundaries maximizing:
        LL = sum_i [ k_i * ln(p_i) + (n_i - k_i) * ln(1 - p_i) ]
    where p_i = k_i / n_i.

    We minimize the negated LL for a uniform DP formulation.
    """
    m = len(scores)
    r = num_buckets
    EPS = 1e-10

    pre_n = np.zeros(m + 1, dtype=np.float64)
    pre_k = np.zeros(m + 1, dtype=np.float64)
    for i in range(m):
        pre_n[i + 1] = pre_n[i] + counts[i]
        pre_k[i + 1] = pre_k[i] + defaults[i]

    INF = float('inf')

    def range_cost(i, j):
        n = pre_n[j + 1] - pre_n[i]
        k = pre_k[j + 1] - pre_k[i]
        if n < min_records:
            return INF
        p = k / n
        p = np.clip(p, EPS, 1 - EPS)
        ll = k * np.log(p) + (n - k) * np.log(1 - p)
        return -ll

    dp = np.full((r + 1, m + 1), INF)
    split = np.zeros((r + 1, m + 1), dtype=int)
    dp[0][0] = 0.0

    for b in range(1, r + 1):
        for i in range(b, m + 1):
            for j in range(b - 1, i):
                cost = dp[b - 1][j] + range_cost(j, i - 1)
                if cost < dp[b][i]:
                    dp[b][i] = cost
                    split[b][i] = j

    boundaries = []
    i = m
    for b in range(r, 0, -1):
        j = split[b][i]
        if b > 1:
            boundaries.append(scores[j])
        i = j
    boundaries.sort()
    return boundaries, -dp[r][m]


# ---------------------------------------------------------------------------
# Rating map construction
# ---------------------------------------------------------------------------

def build_rating_map(scores, counts, defaults, boundaries, method_name):
    """
    Build a rating map from bucket boundaries.
    Rating 1 = best credit (lowest default rate), higher rating = worse.
    """
    all_bounds_idx = [0]
    for b in boundaries:
        idx = int(np.searchsorted(scores, b))
        all_bounds_idx.append(idx)
    all_bounds_idx.append(len(scores))

    buckets = []
    for i in range(len(all_bounds_idx) - 1):
        lo_idx = all_bounds_idx[i]
        hi_idx = all_bounds_idx[i + 1]
        if lo_idx >= hi_idx:
            continue
        n = int(counts[lo_idx:hi_idx].sum())
        k = int(defaults[lo_idx:hi_idx].sum())
        pd_rate = k / n if n > 0 else 0.0
        buckets.append({
            'range_low': int(scores[lo_idx]),
            'range_high': int(scores[hi_idx - 1]),
            'n_records': n,
            'n_defaults': k,
            'default_rate': pd_rate,
        })

    # Rating 1 = lowest PD (safest), NUM_BUCKETS = highest PD (riskiest)
    sorted_by_pd = sorted(range(len(buckets)), key=lambda i: buckets[i]['default_rate'])
    for rank, idx in enumerate(sorted_by_pd, 1):
        buckets[idx]['rating'] = rank

    print(f"\n{'=' * 78}")
    print(f"  RATING MAP — {method_name}")
    print(f"{'=' * 78}")
    print(f"  {'Rating':>6}  {'FICO Range':>15}  {'Records':>8}  {'Defaults':>9}  {'PD Rate':>10}")
    print(f"  {'-' * 6}  {'-' * 15}  {'-' * 8}  {'-' * 9}  {'-' * 10}")
    for b in buckets:
        print(f"  {b['rating']:>6}  {b['range_low']:>6} - {b['range_high']:<6}  "
              f"{b['n_records']:>8}  {b['n_defaults']:>9}  {b['default_rate']:>9.4f}")

    return buckets


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_analysis(df, mse_boundaries, ll_boundaries, mse_buckets, ll_buckets):
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    fico = df['fico_score'].values
    default = df['default'].values

    # --- Row 0: FICO score distribution with boundaries ---
    for col, boundaries, title in [
        (0, mse_boundaries, 'MSE Optimization'),
        (1, ll_boundaries, 'Log-Likelihood Optimization'),
    ]:
        ax = fig.add_subplot(gs[0, col])
        ax.hist(fico, bins=80, alpha=0.7, color='#3498db', edgecolor='white')
        for b in boundaries:
            ax.axvline(x=b, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('FICO Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'FICO Distribution with Bucket Boundaries\n({title})')
        ax.grid(True, alpha=0.3)

    # --- Row 1: Default rate per bucket ---
    for col, buckets, title in [
        (0, mse_buckets, 'MSE Optimization'),
        (1, ll_buckets, 'Log-Likelihood Optimization'),
    ]:
        ax = fig.add_subplot(gs[1, col])
        labels = [f"{b['range_low']}-\n{b['range_high']}" for b in buckets]
        rates = [b['default_rate'] for b in buckets]
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(buckets)))
        bars = ax.bar(range(len(buckets)), rates, color=colors, edgecolor='white',
                      width=0.8)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Default Rate (PD)')
        ax.set_xlabel('FICO Score Range')
        ax.set_title(f'Default Rate by Bucket ({title})')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # --- Row 2: Rating map ---
    for col, buckets, title in [
        (0, mse_buckets, 'MSE Optimization'),
        (1, ll_buckets, 'Log-Likelihood Optimization'),
    ]:
        ax = fig.add_subplot(gs[2, col])
        labels = [f"{b['range_low']}-\n{b['range_high']}" for b in buckets]
        ratings = [b['rating'] for b in buckets]
        num_b = len(buckets)
        colors = plt.cm.RdYlGn(np.array([(num_b - r + 1) / num_b for r in ratings]))
        bars = ax.bar(range(len(buckets)), ratings, color=colors, edgecolor='white',
                      width=0.8)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Rating (1=Best, higher=Worse)')
        ax.set_xlabel('FICO Score Range')
        ax.set_title(f'Credit Rating by Bucket ({title})')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, rating in zip(bars, ratings):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.15,
                    str(rating), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Row 3: Side-by-side comparison (step plot) ---
    ax = fig.add_subplot(gs[3, :])
    for buckets, label, color in [
        (mse_buckets, 'MSE Buckets', '#e74c3c'),
        (ll_buckets, 'Log-Likelihood Buckets', '#2980b9'),
    ]:
        xs = [b['range_low'] for b in buckets] + [buckets[-1]['range_high']]
        ys = [b['default_rate'] for b in buckets] + [buckets[-1]['default_rate']]
        ax.step(xs, ys, where='post', label=label, color=color, linewidth=2)

    window = 10
    min_f, max_f = int(fico.min()), int(fico.max())
    raw_x, raw_y = [], []
    for lo in range(min_f, max_f, window):
        mask = (fico >= lo) & (fico < lo + window)
        if mask.sum() > 5:
            raw_x.append(lo + window / 2)
            raw_y.append(default[mask].mean())
    ax.scatter(raw_x, raw_y, s=14, alpha=0.4, color='gray',
               label=f'Empirical PD ({window}-pt window)')

    ax.set_xlabel('FICO Score')
    ax.set_ylabel('Default Rate (PD)')
    ax.set_title('Bucketed PD vs. Empirical Default Rate')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.savefig('fico_bucketing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nAll plots saved to fico_bucketing_analysis.png")


# ---------------------------------------------------------------------------
# FICO score → rating lookup
# ---------------------------------------------------------------------------

def create_fico_to_rating(buckets):
    """Return a function that maps any FICO score to its credit rating."""
    def lookup(score):
        for b in buckets:
            if b['range_low'] <= score <= b['range_high']:
                return b['rating'], b['default_rate']
        return None, None
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    fico = df['fico_score'].values
    defaults = df['default'].values

    print(f"Loaded {len(df)} records")
    print(f"FICO range: {fico.min()} — {fico.max()}")
    print(f"Overall default rate: {defaults.mean():.4f} "
          f"({defaults.sum()} defaults / {len(df)} total)")

    scores, counts, def_counts = aggregate_by_fico(df)
    print(f"Unique FICO scores: {len(scores)}  (DP runs on this reduced space)")
    print(f"Minimum bucket size: {MIN_BUCKET_RECORDS} records")
    print(f"\nFinding optimal {NUM_BUCKETS} buckets...\n")

    # --- MSE optimization ---
    print(">>> MSE dynamic programming...")
    mse_bounds, mse_cost = dp_mse_quantize(scores, counts, def_counts, NUM_BUCKETS)
    print(f"    Boundaries: {[int(b) for b in mse_bounds]}")
    print(f"    Total MSE:  {mse_cost:.4f}")
    mse_buckets = build_rating_map(scores, counts, def_counts, mse_bounds,
                                   "MSE Minimization")

    # --- Log-likelihood optimization ---
    print("\n>>> Log-Likelihood dynamic programming...")
    ll_bounds, ll_val = dp_ll_quantize(scores, counts, def_counts, NUM_BUCKETS)
    print(f"    Boundaries: {[int(b) for b in ll_bounds]}")
    print(f"    Log-Likelihood: {ll_val:.4f}")
    ll_buckets = build_rating_map(scores, counts, def_counts, ll_bounds,
                                  "Log-Likelihood Maximization")

    # --- Visualize ---
    plot_analysis(df, mse_bounds, ll_bounds, mse_buckets, ll_buckets)

    # --- Demo lookups ---
    ll_lookup = create_fico_to_rating(ll_buckets)
    mse_lookup = create_fico_to_rating(mse_buckets)

    print("\n" + "=" * 70)
    print("  EXAMPLE FICO → RATING LOOKUPS")
    print("=" * 70)
    header = (f"  {'FICO':>6}  {'MSE Rating':>11}  {'MSE PD':>10}  "
              f"{'LL Rating':>10}  {'LL PD':>10}")
    print(header)
    print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*10}")
    for score in [420, 480, 520, 560, 600, 640, 680, 720, 760, 800, 845]:
        r_mse, pd_mse = mse_lookup(score)
        r_ll, pd_ll = ll_lookup(score)
        fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
        fmt_r = lambda v: str(v) if v is not None else "N/A"
        print(f"  {score:>6}  {fmt_r(r_mse):>11}  {fmt(pd_mse):>10}  "
              f"{fmt_r(r_ll):>10}  {fmt(pd_ll):>10}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Both methods use dynamic programming to find globally optimal")
    print(f"  bucket boundaries for {NUM_BUCKETS} buckets over {len(scores)} unique FICO scores.")
    print(f"\n  MSE approach: Minimizes total squared error of approximating each")
    print(f"  borrower's default status (0/1) by the bucket's default rate.")
    print(f"  → Best for prediction accuracy when used as a regression target.")
    print(f"\n  Log-Likelihood approach: Maximizes the binomial log-likelihood of")
    print(f"  the observed defaults within each bucket.")
    print(f"  → Best for statistical modeling (captures default density + discretization).")
    print(f"\n  The LL method tends to produce more granular separation in the high-risk")
    print(f"  region (low FICO scores) where default rate changes rapidly, while MSE")
    print(f"  distributes buckets more evenly across the default rate spectrum.")

    return mse_buckets, ll_buckets


if __name__ == '__main__':
    mse_buckets, ll_buckets = main()
