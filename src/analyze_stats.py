import csv, math, itertools, random
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# ---- 0) Yükleme ----
# results_all.csv: columns = method,run,cost
data = defaultdict(list)  # method -> list of costs (run-sırası ile)
order = []                # method sırası (ilk görülen)
with open("outputs/results_all.csv", newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    # run'ları 1..N olarak varsayıyoruz; her method-run tekil satır
    tmp = defaultdict(dict)  # method -> {run: cost}
    for row in rdr:
        m = row["method"].strip()
        r = int(row["run"])
        c = float(row["cost"])
        tmp[m][r] = c
        if m not in order:
            order.append(m)
    # run indexlerini ortak sıraya sokalım
    runs = None
    for m in order:
        rkeys = sorted(tmp[m].keys())
        if runs is None:
            runs = rkeys
        else:
            # ortak kesişim al
            runs = [r for r in runs if r in rkeys]
    for m in order:
        data[m] = [tmp[m][r] for r in runs]

methods = order
k = len(methods)
n = len(next(iter(data.values())))  # blok sayısı (run sayısı)

# ---- 1) Özet istatistikler + bootstrap CI ----
def bootstrap_ci(x, stat_fn, B=10000, alpha=0.05, rng=np.random.default_rng(42)):
    x = np.asarray(x)
    stats = []
    N = len(x)
    for _ in range(B):
        idx = rng.integers(0, N, N)
        stats.append(stat_fn(x[idx]))
    lo = np.percentile(stats, 100*alpha/2)
    hi = np.percentile(stats, 100*(1-alpha/2))
    return lo, hi

summary_rows = []
for m in methods:
    arr = np.asarray(data[m], dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    med = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    mn = arr.min()
    mx = arr.max()
    mean_ci = bootstrap_ci(arr, np.mean, B=5000)
    med_ci  = bootstrap_ci(arr, np.median, B=5000)
    summary_rows.append({
        "method": m,
        "n_runs": n,
        "mean": mean,
        "std": std,
        "mean_ci_lo": mean_ci[0],
        "mean_ci_hi": mean_ci[1],
        "median": med,
        "median_ci_lo": med_ci[0],
        "median_ci_hi": med_ci[1],
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "min": mn,
        "max": mx
    })

with open("summary_stats.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

# ---- 2) Friedman + Kendall's W ----
# SciPy friedmanchisquare her bir yöntemi ayrı argüman ister:
arrays_for_friedman = [data[m] for m in methods]
chi2, p_fried = friedmanchisquare(*arrays_for_friedman)

# Kendall's W: W = chi2 / (n * (k - 1))
kendall_W = chi2 / (n * (k - 1))

with open("friedman_kendallW.txt", "w", encoding="utf-8") as f:
    f.write(f"Friedman chi2={chi2:.6f}, p={p_fried:.6g}\n")
    f.write(f"Kendall_W={kendall_W:.6f} (k={k}, n={n})\n")

# ---- 3) EGWWOA vs diğerleri: Wilcoxon (eşleşmiş), Holm düzeltmesi ----
target = "EGWWOA"  # sizin makronuzla uyumlu isim (CSV'de böyle yazdığınızdan emin olun)
assert target in methods, "CSV'de EGWWOA ismi bulunamadı."

pairs = []
raw_ps = []
wilc_rows = []

def rank_biserial_from_wilcoxon_stat(W_plus, n):
    # SciPy wilcoxon().statistic = "R+" (pozitif farkların rank toplamı)
    return (2.0 * W_plus / (n * (n + 1))) - 1.0  # r_rb in [-1,1]

for m in methods:
    if m == target: 
        continue
    x = np.asarray(data[target], dtype=float)
    y = np.asarray(data[m], dtype=float)
    # EGWWOA vs m: fark = x - y (işaretin yorumu)
    # SciPy 'wilcoxon' sıfır farkları otomatik atar; zero_method varsayılan 'wilcox'
    res = wilcoxon(x, y, alternative="two-sided")
    p = res.pvalue
    Wplus = res.statistic  # R+
    # n_eff: sıfır olmayan fark sayısı
    n_eff = int(np.sum(x != y))
    r_rb = rank_biserial_from_wilcoxon_stat(Wplus, n_eff) if n_eff > 0 else 0.0
    pairs.append(m)
    raw_ps.append(p)
    wilc_rows.append({
        "baseline": m,
        "n_pairs": n_eff,
        "wilcoxon_Wplus": float(Wplus),
        "p_raw": p,
        "rank_biserial_r": r_rb
    })

# Holm-Bonferroni düzeltmesi
mtests = len(raw_ps)
idx = np.argsort(raw_ps)
sorted_ps = np.array(raw_ps)[idx]
sorted_names = np.array(pairs)[idx]
adj = np.empty_like(sorted_ps)
# Holm: p_(i)^* = max_{j<=i} ( (m - j + 1) * p_(j) ), 1 ile sınırla
running_max = 0.0
for i, p in enumerate(sorted_ps, start=1):
    val = (mtests - i + 1) * p
    running_max = max(running_max, val)
    adj[i-1] = min(1.0, running_max)

# Orijinal sıraya geri koy
adj_map = {name: adj[i] for i, name in enumerate(sorted_names)}
for row in wilc_rows:
    row["p_holm"] = float(adj_map[row["baseline"]])

with open("pairwise_wilcoxon_vs_EGWWOA.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(wilc_rows[0].keys()))
    w.writeheader()
    w.writerows(wilc_rows)
