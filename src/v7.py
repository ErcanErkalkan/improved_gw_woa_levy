"""
Benchmark script for Renewable Storage Scheduling with timing & evaluation accounting.

What this script produces (all reproducible):
- average_convergence.png                         : mean ± std convergence by iterations
- soc_group_<GroupName>.png                       : grouped SOC plots
- population_sensitivity.png                      : population size sensitivity (EGW-WOA)
- time_budget.csv                                 : per-method timing & evaluation stats (100 runs)
- time_budget_table.tex                           : LaTeX table snippet for timing/eval stats
- time_convergence_<METHOD>.csv (16 files)        : mean best-so-far vs time (±95% CI) grids
- time_quality_convergence.png                    : time–quality (wall-clock) plot (mean ± 95% CI)

Notes:
- All comments and outputs are in English, as requested.
- The proposed method is labeled "EGW-WOA" below and maps to .run_gwwoa().
"""

import time
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma  # kept if needed by external algorithms
from _algorithms import (
    GA, GWWOA, WOA, HS, FPA, PSO, GWO, CSO, HHO,
    BFO, FSS, MFO, MayflyAlgorithm, PFA, HOA, APO, TTA, CPSO
)

# ---------------------- Renewable Optimizer ----------------------
class RenewableOptimizer:
    def __init__(self, hours=24, population=30, max_iter=50, levy=0.05, chaos=0.1):
        self.hours = hours
        self.pop_size = population
        self.max_iter = max_iter
        self.levy_prob = levy
        self.chaos_prob = chaos
        # Parameters
        self.capital_cost = 500
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.SOC_min, self.SOC_max = 0.1, 0.9
        self.data_variation = 0.15
        self.SOC_capacity_decay = [1.0 - 0.005 * i for i in range(hours)]

        # --- Timing & evaluation accounting (reset per run) ---
        self.eval_count = 0
        self.t0 = None
        self.eval_log = []  # list of tuples (t_seconds_since_run_start, cost)

    def reset_run_counters(self):
        """Reset counters at the start of each algorithm run."""
        self.eval_count = 0
        self.eval_log = []
        self.t0 = time.perf_counter()

    def load_data(self, seed):
        np.random.seed(seed)
        t = np.arange(self.hours)
        base_solar = np.maximum(50 * np.sin(np.pi * (t - 6) / 12), 0)
        base_wind  = np.maximum(30 * np.cos(np.pi * (t - 12) / 6), 0)
        base_demand = 80 + 30 * np.sin(np.pi * (t + 6) / 12)
        base_price  = 0.15 + 0.05 * np.sin(np.pi * (t - 8) / 12) + 0.05
        var = 1 + self.data_variation * np.random.randn(self.hours)

        # Availability/failure/emergency flags
        self.solar_fail = np.random.rand(self.hours) < 0.1
        self.wind_fail  = np.random.rand(self.hours) < 0.15
        self.grid_ok    = np.random.rand(self.hours) < 0.8
        self.emergency  = np.random.rand(self.hours) < 0.1

        # Realizations
        self.P_solar   = base_solar * var * self.solar_fail
        self.P_wind    = base_wind  * var * self.wind_fail
        self.P_gen     = self.P_solar + self.P_wind
        self.P_demand  = base_demand * var * np.random.normal(1, 0.15, self.hours)

        self.grid_price = np.clip(base_price * var, 0.05, None)
        spikes = np.random.choice(self.hours, 4, replace=False)
        self.grid_price[spikes] *= np.random.uniform(3, 5, size=4)

    def energy_cost(self, x):
        # Accounting: count this evaluation and log time/cost later
        self.eval_count += 1

        S, u = x[0], x[1:]
        cost = self.capital_cost * S
        SOC = 0.5
        for t in range(self.hours):
            eff_S = S * self.SOC_capacity_decay[t]
            P_bess = u[t] * eff_S
            P_grid = self.P_demand[t] - self.P_gen[t] - P_bess

            # Emergency shortage penalty
            if self.emergency[t]:
                req = self.P_demand[t] * 1.5
                short = max(0, req - (self.P_gen[t] + P_bess))
                cost += short * 1e3

            # Grid unavailability penalty when importing
            if not self.grid_ok[t] and P_grid > 0:
                cost += 1e6

            # Energy & carbon cost when importing
            if P_grid > 0:
                cost += P_grid * self.grid_price[t] + P_grid * 0.487 * 2

            # Degradation proxy
            cost += 0.02 * abs(P_bess) ** 1.5 * (1 + SOC / 0.9)

            # Thermal proxy (simplified)
            deltaT = abs(P_bess / eff_S) / 0.05 if eff_S != 0 else 0
            if deltaT > 0:
                cost += max(0, deltaT - 20) ** 2 * 10

            # SOC update
            if P_bess > 0:
                dSOC = (P_bess * self.charge_eff) / eff_S if eff_S != 0 else 0
            else:
                dSOC = (P_bess / self.discharge_eff) / eff_S if eff_S != 0 else 0
            SOC = np.clip(SOC + dSOC, self.SOC_min, self.SOC_max)

        # Efficiency requirement penalty
        eff = (self.P_gen.sum() + (u * S).sum()) / self.P_demand.sum()
        if eff < 0.85:
            cost += (0.85 - eff) * 1e4

        # Time-cost logging for time–quality curves
        if self.t0 is not None:
            t_rel = time.perf_counter() - self.t0
            self.eval_log.append((t_rel, float(cost)))

        return cost

    # Wrapper methods (assume each algorithm class exposes .optimize() -> (best_solution, history))
    def run_ga(self): return GA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_gwo(self): return GWO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_gwwoa(self): return GWWOA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter, self.levy_prob, self.chaos_prob, 1.5).optimize()
    def run_woa(self): return WOA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_hs(self): return HS(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_fpa(self): return FPA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_pso(self): return PSO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_cso(self): return CSO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_hho(self): return HHO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_bfo(self): return BFO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_fss(self): return FSS(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_mfo(self): return MFO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_mayfly(self): return MayflyAlgorithm(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_pfa(self): return PFA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_hoa(self): return HOA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_apo(self): return APO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_tta(self): return TTA(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, self.pop_size, self.max_iter).optimize()
    def run_cpso(self): return CPSO(self.energy_cost, 25, [(1, 500)] + [(-0.5, 0.5)] * 24, [lambda x: x[0] > 0], self.pop_size, self.max_iter).optimize()


# ---------------------- Utility Functions ----------------------
def calculate_soc(x, hours=24):
    """Reconstruct SOC trajectory from the decision vector."""
    S, u = x[0], x[1:]
    soc = np.zeros(hours)
    soc[0] = 0.5
    for t in range(1, hours):
        Pb = u[t] * S
        if Pb > 0:
            d = (Pb * 0.95) / S if S != 0 else 0
        else:
            d = (Pb / 0.95) / S if S != 0 else 0
        soc[t] = np.clip(soc[t - 1] + d, 0.1, 0.9)
    return soc


# ---------------------- Plotting ----------------------
def plot_mean_convergence(results, analysis):
    """Plot average convergence by iterations (mean ± std) across successful runs."""
    plt.figure(figsize=(10, 6))
    for algo, data in results.items():
        if analysis[algo]['success_rate'] > 0:
            if not data['histories']:
                continue
            lengths = [len(h) for h in data['histories'] if len(h) > 0]
            if not lengths:
                continue
            maxlen = max(lengths)
            padded = []
            for h in data['histories']:
                if len(h) == 0:
                    continue
                # pad with final value to align sequences
                padded.append(h + [h[-1]] * (maxlen - len(h)))
            if not padded:
                continue
            M = np.array(padded, dtype=float)
            m = np.mean(M, axis=0)
            s = np.std(M, axis=0)
            plt.plot(m, label=algo)
            plt.fill_between(range(maxlen), m - s, m + s, alpha=0.2)
    plt.title('Average Convergence (100 trials)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_convergence.png', dpi=200)


def plot_soc_groups(results, groups):
    """Plot grouped SOC profiles using the last available solution per algorithm."""
    for name, algos in groups.items():
        plt.figure(figsize=(10, 4 * len(algos)))
        for i, algo in enumerate(algos, 1):
            plt.subplot(len(algos), 1, i)
            sols = [s for s in results[algo]['solutions'] if s is not None]
            if sols:
                soc = calculate_soc(sols[-1])
                plt.plot(soc, linewidth=1.5)
                plt.fill_between(range(len(soc)), soc, alpha=0.2)
            plt.title(algo)
            plt.ylim(0, 1)
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'soc_group_{name}.png', dpi=200)


def plot_population_sensitivity(best_levy, best_chaos):
    """Quick population-size sensitivity for EGW-WOA under a fixed seed."""
    pops = [20, 30, 50, 70]
    costs = []
    for p in pops:
        opt = RenewableOptimizer(population=p, levy=best_levy, chaos=best_chaos)
        opt.load_data(0)
        opt.reset_run_counters()
        _, h = opt.run_gwwoa()
        costs.append(h[-1] if len(h) else np.nan)
    plt.figure(figsize=(8, 5))
    plt.plot(pops, costs, '-o')
    plt.xlabel('Population Size')
    plt.ylabel('Final Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('population_sensitivity.png', dpi=200)


# ---------------------- Execution ----------------------
if __name__ == '__main__':
    best_levy, best_chaos = 0.05, 0.1

    # Map methods to their run_* wrappers.
    # NOTE: The proposed method is labeled "EGW-WOA" for consistency with the manuscript.
    algos = {
        'GA': 'run_ga',
        'GWO': 'run_gwo',
        'EGW-WOA': 'run_gwwoa',
        'WOA': 'run_woa',
        'HS': 'run_hs',
        'FPA': 'run_fpa',
        'PSO': 'run_pso',
        'CSO': 'run_cso',
        'BFO': 'run_bfo',
        'FSS': 'run_fss',
        'Mayfly': 'run_mayfly',
        'PFA': 'run_pfa',
        'HOA': 'run_hoa',
        'APO': 'run_apo',
        'TTA': 'run_tta',
        'CPSO': 'run_cpso'
    }

    # Collect raw outcomes
    results = {a: {'costs': [], 'histories': [], 'solutions': []} for a in algos}

    # Timing & evaluation accounting
    from collections import defaultdict
    run_times = defaultdict(list)              # seconds per run
    run_evals = defaultdict(list)              # evaluation count per run
    time_curves_by_trial = defaultdict(list)   # list of arrays per trial: [ [t, best_so_far], ... ]

    # Use 100 paired trials with shared seeds (100..199 for clarity)
    trial_seeds = list(range(100, 200))

    for t_seed in trial_seeds:
        opt = RenewableOptimizer(levy=best_levy, chaos=best_chaos)
        opt.load_data(t_seed)
        for name, fn in algos.items():
            try:
                opt.reset_run_counters()
                t_start = time.perf_counter()
                sol, h = getattr(opt, fn)()
                elapsed = time.perf_counter() - t_start

                results[name]['solutions'].append(sol)
                results[name]['histories'].append(h)
                results[name]['costs'].append(min(h) if len(h) else np.inf)

                run_times[name].append(elapsed)
                run_evals[name].append(opt.eval_count)

                # Build best-so-far vs time from evaluation log
                if opt.eval_log:
                    log_arr = np.array(opt.eval_log, dtype=float)
                    t_vec = log_arr[:, 0]
                    c_vec = log_arr[:, 1]
                    best_so_far = np.minimum.accumulate(c_vec)
                    time_curves_by_trial[name].append(np.column_stack([t_vec, best_so_far]))
                else:
                    time_curves_by_trial[name].append(np.zeros((0, 2)))
            except Exception as e:
                # Robust handling: record failure but keep the pipeline going
                results[name]['solutions'].append(None)
                results[name]['histories'].append([])
                results[name]['costs'].append(np.inf)
                run_times[name].append(np.nan)
                run_evals[name].append(0)
                time_curves_by_trial[name].append(np.zeros((0, 2)))

    # --- Aggregate statistics over runs ---
    analysis = {
        a: {
            'mean': np.nanmean(r['costs']),
            'std': np.nanstd(r['costs']),
            'min': np.nanmin(r['costs']),
            'max': np.nanmax(r['costs']),
            'success_rate': float(np.isfinite(r['costs']).mean())
        }
        for a, r in results.items()
    }

    print('Final Performance Summary (mean±std, min, max, success %):')
    for a, v in analysis.items():
        print(f"{a}: mean={v['mean']:.1f}, std={v['std']:.1f}, "
              f"min={v['min']:.1f}, max={v['max']:.1f}, success={v['success_rate']:.0%}")

    # --- (A) Time/Evaluation summary table -> CSV ---
    with open('time_budget.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'method', 'n_runs',
            'mean_time_s', 'std_time_s',
            'mean_evals', 'std_evals', 'total_evals',
            's_per_eval_mean', 'evals_per_s_mean'
        ])
        for a in algos:
            times = np.array(run_times[a], dtype=float)
            evals = np.array(run_evals[a], dtype=float)
            mask = np.isfinite(times) & (evals > 0)
            if mask.sum() == 0:
                w.writerow([a, 0, '', '', '', '', '', '', ''])
                continue
            times = times[mask]
            evals = evals[mask]
            mean_t = times.mean()
            std_t = times.std()
            mean_e = evals.mean()
            std_e = evals.std()
            total_e = int(evals.sum())
            s_per_eval = mean_t / mean_e
            evals_per_s = mean_e / mean_t
            w.writerow([
                a, int(mask.sum()),
                f"{mean_t:.6f}", f"{std_t:.6f}",
                f"{mean_e:.1f}", f"{std_e:.1f}", total_e,
                f"{s_per_eval:.6e}", f"{evals_per_s:.3f}"
            ])

    # --- (B) Time–quality curves (mean ± 95% CI) and figure ---
    plt.figure(figsize=(10, 6))
    for a in algos:
        curves = time_curves_by_trial[a]
        # Find max wall time to define a grid
        tmax = 0.0
        for c in curves:
            if c.shape[0]:
                tmax = max(tmax, float(c[-1, 0]))
        if tmax == 0.0:
            continue

        grid = np.linspace(0.0, tmax, 120)
        mat = []
        for c in curves:
            if c.shape[0] == 0:
                continue
            t = c[:, 0]
            y = c[:, 1]
            # Piecewise-constant interpolation then enforce monotone non-increasing best-so-far
            yi = np.interp(grid, t, y, left=y[0], right=y[-1])
            yi = np.minimum.accumulate(yi)
            mat.append(yi)
        if not mat:
            continue
        M = np.vstack(mat)
        m = M.mean(axis=0)
        s = M.std(axis=0) / math.sqrt(M.shape[0]) * 1.96  # ~95% CI (normal approx.)

        plt.plot(grid, m, label=a)
        plt.fill_between(grid, m - s, m + s, alpha=0.15)

        # Save per-method CSV for Supplement
        out = np.column_stack([grid, m, np.maximum(0, m - s), m + s])
        np.savetxt(
            f"time_convergence_{a}.csv",
            out, delimiter=",",
            header="t_s,mean_best,ci_lo,ci_hi",
            comments="",
            fmt="%.6f"
        )

    plt.title('Best-so-far cost vs. wall-clock time (mean ± 95% CI)')
    plt.xlabel('Time (s)')
    plt.ylabel('Cost ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_quality_convergence.png', dpi=200)

    # --- (C) LaTeX snippet for timing/evaluation table (booktabs/float friendly) ---
    df = pd.read_csv('time_budget.csv')
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Computation budget and timing summary per method (100 runs).}")
    lines.append("\\label{tab:time_budget}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Method & $n$ & Mean time (s) & Std (s) & Mean eval & Total eval & s/eval & eval/s\\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        if str(r['n_runs']).strip() == '' or int(r['n_runs']) == 0:
            continue
        lines.append(
            f"{r['method']} & {int(r['n_runs'])} & "
            f"{float(r['mean_time_s']):.3f} & {float(r['std_time_s']):.3f} & "
            f"{float(r['mean_evals']):.1f} & {int(r['total_evals'])} & "
            f"{float(r['s_per_eval_mean']):.3e} & {float(r['evals_per_s_mean']):.2f}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")
    with open('time_budget_table.tex', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # --- Existing figures ---
    plot_mean_convergence(results, analysis)
    groups = {
        'Group1': ['GA', 'GWO', 'EGW-WOA', 'WOA'],
        'Group2': ['HS', 'FPA', 'PSO', 'CSO'],
        'Group3': ['BFO', 'FSS', 'Mayfly', 'PFA'],
        'Group4': ['HOA', 'APO', 'TTA', 'CPSO']
    }
    plot_soc_groups(results, groups)
    plot_population_sensitivity(best_levy, best_chaos)

    print("\nArtifacts generated:")
    print(" - average_convergence.png")
    print(" - soc_group_Group1.png, soc_group_Group2.png, soc_group_Group3.png, soc_group_Group4.png")
    print(" - population_sensitivity.png")
    print(" - time_budget.csv, time_budget_table.tex")
    print(" - time_convergence_<METHOD>.csv for each method")
    print(" - time_quality_convergence.png")
