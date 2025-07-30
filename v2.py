import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma
# Import all metaheuristic classes
from algorithms import GA, GWWOA, WOA, HS, FPA, PSO, GWO, CSO, HHO, BFO, FSS, MFO
from algorithms import MayflyAlgorithm, PFA, HOA, APO, TTA, CPSO

class RenewableOptimizer:
    def __init__(self, hours=24, population=30, max_iter=50):
        self.hours = hours
        self.pop_size = population
        self.max_iter = max_iter
        # Problem parameters
        self.capital_cost = 500  # $/kWh
        self.c_deg = 0.02
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.SOC_min, self.SOC_max = 0.1, 0.9
        self.data_variation = 0.15  # 15% random variation
        self.temperature = 25
        self.SOC_capacity_decay = [1.0 - 0.005 * i for i in range(self.hours)]
        self.emergency_event = np.zeros(self.hours)

    def load_data(self, trial_num):
        np.random.seed(trial_num)
        t = np.arange(self.hours)
        # Base profiles
        base_solar = 50 * np.sin(np.pi * (t - 6) / 12) + 1
        base_wind = 30 * np.cos(np.pi * (t - 12) / 6) + 5
        base_demand = 80 + 30 * np.sin(np.pi * (t + 6) / 12)
        base_price = 0.15 + 0.05 * np.sin(np.pi * (t - 8) / 12) + 0.05
        # Random variation and failures
        variation = 1 + self.data_variation * np.random.randn(self.hours)
        self.solar_failure = np.random.choice([0,1], self.hours, p=[0.9,0.1])
        self.wind_failure = np.random.choice([0,1], self.hours, p=[0.85,0.15])
        self.grid_available = np.random.choice([0,1], self.hours, p=[0.2,0.8])
        self.emergency_event = np.random.choice([0,1], self.hours, p=[0.9,0.1])
        # Generation and demand
        self.P_solar = np.clip(base_solar * variation * self.solar_failure, 0, None)
        self.P_wind = np.clip(base_wind * variation * self.wind_failure, 0, None)
        self.P_gen = self.P_solar + self.P_wind
        self.P_demand = base_demand * variation * np.random.normal(1, 0.15, self.hours)
        # Pricing with spikes
        self.grid_price = np.clip(base_price * variation, 0.05, None)
        spike_hours = np.random.choice(self.hours, size=4, replace=False)
        self.grid_price[spike_hours] *= np.random.uniform(3, 5, size=4)

    def energy_cost(self, solution):
        S = solution[0]
        u = solution[1:]
        total_cost = self.capital_cost * S
        SOC = 0.5
        SOC_history = []
        self.temperature = 25
        for t_i in range(self.hours):
            effective_S = S * self.SOC_capacity_decay[t_i]
            P_bess = u[t_i] * effective_S
            P_grid = self.P_demand[t_i] - self.P_gen[t_i] - P_bess
            # Emergency event penalty
            if self.emergency_event[t_i]:
                req = self.P_demand[t_i] * 1.5
                shortage = max(0, req - (self.P_gen[t_i] + P_bess))
                total_cost += shortage * 1000
            # Grid availability violation
            if (not self.grid_available[t_i]) and P_grid > 0:
                total_cost += 1e6
            # Costs
            grid_cost = P_grid * self.grid_price[t_i] if P_grid > 0 else 0
            degradation = 0.02 * (abs(P_bess)**1.5) * (1 + SOC/0.9)
            carbon_cost = P_grid * 0.487 * 2
            total_cost += grid_cost + degradation + carbon_cost
            # Thermal penalty
            delta_temp = abs(P_bess / effective_S) / 0.05
            self.temperature += delta_temp
            if self.temperature > 45:
                total_cost += (self.temperature - 45)**2 * 10
            # Update SOC
            if P_bess < 0:
                delta = (-P_bess * self.charge_eff) / effective_S
            else:
                delta = -P_bess / (self.discharge_eff * effective_S)
            SOC += delta
            SOC = np.clip(SOC, self.SOC_min, self.SOC_max)
            SOC_history.append(SOC)
            # SOC smoothness penalty
            if t_i >= 3:
                avg_soc = np.mean(SOC_history[-3:])
                if abs(SOC - avg_soc) > 0.2:
                    total_cost += 1e4
        # Efficiency target
        efficiency = (self.P_gen.sum() + (u*S).sum()) / self.P_demand.sum()
        if efficiency < 0.85:
            total_cost += (0.85 - efficiency) * 1e4
        # Periodic smoothing cost
        total_cost += 100 * abs(np.sin(S * 0.01))
        return total_cost

# -----------------------------------------------------------------------------
# Analysis and Table Generation
# -----------------------------------------------------------------------------

def run_trials(optimizer_fn, config_kwargs, n_trials=100):
    costs = []
    for trial in range(n_trials):
        opt = RenewableOptimizer(population=config_kwargs.get('population_size',30), max_iter=config_kwargs.get('max_iter',50))
        opt.load_data(trial)
        best_x, history = optimizer_fn(opt, **config_kwargs)
        costs.append(history[-1] if history else np.inf)
    return np.array(costs)

# Wrapper functions for different variants

def optimize_gwo(opt, **kwargs):
    algo = GWO(
        obj_func=lambda x: opt.energy_cost(x), dim=25,
        bounds=[(1,2000)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter)
    return algo.optimize()

def optimize_gwwoa(opt, levy_prob, chaos_prob, **kwargs):
    algo = GWWOA(
        obj_func=lambda x: opt.energy_cost(x), dim=25,
        bounds=[(1,2000)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter,
        levy_prob=levy_prob, chaos_prob=chaos_prob, beta=1.5)
    return algo.optimize()

# Ablation Study
configs = {
    'GWO only': {'fn': optimize_gwo},
    'GWO + WOA': {'fn': optimize_gwwoa, 'levy_prob':0.0, 'chaos_prob':0.0},
    'GWO + WOA + Lévy': {'fn': optimize_gwwoa, 'levy_prob':0.1, 'chaos_prob':0.0},
    'Full hybrid (+ chaos)': {'fn': optimize_gwwoa, 'levy_prob':0.1, 'chaos_prob':0.1},
}
ablation_stats = {}
for name, cfg in configs.items():
    costs = run_trials(cfg['fn'], cfg)
    mean_cost = costs.mean()
    ablation_stats[name] = mean_cost
# Build DataFrame
ablation_df = pd.DataFrame.from_dict(ablation_stats, orient='index', columns=['Mean Cost'])
base = ablation_df.iloc[0,0]
ablation_df['% Change vs. base'] = ((ablation_df['Mean Cost'] - base)/base*100).round(2) # type: ignore

# Final performance summary
from itertools import zip_longest
# Running only GA and GWWOA here as example
algo_map = {
    'GA': lambda opt: GA(
        obj_func=lambda x: opt.energy_cost(x), dim=25,
        bounds=[(1,2000)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter,
        crossover_rate=0.8, mutation_rate=0.1).run(),
    'EGW-WOA': lambda opt: GWWOA(
        obj_func=lambda x: opt.energy_cost(x), dim=25,
        bounds=[(1,2000)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter,
        levy_prob=0.1, chaos_prob=0.1, beta=1.5).optimize(),
}
perf_stats = {}
for name, fn in algo_map.items():
    costs = []
    for trial in range(100):
        opt = RenewableOptimizer()
        opt.load_data(trial)
        _, hist = fn(opt)
        costs.append(hist[-1])
    arr = np.array(costs)
    perf_stats[name] = {
        'Mean': arr.mean(),
        'Std Dev': arr.std(),
        'Min': arr.min(),
        'Max': arr.max(),
        'Success Rate': np.isfinite(arr).mean()
    }
final_df = pd.DataFrame.from_dict(perf_stats, orient='index')

# Population sensitivity
pop_sizes = [20, 30, 50, 70]
pop_stats = {}
for pop in pop_sizes:
    costs = []
    for trial in range(100):
        opt = RenewableOptimizer(population=pop)
        opt.load_data(trial)
        _, hist = GWWOA(
            obj_func=lambda x: opt.energy_cost(x), dim=25,
            bounds=[(1,2000)]+[(-0.5,0.5)]*24,
            population_size=pop, max_iter=50,
            levy_prob=0.1, chaos_prob=0.1, beta=1.5).optimize()
        costs.append(hist[-1])
    pop_stats[pop] = np.mean(costs)
pop_df = pd.Series(pop_stats, name='Final Cost')

# Display results
print("\n=== Ablation Study ===")
print(ablation_df)
print("\n=== Final Performance Summary ===")
print(final_df)
print("\n=== Population Sensitivity ===")
print(pop_df)

# Save tables to CSV for manuscript inclusion
ablation_df.to_csv('ablation_results.csv')
final_df.to_csv('final_performance_summary.csv')
pop_df.to_csv('population_sensitivity.csv')

# (Optional) You can extend this script to plot and save figures automatically as well.
