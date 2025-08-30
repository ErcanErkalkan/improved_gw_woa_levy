import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms import GWO, GWWOA

class RenewableOptimizer:
    def __init__(self, hours=24, population=30, max_iter=50):
        self.hours = hours
        self.pop_size = population
        self.max_iter = max_iter
        self.capital_cost = 500
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.SOC_min, self.SOC_max = 0.1, 0.9
        self.data_variation = 0.15
        self.SOC_capacity_decay = [1.0 - 0.005 * i for i in range(self.hours)]

    def load_data(self, seed):
        np.random.seed(seed)
        t = np.arange(self.hours)
        base_solar = np.maximum(50 * np.sin(np.pi * (t - 6) / 12), 0)
        base_wind  = np.maximum(30 * np.cos(np.pi * (t - 12) / 6), 0)
        base_demand = 80 + 30 * np.sin(np.pi * (t + 6) / 12)
        base_price  = 0.15 + 0.05 * np.sin(np.pi * (t - 8) / 12) + 0.05
        var = 1 + self.data_variation * np.random.randn(self.hours)
        self.solar_fail = np.random.rand(self.hours) < 0.1
        self.wind_fail  = np.random.rand(self.hours) < 0.15
        self.grid_ok    = np.random.rand(self.hours) < 0.8
        self.emergency  = np.random.rand(self.hours) < 0.1
        self.P_solar   = base_solar * var * self.solar_fail
        self.P_wind    = base_wind  * var * self.wind_fail
        self.P_gen     = self.P_solar + self.P_wind
        self.P_demand  = base_demand * var * np.random.normal(1,0.15,self.hours)
        self.grid_price = np.clip(base_price * var, 0.05, None)
        spikes = np.random.choice(self.hours, 4, replace=False)
        self.grid_price[spikes] *= np.random.uniform(3,5, size=4)

    def energy_cost(self, x):
        S = x[0]
        u = x[1:]
        cost = self.capital_cost * S
        SOC = 0.5
        for t in range(self.hours):
            eff_S = S * self.SOC_capacity_decay[t]
            P_bess = u[t] * eff_S
            P_grid = self.P_demand[t] - self.P_gen[t] - P_bess
            if self.emergency[t]:
                req = self.P_demand[t] * 1.5
                short = max(0, req - (self.P_gen[t] + P_bess))
                cost += short * 1e3
            if not self.grid_ok[t] and P_grid > 0:
                cost += 1e6
            if P_grid > 0:
                cost += P_grid * self.grid_price[t]
                cost += P_grid * 0.487 * 2
            cost += 0.02 * abs(P_bess)**1.5 * (1 + SOC/0.9)
            deltaT = abs(P_bess/eff_S)/0.05
            if deltaT > 0:
                cost += max(0, (deltaT - 20))**2 * 10
            if P_bess > 0:
                dSOC = (P_bess * self.charge_eff) / eff_S
            else:
                dSOC = (P_bess / self.discharge_eff) / eff_S
            SOC = np.clip(SOC + dSOC, self.SOC_min, self.SOC_max)
        eff = (self.P_gen.sum() + (u*S).sum()) / self.P_demand.sum()
        if eff < 0.85:
            cost += (0.85 - eff)*1e4
        return cost

# Wrappers

def optimize_gwo(opt):
    algo = GWO(
        obj_func=opt.energy_cost, dim=25,
        bounds=[(1,500)] + [(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter)
    return algo.optimize()


def optimize_gwwoa(opt, levy, chaos):
    algo = GWWOA(
        obj_func=opt.energy_cost, dim=25,
        bounds=[(1,500)] + [(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter,
        levy_prob=levy, chaos_prob=chaos, beta=1.5)
    return algo.optimize()

#-----------------------------------------------------------------------------
# Ablation Study with Best Params
#-----------------------------------------------------------------------------
best_levy, best_chaos = 0.05, 0.1
variants = [
    ('GWO only', None, None),
    ('GWO + WOA', 0.0, 0.0),
    ('GWO + WOA + Lévy', best_levy, 0.0),
    ('Full hybrid (+ chaos)', best_levy, best_chaos)
]

n_trials = 100
results = {}
for name, levy, chaos in variants:
    costs = []
    for t in range(n_trials):
        opt = RenewableOptimizer(population=30, max_iter=50)
        opt.load_data(t)
        if name == 'GWO only':
            _, hist = optimize_gwo(opt)
        else:
            _, hist = optimize_gwwoa(opt, levy, chaos)
        costs.append(min(hist))
    results[name] = np.mean(costs)

# Build DataFrame
ablation_df = pd.DataFrame.from_dict(results, orient='index', columns=['Mean Cost'])
base_cost = ablation_df.loc['GWO only', 'Mean Cost']
ablation_df['% Change vs. base'] = ((ablation_df['Mean Cost'] - base_cost)/base_cost*100).round(2)

print(ablation_df)
# Save for manuscript
ablation_df.to_csv('ablation_results.csv')
