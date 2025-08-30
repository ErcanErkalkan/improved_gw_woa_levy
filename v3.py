import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms import GA, GWWOA, GWO

class RenewableOptimizer:
    def __init__(self, hours=24, population=30, max_iter=50):
        self.hours = hours
        self.pop_size = population
        self.max_iter = max_iter
        # Problem parameters
        self.capital_cost = 500    # $/kWh
        self.c_deg = 0.02
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.SOC_min, self.SOC_max = 0.1, 0.9
        self.data_variation = 0.15
        # capacity decay per hour
        self.SOC_capacity_decay = [1.0 - 0.005 * i for i in range(self.hours)]

    def load_data(self, trial_num):
        np.random.seed(trial_num)
        t = np.arange(self.hours)
        base_solar = np.maximum(50 * np.sin(np.pi * (t - 6) / 12), 0)
        base_wind  = np.maximum(30 * np.cos(np.pi * (t - 12) / 6),  0)
        base_demand = 80 + 30 * np.sin(np.pi * (t + 6) / 12)
        base_price  = 0.15 + 0.05 * np.sin(np.pi * (t - 8) / 12) + 0.05
        var = 1 + self.data_variation * np.random.randn(self.hours)
        self.solar_fail = np.random.rand(self.hours) < 0.1
        self.wind_fail  = np.random.rand(self.hours) < 0.15
        self.grid_ok    = np.random.rand(self.hours) < 0.8
        self.emergency  = np.random.rand(self.hours) < 0.1

        self.P_solar = base_solar * var * self.solar_fail
        self.P_wind  = base_wind  * var * self.wind_fail
        self.P_gen   = self.P_solar + self.P_wind
        self.P_demand = base_demand * var * np.random.normal(1,0.15,self.hours)
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
            P_bess = u[t] * eff_S      # + -> charge, - -> discharge
            P_grid = self.P_demand[t] - self.P_gen[t] - P_bess
            # emergency penalty
            if self.emergency[t]:
                req = self.P_demand[t] * 1.5
                short = max(0, req - (self.P_gen[t] + P_bess))
                cost += short * 1e3
            # grid availability
            if not self.grid_ok[t] and P_grid > 0:
                cost += 1e6
            # grid cost & carbon
            if P_grid > 0:
                cost += P_grid * self.grid_price[t]
                cost += P_grid * 0.487 * 2
            # degradation
            cost += 0.02 * abs(P_bess)**1.5 * (1 + SOC/0.9)
            # temperature penalty
            deltaT = abs(P_bess/eff_S)/0.05
            if deltaT > 0:
                cost += max(0, (deltaT - (45-25)))**2 * 10
            # update SOC
            if P_bess > 0:
                dSOC = (P_bess * self.charge_eff) / eff_S
            else:
                dSOC = (P_bess / self.discharge_eff) / eff_S
            SOC = np.clip(SOC + dSOC, self.SOC_min, self.SOC_max)
        # efficiency penalty
        eff = (self.P_gen.sum() + (u*S).sum())/self.P_demand.sum()
        if eff < 0.85:
            cost += (0.85 - eff)*1e4
        return cost

# Optimization wrappers

def optimize_gwo(opt):
    algo = GWO(
        obj_func=opt.energy_cost, dim=25,
        bounds=[(1,500)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter)
    return algo.optimize()


def optimize_gwwoa(opt, levy_prob, chaos_prob):
    algo = GWWOA(
        obj_func=opt.energy_cost, dim=25,
        bounds=[(1,500)]+[(-0.5,0.5)]*24,
        population_size=opt.pop_size, max_iter=opt.max_iter,
        levy_prob=levy_prob, chaos_prob=chaos_prob, beta=1.5)
    return algo.optimize()

# Run parameter sweep and plot
levy_vals  = [0.02, 0.05, 0.1]
chaos_vals = [0.01, 0.05, 0.1]
results = []
for l in levy_vals:
    for c in chaos_vals:
        cfg = {'pop':30, 'iters':50}
        opt = RenewableOptimizer(population=30, max_iter=50)
        opt.load_data(0)
        costs = []
        for i in range(20):
            opt.load_data(i)
            _, hist = optimize_gwwoa(opt, l, c)
            costs.append(min(hist))
        results.append({'levy':l, 'chaos':c, 'mean_cost':np.mean(costs)})
# DataFrame and bar chart
sweep_df = pd.DataFrame(results)
sweep_pivot = sweep_pivot = sweep_df.pivot(index='levy', columns='chaos', values='mean_cost')
sweep_pivot.plot(kind='bar', figsize=(8,5))
plt.title('Param Sweep: Mean Cost')
plt.ylabel('Cost ($)')
plt.tight_layout()
plt.savefig('param_sweep.png')
plt.show()
# Best params
best = sweep_df.loc[sweep_df['mean_cost'].idxmin()]
best_levy, best_chaos = best['levy'], best['chaos']
print(f"Best levy={best_levy}, chaos={best_chaos}, cost={best['mean_cost']:.1f}")

# Final comparison GWO vs Hybrid
stats = {}
def run_trials(fn, name):
    arr=[]
    for t in range(100):
        opt = RenewableOptimizer(population=30, max_iter=50)
        opt.load_data(t)
        _, hist = fn(opt) if name=='GWO' else fn(opt, best_levy, best_chaos)
        arr.append(min(hist))
    return np.array(arr)
stats['GWO'] = run_trials(optimize_gwo, 'GWO')
stats['EGW-WOA'] = run_trials(lambda o,lv,ch: optimize_gwwoa(o,lv,ch), 'EGW-WOA')
# Summary table
summary = {k: [v.mean(), v.std(), v.min(), v.max()] for k,v in stats.items()}
summary_df = pd.DataFrame(summary, index=['Mean','Std','Min','Max']).T
print(summary_df)
summary_df.to_csv('performance_summary.csv')
# Convergence plots for one trial
opt = RenewableOptimizer(); opt.load_data(0)
_, hist_gwo = optimize_gwo(opt)
_, hist_hyb = optimize_gwwoa(opt, best_levy, best_chaos)
plt.figure()
plt.plot(hist_gwo, label='GWO')
plt.plot(hist_hyb, label='EGW-WOA')
plt.xlabel('Iteration'); plt.ylabel('Best Cost')
plt.legend(); plt.title('Convergence Comparison')
plt.savefig('convergence.png'); plt.show()
