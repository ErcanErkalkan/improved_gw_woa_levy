import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
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
        self.P_demand  = base_demand * var * np.random.normal(1, 0.15, self.hours)
        self.grid_price = np.clip(base_price * var, 0.05, None)
        spikes = np.random.choice(self.hours, 4, replace=False)
        self.grid_price[spikes] *= np.random.uniform(3,5,size=4)

    def energy_cost(self, x):
        S, u = x[0], x[1:]
        cost = self.capital_cost * S
        SOC = 0.5
        for t in range(self.hours):
            eff_S = S * self.SOC_capacity_decay[t]
            P_bess = u[t] * eff_S
            P_grid = self.P_demand[t] - self.P_gen[t] - P_bess
            # emergency
            if self.emergency[t]:
                req = self.P_demand[t]*1.5
                short = max(0, req-(self.P_gen[t]+P_bess))
                cost += short*1e3
            # grid avail
            if not self.grid_ok[t] and P_grid>0:
                cost += 1e6
            # grid & carbon
            if P_grid>0:
                cost += P_grid*self.grid_price[t] + P_grid*0.487*2
            # degradation
            cost += 0.02*abs(P_bess)**1.5*(1+SOC/0.9)
            # temp
            deltaT = abs(P_bess/eff_S)/0.05
            if deltaT>0:
                cost += max(0,deltaT-20)**2*10
            # SOC update
            if P_bess>0:
                dSOC = (P_bess*self.charge_eff)/eff_S
            else:
                dSOC = (P_bess/self.discharge_eff)/eff_S
            SOC = np.clip(SOC+dSOC, self.SOC_min, self.SOC_max)
        eff = (self.P_gen.sum()+(u*S).sum())/self.P_demand.sum()
        if eff<0.85:
            cost += (0.85-eff)*1e4
        return cost

    # wrapper methods
    def run_ga(self): return GA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_gwo(self): return GWO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_gwwoa(self): return GWWOA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter,self.levy_prob,self.chaos_prob,1.5).optimize()
    def run_woa(self): return WOA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_hs(self): return HS(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_fpa(self): return FPA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_pso(self): return PSO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_cso(self): return CSO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_hho(self): return HHO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_bfo(self): return BFO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_fss(self): return FSS(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_mfo(self): return MFO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_mayfly(self): return MayflyAlgorithm(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_pfa(self): return PFA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_hoa(self): return HOA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_apo(self): return APO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_tta(self): return TTA(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,self.pop_size,self.max_iter).optimize()
    def run_cpso(self): return CPSO(self.energy_cost,25,[(1,500)]+[(-0.5,0.5)]*24,[lambda x: x[0]>0],self.pop_size,self.max_iter).optimize()

# ---------------------- Utility Functions ----------------------
def calculate_soc(x, hours=24):
    S, u = x[0], x[1:]
    soc = np.zeros(hours); soc[0]=0.5
    for t in range(1,hours):
        Pb = u[t]*S
        if Pb>0: d=(Pb*0.95)/S
        else:    d=(Pb/0.95)/S
        soc[t]=np.clip(soc[t-1]+d,0.1,0.9)
    return soc

# ---------------------- Plotting ----------------------
def plot_mean_convergence(results, analysis):
    plt.figure(figsize=(10,6))
    for algo,data in results.items():
        if analysis[algo]['success_rate']>0:
            maxlen = max(len(h) for h in data['histories'])
            padded = [h+ [h[-1]]*(maxlen-len(h)) for h in data['histories']]
            m = np.mean(padded,axis=0); s=np.std(padded,axis=0)
            plt.plot(m,label=algo)
            plt.fill_between(range(maxlen),m-s,m+s,alpha=0.2)
    plt.title('Avg Convergence (100 trials)');plt.xlabel('Iter');plt.ylabel('Cost');plt.legend();plt.grid();plt.savefig('average_convergence.png')

# Grouped SOC plots
def plot_soc_groups(results, groups):
    for name, algos in groups.items():
        plt.figure(figsize=(10,4*len(algos)))
        for i,algo in enumerate(algos,1):
            plt.subplot(len(algos),1,i)
            sols=[s for s in results[algo]['solutions'] if s is not None]
            if sols:
                soc=calculate_soc(sols[-1])
                plt.plot(soc); plt.fill_between(range(len(soc)),soc,alpha=0.2)
            plt.title(algo)
            plt.ylim(0,1); plt.grid()
        plt.tight_layout(); plt.savefig(f'soc_group_{name}.png')

# Population Sensitivity
def plot_population_sensitivity(best_levy,best_chaos):
    pops=[20,30,50,70]; costs=[]
    for p in pops:
        opt=RenewableOptimizer(population=p,levy=best_levy,chaos=best_chaos)
        opt.load_data(0); _,h=opt.run_gwwoa(); costs.append(h[-1])
    plt.figure();plt.plot(pops,costs,'-o');plt.xlabel('Pop Size');plt.ylabel('Cost');plt.grid();plt.savefig('population_sensitivity.png')

# ---------------------- Execution ----------------------
if __name__=='__main__':
    best_levy, best_chaos = 0.05, 0.1
    algos = {
        'GA':'run_ga','GWO':'run_gwo','GW-WOA':'run_gwwoa','WOA':'run_woa',
        'HS':'run_hs','FPA':'run_fpa','PSO':'run_pso','CSO':'run_cso',
        'HHO':'run_hho','BFO':'run_bfo','FSS':'run_fss','MFO':'run_mfo',
        'Mayfly':'run_mayfly','PFA':'run_pfa','HOA':'run_hoa','APO':'run_apo','TTA':'run_tta','CPSO':'run_cpso'
    }
    results={a:{'costs':[],'histories':[],'solutions':[]} for a in algos}
    for t in range(100):
        opt=RenewableOptimizer(levy=best_levy,chaos=best_chaos)
        opt.load_data(t)
        for name,fn in algos.items():
            try:
                sol,h=getattr(opt,fn)()
                results[name]['solutions'].append(sol)
                results[name]['histories'].append(h)
                results[name]['costs'].append(min(h))
            except:
                results[name]['solutions'].append(None)
                results[name]['histories'].append([])
                results[name]['costs'].append(np.inf)
    analysis={a:{'mean':np.nanmean(r['costs']),'std':np.nanstd(r['costs']),'min':np.nanmin(r['costs']),'max':np.nanmax(r['costs']),'success_rate':np.isfinite(r['costs']).mean()} for a,r in results.items()}
    print('Final Performance Summary:')
    for a,v in analysis.items(): print(f"{a}: mean={v['mean']:.1f}, std={v['std']:.1f}, min={v['min']:.1f}, max={v['max']:.1f}, success={v['success_rate']:.0%}")
    plot_mean_convergence(results,analysis)
    groups={'Group1':['GA','GWO','GW-WOA','WOA'],'Group2':['HS','FPA','PSO','CSO'],'Group3':['HHO','BFO','FSS','MFO'],'Group4':['Mayfly','PFA','HOA','APO','TTA','CPSO']}
    plot_soc_groups(results,groups)
    plot_population_sensitivity(best_levy,best_chaos)
