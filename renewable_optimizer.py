import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma
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
        self.data_variation = 0.15  # %15 rastgele varyasyon
        self.temperature = 25
        self.SOC_capacity_decay = [1.0 - 0.005 * i for i in range(24)]
        self.emergency_event = np.zeros(24)

    def load_data(self, trial_num):
        np.random.seed(trial_num)
        t = np.arange(self.hours)

        # Temel veri oluşturma
        base_solar = 50 * np.sin(np.pi * (t - 6) / 12) + 1
        base_wind = 30 * np.cos(np.pi * (t - 12) / 6) + 5
        base_demand = 80 + 30 * np.sin(np.pi * (t + 6) / 12)
        base_price = 0.15 + 0.05 * np.sin(np.pi * (t - 8) / 12) + 0.05

        # Varyasyon ve kesintiler
        variation = 1 + self.data_variation * np.random.randn(self.hours)
        self.solar_failure = np.random.choice([0, 1], 24, p=[0.9, 0.1])
        self.wind_failure = np.random.choice([0, 1], 24, p=[0.85, 0.15])
        self.grid_available = np.random.choice([0, 1], 24, p=[0.2, 0.8])
        self.emergency_event = np.random.choice([0, 1], 24, p=[0.9, 0.1])

        # Üretim ve talep
        self.P_solar = np.clip(base_solar * variation * self.solar_failure, 0, None)
        self.P_wind = np.clip(base_wind * variation * self.wind_failure, 0, None)
        self.P_gen = self.P_solar + self.P_wind
        self.P_demand = base_demand * variation * np.random.normal(1, 0.15, 24)

        # Fiyatlandırma ve spike'lar
        self.grid_price = np.clip(base_price * variation, 0.05, None)
        spike_hours = np.random.choice(24, size=4, replace=False)
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

            # Acil durum yükü
            if self.emergency_event[t_i]:
                required_power = self.P_demand[t_i] * 1.5
                shortage = max(0, required_power - (self.P_gen[t_i] + P_bess))
                total_cost += shortage * 1000

            # Şebeke kısıtları
            if not self.grid_available[t_i] and P_grid > 0:
                total_cost += 1e6

            # Maliyetler
            grid_cost = P_grid * self.grid_price[t_i] if P_grid > 0 else 0
            degradation = 0.02 * (abs(P_bess) ** 1.5) * (1 + SOC / 0.9)
            carbon_cost = P_grid * 0.487 * 2
            total_cost += grid_cost + degradation + carbon_cost

            # Termal model
            delta_temp = abs(P_bess / effective_S) / 0.05
            self.temperature += delta_temp
            if self.temperature > 45:
                total_cost += (self.temperature - 45) ** 2 * 10

            # SOC güncelleme
            if P_bess < 0:
                delta = (-P_bess * self.charge_eff) / effective_S
            else:
                delta = -P_bess / (self.discharge_eff * effective_S)
            SOC += delta
            SOC = np.clip(SOC, self.SOC_min, self.SOC_max)
            SOC_history.append(SOC)

            # SOC zincirleme kısıt
            if t_i >= 3:
                avg_soc = np.mean(SOC_history[-3:])
                if abs(SOC - avg_soc) > 0.2:
                    total_cost += 1e4

        # Verimlilik hedefi
        efficiency = (sum(self.P_gen) + sum(u * S)) / sum(self.P_demand)
        if efficiency < 0.85:
            total_cost += (0.85 - efficiency) * 1e4

        # Periyodik maliyet
        total_cost += 100 * abs(np.sin(S * 0.01))

        return total_cost
    
    def run_ga(self):
        """Genetic Algorithm implementation with callback history."""
        # Alt ve üst sınırlarınızı önce tanımlayın:
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), …]

        ga = GA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            crossover_rate=0.8,
            mutation_rate=0.1,
        )

        # Callback sınıfı
        class GACallback:
            def __init__(self):
                self.history = []

            def register(self, ga_instance):
                self.ga = ga_instance

            def update(self):
                # ga.best_y en güncel en iyi fitness
                self.history.append(self.ga.best_y)

        callback = GACallback()
        ga.callback = callback
        callback.register(ga)

        # GA'yı çalıştır
        best_x, history = ga.run()

        # Eğer callback hiçbir güncelleme yapmadıysa yine de final değeri ekle
        if not callback.history:
            callback.history.append(history[-1])

        return best_x, callback.history

    def run_gwwoa(self):
        """GW-WOA Algorithm implementation"""
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        gwwoa = GWWOA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            levy_prob=0.1,
            chaos_prob=0.1,
            beta=1.5,
        )
        best_solution, fitness_history = gwwoa.optimize()
        return best_solution, fitness_history
    
    def run_woa(self):
        """Whale Optimization Algorithm implementation"""
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        woa = WOA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
        )
        best_solution, fitness_history = woa.optimize()
        return best_solution, fitness_history
    
    def run_hs(self):
        """Harmony Search implementation"""
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        hs = HS(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            hmcr=0.95,
            par=0.3,
            bandwidth=0.05,
        )
        best_solution, fitness_history = hs.optimize()
        return best_solution, fitness_history
    
    def run_fpa(self):
        """Flower Pollination Algorithm implementation"""
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        fpa = FPA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            p=0.8,
            beta=1.5,
        )
        best_solution, fitness_history = fpa.optimize()
        return best_solution, fitness_history

    def run_pso(self):
        """Particle Swarm Optimization implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # PSO nesnesi
        pso = PSO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            inertia=0.7,
            c1=1.5,
            c2=1.5,
        )

        # Çalıştır ve sonuçları al
        best_x, history = pso.optimize()
        return best_x, history

    def run_gwo(self):
        """Grey Wolf Optimizer implementation"""
        # Alt ve üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # GWO nesnesini oluştur
        gwo = GWO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )

        # Çalıştır ve sonuçları al
        best_x, history = gwo.optimize()
        return best_x, history

    def run_cso(self):
        """Cat Swarm Optimization implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # CSO nesnesi
        cso = CSO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            mr=0.3,
            smp=5,
            c1=2.0
        )

        # Optimize et ve sonuçları al
        best_x, history = cso.optimize()
        return best_x, history

    def run_hho(self):
        """Harris Hawks Optimization implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # HHO nesnesini oluştur
        hho = HHO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )

        # Çalıştır ve sonuçları al
        best_x, history = hho.optimize()
        return best_x, history
    
    def run_bfo(self):
        """Bacterial Foraging Optimization implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # BFO nesnesi: chem_steps olarak self.max_iter kullanıyoruz
        bfo = BFO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            chem_steps=self.max_iter,
            # Diğer parametreler için varsayılan değerleri kullanıyoruz:
            # swim_length=4, repro_steps=2, elim_disp_steps=2, Ped=0.25
        )

        # Optimize et ve sonuçları al
        best_x, history = bfo.optimize()
        return best_x, history
   
    def run_fss(self):
        """Fish School Search implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # FSS nesnesi
        fss = FSS(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter,
            step_ind=0.1,
            step_vol=0.01
        )

        # Optimize et ve sonuçları al
        best_x, history = fss.optimize()
        return best_x, history


    def run_mfo(self):
        """Moth-Flame Optimization implementation"""
        # Alt/üst sınırlar
        lb = [1.0] + [-0.5] * 24
        ub = [2000.0] + [0.5] * 24
        bounds = list(zip(lb, ub))  # [(1.0,2000.0), (-0.5,0.5), ...]

        # MFO nesnesi
        mfo = MFO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )

        # Optimize et ve sonuçları al
        best_x, history = mfo.optimize()
        return best_x, history
    
    def run_pfa(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        algo = PFA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history
    
    def run_hoa(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        algo = HOA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history
    
    def run_apo(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        algo = APO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history
    
    def run_tta(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        algo = TTA(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history
    
    def run_mayfly(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        algo = MayflyAlgorithm(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history
        
    def run_cpso(self):
        bounds = [[1, 2000]] + [[-0.5, 0.5]] * 24
        # Burada basit constraint: S>0, tüm u in [-0.5, 0.5] kalmalı
        constraints = [
            lambda x: x[0] > 0,            # Battery capacity
            lambda x: np.all(x[1:] >= -0.5),
            lambda x: np.all(x[1:] <= 0.5)
        ]
        algo = CPSO(
            obj_func=lambda x: self.energy_cost(x),
            dim=25,
            bounds=bounds,
            constraints=constraints,
            population_size=self.pop_size,
            max_iter=self.max_iter
        )
        best_solution, fitness_history = algo.optimize()
        return best_solution, fitness_history

def calculate_soc(solution, hours=24):
    """Calculate SOC time series from solution"""
    S = solution[0]
    u = solution[1:]
    SOC = np.zeros(hours)
    SOC[0] = 0.5

    for t in range(1, hours):
        P_bess = u[t] * S
        if P_bess < 0:  # Charging
            delta = (-P_bess * 0.95) / S
        else:  # Discharging
            delta = -P_bess / (0.95 * S)
        SOC[t] = SOC[t - 1] + delta
        SOC[t] = np.clip(SOC[t], 0.1, 0.9)

    return SOC


def plot_convergence(results):
    plt.figure(figsize=(12, 8))
    for algo, data in results.items():
        plt.plot(data["history"], label=algo)
    plt.title("Algorithm Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_comparison.png")
    plt.close()


# Analiz ve görselleştirme fonksiyonları
def plot_convergence(results):
    plt.figure(figsize=(12, 8))
    for algo, data in results.items():
        if len(data["histories"]) > 0:
            plt.plot(np.nanmean(data["histories"], axis=0), label=algo)
    plt.title("Algorithm Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_comparison.png")
    plt.close()


def plot_soc_comparison(results):
    algorithms = list(results.keys())
    num_algorithms = len(algorithms)

    # Her algoritma için 3 birim yükseklik
    plt.figure(figsize=(14, 3 * num_algorithms))

    for idx, algo in enumerate(algorithms, 1):
        plt.subplot(num_algorithms, 1, idx)  # n satır, 1 sütun, pozisyon idx

        data = results[algo]
        solutions = [s for s in data["solutions"] if s is not None]

        if solutions:
            soc = calculate_soc(solutions[-1])  # Son başarılı çözümü al
            plt.plot(soc, color="tab:blue", linewidth=2)
            plt.fill_between(range(len(soc)), soc, alpha=0.2, color="tab:blue")

            # Grafik düzenlemeleri
            plt.title(f"{algo} - State of Charge", fontsize=12, pad=10)
            plt.xlabel("Hour", fontsize=10)
            plt.ylabel("SOC (%)", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.xticks(range(0, 24, 2))
            plt.ylim(0, 1)

        else:
            plt.text(
                0.5,
                0.5,
                "No Valid Solution",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                color="red",
                fontsize=12,
            )
            plt.axis("off")

    plt.tight_layout(pad=3.0)
    plt.savefig("soc_comparison_subplots.png", dpi=300, bbox_inches="tight")
    plt.close()


def population_sensitivity():
    populations = [20, 30, 50, 70]
    results = {}

    for pop in populations:
        optimizer = RenewableOptimizer(population=pop)
        optimizer.load_data(0)  # Load data with trial=0
        _, cost_history = optimizer.run_gwwoa()
        results[pop] = cost_history[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), "bo-")
    plt.title("Population Size Sensitivity")
    plt.xlabel("Population Size")
    plt.ylabel("Final Cost ($)")
    plt.grid(True)
    plt.savefig("population_sensitivity.png")
    plt.close()


def run_multiple_trials(optimizer_class, algorithms, num_trials=100):
    results = {
        name: {"costs": [], "histories": [], "solutions": []}
        for name in algorithms.keys()
    }

    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")
        optimizer = optimizer_class()  # Create a new instance for each trial
        optimizer.load_data(trial)  # Load data for this trial

        trial_results = {}
        for name, method_name in algorithms.items():
            try:
                # Get the method by name from the current optimizer instance
                method = getattr(optimizer, method_name)
                best_x, history = method()
                trial_results[name] = {
                    "cost": history[-1] if history else np.inf,
                    "history": history,
                    "solution": best_x,
                }
            except Exception as e:
                print(f"{name} failed: {str(e)}")
                trial_results[name] = {"cost": np.inf, "history": [], "solution": None}

        for name, data in trial_results.items():
            results[name]["costs"].append(data["cost"])
            results[name]["histories"].append(data["history"])
            results[name]["solutions"].append(data["solution"])

    return results


def analyze_results(results):
    """Sonuçların istatistiksel analizi"""
    analysis = {}

    for algo, data in results.items():
        costs = np.array(data["costs"])
        valid_trials = costs[np.isfinite(costs)]

        if len(valid_trials) == 0:
            analysis[algo] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "success_rate": 0.0,
            }
            continue

        analysis[algo] = {
            "mean": np.mean(valid_trials),
            "std": np.std(valid_trials),
            "min": np.min(valid_trials),
            "max": np.max(valid_trials),
            "success_rate": len(valid_trials) / len(costs),
        }

    return analysis


def plot_mean_convergence(results, analysis):
    """Ortalama yakınsama eğrilerini çizdirme"""
    plt.figure(figsize=(12, 8))

    for algo, data in results.items():
        if analysis[algo]["success_rate"] == 0:
            continue

        # Tarihçeleri aynı uzunluğa getirme
        max_length = max(len(h) for h in data["histories"])
        padded_histories = [
            h + [h[-1]] * (max_length - len(h)) for h in data["histories"]
        ]

        mean_history = np.nanmean(padded_histories, axis=0)
        std_history = np.nanstd(padded_histories, axis=0)

        plt.plot(mean_history, label=algo)
        plt.fill_between(
            range(max_length),
            mean_history - std_history,
            mean_history + std_history,
            alpha=0.2,
        )

    plt.title("Average Convergence Trends (100 Trials)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Cost ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("average_convergence.png")
    plt.close()


# Çalıştırma ve sonuç analizi
if __name__ == "__main__":
    algorithms = {
        "GA": "run_ga",
        "GWWOA": "run_gwwoa",
        "WOA": "run_woa",
        "HS": "run_hs",
        "FPA": "run_fpa",
        "Mayfly": "run_mayfly",
        "PFA": "run_pfa",
        "HOA": "run_hoa",
        "APO": "run_apo",
        "TTA": "run_tta",
        "CPSO": "run_cpso"
    }
    results = run_multiple_trials(RenewableOptimizer, algorithms, num_trials=100)
    analysis = analyze_results(results)

    print("\n=== Final Performance Summary ===")
    print(
        f"{'Algorithm':<10}\t{'Mean':<12}\t{'Std':<10}\t{'Min':<12}\t{'Max':<12}\t{'Success':<8}"
    )
    for algo, data in analysis.items():
        print(
            f"{algo:<10}\t${data['mean']:,.2f}\t±{data['std']:.2f}  "
            f"${data['min']:,.2f}\t${data['max']:,.2f}\t{data['success_rate']:.1%}"
        )

    plot_mean_convergence(results, analysis)
    plot_soc_comparison(results)
    population_sensitivity()
