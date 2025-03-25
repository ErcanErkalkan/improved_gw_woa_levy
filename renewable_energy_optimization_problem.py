import numpy as np
from gwwo import GWWOA
# Problem parameters
hours = 24
capital_cost = 500  # $/kWh
c_deg = 0.02        # $/kWh cycled
charge_eff = 0.95
discharge_eff = 0.95
SOC_min, SOC_max = 0.1, 0.9

# Synthetic data
t = np.arange(hours)
P_gen = 50 * (np.sin(np.pi*(t-6)/12) + 1)       # Generation profile
P_demand = 80 + 30*np.sin(np.pi*(t+6)/12)        # Demand profile
grid_price = 0.15 + 0.05*np.sin(np.pi*(t-8)/12) + 0.05  # Time-of-use pricing

def energy_cost(solution):
    S = solution[0]
    u = solution[1:]
    
    if S < 1: return np.inf
    
    total_cost = capital_cost * S
    SOC = 0.5
    
    for t_i in range(hours):
        P_bess = u[t_i] * S
        P_grid = P_demand[t_i] - P_gen[t_i] - P_bess
        
        # Penalize grid overproduction
        if P_grid < 0:
            total_cost += -P_grid * 0.1
            P_grid = 0
        
        total_cost += P_grid * grid_price[t_i] + c_deg * abs(P_bess)
        
        # Update SOC
        if P_bess < 0:  # Charging
            delta = (-P_bess * charge_eff) / S
        else:            # Discharging
            delta = -P_bess / (discharge_eff * S)
        
        SOC += delta
        if SOC < SOC_min or SOC > SOC_max:
            total_cost += 1e5
        
        SOC = np.clip(SOC, SOC_min, SOC_max)
    
    return total_cost

# Problem dimensions: 1 (capacity) + 24 (hourly control)
dim = 25
bounds = np.array([[1, 200]] + [[-0.5, 0.5]]*24)

# Initialize and run GW-WOA
gwwa = GWWOA(energy_cost, dim, bounds, max_iter=100)
best_sol, fitness_hist = gwwa.optimize()

print(f"Optimal Battery Capacity: {best_sol[0]:.2f} kWh")
print(f"Minimum Total Cost: ${fitness_hist[-1]:.2f}")

# Implement other algorithms (PSO, GWO) similarly and compare convergence
import matplotlib.pyplot as plt

# Assume pso_fitness and gwo_fitness are obtained from other implementations
plt.figure(figsize=(10,6))
plt.plot(fitness_hist, label='GW-WOA')
#plt.plot(pso_fitness, label='PSO')
#plt.plot(gwo_fitness, label='GWO')
plt.xlabel('Iteration')
plt.ylabel('Total Cost ($)')
plt.title('Algorithm Convergence Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Plot optimal battery schedule
plt.figure(figsize=(12,6))
plt.plot(t, best_sol[1:], 'b-o', label='Charge/Discharge Rate')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Normalized Power')
plt.title('Optimal Battery Schedule')
plt.grid(True)
plt.show()