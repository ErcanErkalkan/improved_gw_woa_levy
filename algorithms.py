import numpy as np
from scipy.special import gamma
# ---------------------- Common Helpers ----------------------

def clip_positions(positions, bounds):
    return np.clip(positions, bounds[:, 0], bounds[:, 1])

# ---------------------- Genetic Algorithm ----------------------
class GA:
    def __init__(self, obj_func, dim, bounds,
                 population_size=50, max_iter=100,
                 crossover_rate=0.8, mutation_rate=0.1):
        self.obj_func = obj_func
        self.dim = dim
        # bounds: list of (low, high) tuples, length == dim
        self.bounds = np.array(bounds)               # shape (dim,2)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.cr = crossover_rate
        self.mr = mutation_rate

        # initialize population
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        self.positions = np.random.uniform(lows, highs,
                                           size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.history = [np.min(self.fitness)]
        self.best_y = self.history[0]

    def tournament_selection(self, k=3):
        indices = np.random.choice(self.pop_size, k, replace=False)
        best = indices[np.argmin(self.fitness[indices])]
        return self.positions[best].copy()

    def crossover(self, p1, p2):
        if np.random.rand() < self.cr:
            point = np.random.randint(1, self.dim)
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mr:
                span = self.bounds[i,1] - self.bounds[i,0]
                perturb = np.random.randn() * span * 0.1
                individual[i] += perturb
        lows = self.bounds[:,0]
        highs = self.bounds[:,1]
        return np.clip(individual, lows, highs)

    def optimize(self):
        for _ in range(self.max_iter):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))
            self.positions = np.array(new_pop)
            self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)

            current_best = np.min(self.fitness)
            self.history.append(current_best)
            self.best_y = current_best

            # Callback varsa, kaydet
            if hasattr(self, 'callback') and self.callback is not None:
                self.callback.update()

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

    # run metodu olarak alias
    def run(self):
        return self.optimize()

class GWWOA:
    def __init__(self, obj_func, dim, bounds, population_size=50, max_iter=100, 
                 levy_prob=0.1, chaos_prob=0.1, beta=1.5):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.levy_prob = levy_prob
        self.chaos_prob = chaos_prob
        self.beta = beta 
        
        # Initialize population with chaotic logistic map for half the population
        self.positions = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], 
                                         size=(self.pop_size, self.dim))
        for i in range(self.pop_size//2):
            self.positions[i] = self.logistic_map()
        
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.sort_population()
        self.alpha = self.positions[0]
        self.beta_wolf = self.positions[1]
        self.delta = self.positions[2]
        self.best_fitness = [self.fitness[0]]
    
    def logistic_map(self, iterations=100):
        x = np.zeros(self.dim)
        x[0] = np.random.rand()
        for i in range(1, self.dim):
            x[i] = 4 * x[i-1] * (1 - x[i-1])
        return x * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
    
    def sort_population(self):
        idx = np.argsort(self.fitness)
        self.positions = self.positions[idx]
        self.fitness = self.fitness[idx]
    
    def levy_flight(self):
        sigma = (gamma(1+self.beta)*np.sin(np.pi*self.beta/2)/(gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1/self.beta))
    
    def optimize(self):
        for iter in range(self.max_iter):
            a = 2 - 2 * iter / self.max_iter
            omega_initial = 0.9
            omega_final = 0.4
            omega = omega_initial * (omega_final / omega_initial) ** (iter / self.max_iter)
            
            for i in range(self.pop_size):
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)
                p = np.random.rand()
                
                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # GWO encircling
                        D_alpha = np.abs(C * self.alpha - self.positions[i])
                        X1 = self.alpha - A * D_alpha
                        D_beta = np.abs(C * self.beta_wolf - self.positions[i])
                        X2 = self.beta_wolf - A * D_beta
                        D_delta = np.abs(C * self.delta - self.positions[i])
                        X3 = self.delta - A * D_delta
                        new_pos = (X1 + X2 + X3) / 3
                    else:
                        # GWO exploration
                        r1 = np.random.rand(self.dim)
                        r2 = np.random.rand(self.dim)
                        new_pos = self.positions[i] - A * np.abs(r1 * self.positions[i] - r2 * self.alpha)
                else:
                    # WOA spiral
                    l = np.random.uniform(-1, 1, self.dim)
                    D = np.abs(self.alpha - self.positions[i])
                    new_pos = D * np.exp(0.5*l) * np.cos(2*np.pi*l) + self.alpha
                
                # Lévy flight
                if np.random.rand() < self.levy_prob:
                    new_pos += omega * self.levy_flight()
                
                # Chaotic escape
                if np.random.rand() < self.chaos_prob:
                    chaos_pos = self.logistic_map()
                    mask = np.random.rand(self.dim) < 0.3
                    new_pos[mask] = chaos_pos[mask]
                
                # Apply bounds
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                
                # Evaluate new position
                new_fitness = self.obj_func(new_pos)
                
                # Update if improved
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness
            
            # Update leaders
            self.sort_population()
            self.alpha = self.positions[0]
            self.beta_wolf = self.positions[1]
            self.delta = self.positions[2]
            self.best_fitness.append(self.fitness[0])
        
        return self.alpha, self.best_fitness


class WOA:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        
        # Initialize population
        self.positions = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], 
                                       size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.positions[self.best_idx]
        self.best_fitness = self.fitness[self.best_idx]
        self.history = [self.best_fitness]
    
    def optimize(self):
        for iter in range(self.max_iter):
            a = 2 - 2 * iter / self.max_iter  # a decreases linearly from 2 to 0
            a2 = -1 + iter * (-1) / self.max_iter  # a2 decreases linearly from -1 to -2
            
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = np.random.uniform(-1, 1, self.dim)
                p = np.random.rand()
                
                if p < 0.5:
                    if np.abs(A).any() < 1:
                        # Encircling prey
                        D = np.abs(C * self.best_position - self.positions[i])
                        new_pos = self.best_position - A * D
                    else:
                        # Search for prey
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_pos = self.positions[rand_idx]
                        D = np.abs(C * rand_pos - self.positions[i])
                        new_pos = rand_pos - A * D
                else:
                    # Bubble-net attacking
                    D = np.abs(self.best_position - self.positions[i])
                    new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_position
                
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fitness = self.obj_func(new_pos)
                
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness
            
            # Update best solution
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_position = self.positions[current_best_idx]
                self.best_fitness = self.fitness[current_best_idx]
            
            self.history.append(self.best_fitness)
        
        return self.best_position, self.history

class HS:  # Harmony Search
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100,
                 hmcr=0.95, par=0.3, bandwidth=0.05):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.hmcr = hmcr  # Harmony memory considering rate
        self.par = par    # Pitch adjustment rate
        self.bw = bandwidth
        
        # Initialize harmony memory
        self.harmony_memory = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], 
                                              size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.harmony_memory)
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.harmony_memory[self.best_idx]
        self.best_fitness = self.fitness[self.best_idx]
        self.history = [self.best_fitness]
    
    def optimize(self):
        for _ in range(self.max_iter):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    # Memory consideration
                    new_harmony[j] = self.harmony_memory[np.random.randint(0, self.pop_size), j]
                    # Pitch adjustment
                    if np.random.rand() < self.par:
                        new_harmony[j] += self.bw * np.random.uniform(-1, 1)
                else:
                    # Random selection
                    new_harmony[j] = np.random.uniform(self.bounds[j,0], self.bounds[j,1])
            
            new_harmony = np.clip(new_harmony, self.bounds[:,0], self.bounds[:,1])
            new_fitness = self.obj_func(new_harmony)
            
            # Update harmony memory
            worst_idx = np.argmax(self.fitness)
            if new_fitness < self.fitness[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.fitness[worst_idx] = new_fitness
                # Update best solution
                if new_fitness < self.best_fitness:
                    self.best_solution = new_harmony
                    self.best_fitness = new_fitness
            
            self.history.append(self.best_fitness)
        
        return self.best_solution, self.history

class FPA:  # Flower Pollination Algorithm
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100,
                 p=0.8, beta=1.5):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.p = p       # Switch probability
        self.beta = beta # Lévy exponent
        
        # Initialize population
        self.positions = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], 
                                         size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.positions[self.best_idx]
        self.best_fitness = self.fitness[self.best_idx]
        self.history = [self.best_fitness]
    
    def levy_flight(self):
        sigma = (gamma(1+self.beta)*np.sin(np.pi*self.beta/2)/(gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / np.abs(v)**(1/self.beta)
    
    def optimize(self):
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.p:
                    # Global pollination
                    L = self.levy_flight()
                    new_pos = self.positions[i] + L * (self.best_position - self.positions[i])
                else:
                    # Local pollination
                    idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                    epsilon = np.random.rand()
                    new_pos = self.positions[i] + epsilon * (self.positions[idx1] - self.positions[idx2])
                
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fitness = self.obj_func(new_pos)
                
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness
                    # Update best solution
                    if new_fitness < self.best_fitness:
                        self.best_position = new_pos
                        self.best_fitness = new_fitness
            
            self.history.append(self.best_fitness)
        
        return self.best_position, self.history

# ---------------------- Particle Swarm Optimization ----------------------
class PSO:
    def __init__(self, obj_func, dim, bounds, population_size=50, max_iter=100,
                 inertia=0.7, c1=1.5, c2=1.5):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w = inertia
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                           size=(self.pop_size, self.dim))
        v_range = self.bounds[:,1] - self.bounds[:,0]
        self.velocities = np.random.uniform(-v_range, v_range,
                                            size=(self.pop_size, self.dim))
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.apply_along_axis(self.obj_func, 1, self.positions)
        gidx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[gidx].copy()
        self.gbest_val = self.pbest_val[gidx]
        self.history = [self.gbest_val]

    def optimize(self):
        for _ in range(self.max_iter):
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.w*self.velocities 
                               + self.c1*r1*(self.pbest_pos - self.positions)
                               + self.c2*r2*(self.gbest_pos - self.positions))
            self.positions += self.velocities
            self.positions = clip_positions(self.positions, self.bounds)
            fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
            mask = fitness < self.pbest_val
            self.pbest_pos[mask] = self.positions[mask]
            self.pbest_val[mask] = fitness[mask]
            gidx = np.argmin(self.pbest_val)
            if self.pbest_val[gidx] < self.gbest_val:
                self.gbest_val = self.pbest_val[gidx]
                self.gbest_pos = self.pbest_pos[gidx].copy()
            self.history.append(self.gbest_val)
        return self.gbest_pos, self.history

# ---------------------- Grey Wolf Optimizer ----------------------
class GWO:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                           size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj,1,self.positions)
        idx = np.argsort(self.fitness)
        self.alpha, self.beta_wolf, self.delta = [self.positions[i] for i in idx[:3]]
        self.history = [self.fitness[idx[0]]]

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - 2*t/self.max_iter
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = abs(C1*self.alpha - self.positions[i])
                X1 = self.alpha - A1*D_alpha
                # similarly for beta and delta
                A2, C2 = 2*a*np.random.rand(self.dim)-a, 2*np.random.rand(self.dim)
                D_beta = abs(C2*self.beta_wolf - self.positions[i])
                X2 = self.beta_wolf - A2*D_beta
                A3, C3 = 2*a*np.random.rand(self.dim)-a, 2*np.random.rand(self.dim)
                D_delta = abs(C3*self.delta - self.positions[i])
                X3 = self.delta - A3*D_delta
                new_pos = clip_positions((X1+X2+X3)/3, self.bounds)
                f = self.obj(new_pos)
                if f < self.fitness[i]:
                    self.positions[i], self.fitness[i] = new_pos, f
            idx = np.argsort(self.fitness)
            self.alpha, self.beta_wolf, self.delta = [self.positions[i] for i in idx[:3]]
            self.history.append(self.fitness[idx[0]])
        return self.alpha, self.history

# ---------------------- Cat Swarm Optimization ----------------------
class CSO:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100,
                 mr=0.3, smp=5, c1=2.0):
        self.obj = obj_func; self.dim=dim; self.bounds=np.array(bounds)
        self.pop_size=population_size; self.max_iter=max_iter
        self.mr = mr  # mixture ratio
        self.smp = smp  # seeking memory pool
        self.c1 = c1
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                           size=(self.pop_size,self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.fitness = np.apply_along_axis(self.obj,1,self.positions)
        self.history=[np.min(self.fitness)]

    def optimize(self):
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.mr:
                    # seeking mode
                    candidates = np.tile(self.positions[i],(self.smp,1))
                    for j in range(self.smp):
                        idx = np.random.randint(0,self.dim)
                        candidates[j,idx] += np.random.randn()*0.1*(self.bounds[idx,1]-self.bounds[idx,0])
                    c_fits = np.apply_along_axis(self.obj,1,candidates)
                    best = candidates[np.argmin(c_fits)]
                    self.positions[i]=clip_positions(best,self.bounds)
                else:
                    # tracing mode
                    r = np.random.rand(self.dim)
                    self.velocities[i] = self.velocities[i] + self.c1*r*(np.min(self.positions,axis=0)-self.positions[i])
                    self.positions[i] += self.velocities[i]
                    self.positions[i] = clip_positions(self.positions[i],self.bounds)
                self.fitness[i]=self.obj(self.positions[i])
            self.history.append(np.min(self.fitness))
        best_idx=np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Harris Hawks Optimization ----------------------
class HHO:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj=obj_func; self.dim=dim; self.bounds=np.array(bounds)
        self.pop_size=population_size; self.max_iter=max_iter
        self.positions=np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                         size=(self.pop_size,self.dim))
        self.fitness=np.apply_along_axis(self.obj,1,self.positions)
        self.history=[np.min(self.fitness)]
        self.best_idx=np.argmin(self.fitness)
        self.prey=self.positions[self.best_idx]

    def optimize(self):
        for t in range(1,self.max_iter+1):
            E1=2*(1 - t/self.max_iter)
            for i in range(self.pop_size):
                E0=2*np.random.rand()-1
                E=E1*E0
                J=2*(1-np.random.rand())
                if abs(E)>=1:
                    rand=np.random.randint(self.pop_size)
                    X_rand=self.positions[rand]
                    self.positions[i]=X_rand - np.random.rand()*abs(X_rand-2*np.random.rand()*self.positions[i])
                else:
                    if abs(E)<0.5:
                        self.positions[i]=self.prey - E*abs(J*self.prey-self.positions[i])
                    else:
                        X_m=np.mean(self.positions,axis=0)
                        self.positions[i]=(self.prey - self.positions[i]) - E*abs(J*X_m-self.positions[i])
                self.positions[i]=clip_positions(self.positions[i],self.bounds)
                f=self.obj(self.positions[i])
                if f<self.fitness[i]:
                    self.fitness[i]=f
                if f<self.obj(self.prey):
                    self.prey=self.positions[i]
            self.history.append(self.obj(self.prey))
        return self.prey, self.history

# ---------------------- Bacterial Foraging Optimization ----------------------
class BFO:
    def __init__(self, obj_func, dim, bounds, population_size=30, chem_steps=5,
                 swim_length=4, repro_steps=2, elim_disp_steps=2, Ped=0.25):
        self.obj=obj_func; self.dim=dim; self.bounds=np.array(bounds)
        self.pop_size=population_size; self.Cs=0.1
        self.S=swim_length; self.Nre=repro_steps; self.Ned=elim_disp_steps; self.Ped=Ped
        self.positions=np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                         size=(self.pop_size,self.dim))
        self.history=[np.min(np.apply_along_axis(self.obj,1,self.positions))]

    def optimize(self):
        for _ in range(self.Ned):
            for _ in range(self.Nre):
                for i in range(self.pop_size):
                    health=0
                    for _ in range(self.S):
                        cost=self.obj(self.positions[i])
                        delta=self.Cs*np.random.randn(self.dim)
                        new=clip_positions(self.positions[i]+delta,self.bounds)
                        if self.obj(new)<cost:
                            self.positions[i]=new; health+=self.obj(new)
                        else:
                            break
                    health+=self.obj(self.positions[i])
                # reproduction
            idx_sorted=np.argsort([self.obj(p) for p in self.positions])
            self.positions=self.positions[idx_sorted]
            half=self.pop_size//2
            self.positions[half:]=self.positions[:half].copy()
            # elimination-dispersal
            for i in range(self.pop_size):
                if np.random.rand()<self.Ped:
                    self.positions[i]=np.random.uniform(self.bounds[:,0], self.bounds[:,1], self.dim)
            self.history.append(np.min([self.obj(p) for p in self.positions]))
        best_idx=np.argmin([self.obj(p) for p in self.positions])
        return self.positions[best_idx], self.history


# ---------------------- Fish School Search ----------------------
class FSS:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100,
                 step_ind=0.1, step_vol=0.01):
        self.obj, self.dim = obj_func, dim
        self.bounds = np.array(bounds); self.pop_size=population_size; self.max_iter=max_iter
        self.step_ind, self.step_vol = step_ind, step_vol
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                         size=(self.pop_size,self.dim))
        self.fitness = np.apply_along_axis(self.obj,1,self.positions)
        self.vol = np.ones(self.dim)
        self.history=[np.min(self.fitness)]

    def optimize(self):
        for _ in range(self.max_iter):
            prev_pos = self.positions.copy(); prev_fit = self.fitness.copy()
            # Individual movement
            for i in range(self.pop_size):
                dir = np.random.randn(self.dim)
                new = clip_positions(self.positions[i] + self.step_ind*dir, self.bounds)
                f_new = self.obj(new)
                if f_new < self.fitness[i]:
                    self.positions[i], self.fitness[i] = new, f_new
            # Collective-instinctive movement
            diff = prev_fit - self.fitness
            if np.sum(diff) > 0:
                move = np.sum((self.positions - prev_pos)*diff[:, None], axis=0)/np.sum(diff)
                self.positions += move
                self.positions = clip_positions(self.positions, self.bounds)
            # Volitive movement
            if np.min(self.fitness) < np.mean(diff):
                self.vol -= self.step_vol
            else:
                self.vol += self.step_vol
            bary = np.mean(self.positions, axis=0)
            for i in range(self.pop_size):
                direction = (self.positions[i] - bary)
                norm = np.linalg.norm(direction)
                if norm>0:
                    self.positions[i] += self.vol * direction / norm
                self.positions[i] = clip_positions(self.positions[i], self.bounds)
                self.fitness[i] = self.obj(self.positions[i])
            self.history.append(np.min(self.fitness))
        return self.positions[np.argmin(self.fitness)], self.history

# ---------------------- Moth-Flame Optimization ----------------------
class MFO:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj, self.dim = obj_func, dim
        self.bounds = np.array(bounds); self.pop_size=population_size; self.max_iter=max_iter
        self.moths = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                       size=(self.pop_size, self.dim))
        self.fits = np.apply_along_axis(self.obj,1,self.moths)
        self.flames = self.moths.copy(); self.flame_fits = self.fits.copy()
        self.history=[np.min(self.fits)]

    def optimize(self):
        for t in range(self.max_iter):
            # update number of flames
            flame_no = int(self.pop_size - t*((self.pop_size-1)/self.max_iter))
            idx = np.argsort(self.fits)
            self.flames = self.moths[idx][:flame_no]
            self.flame_fits = self.fits[idx][:flame_no]
            for i, moth in enumerate(self.moths):
                for j in range(self.dim):
                    distance = abs(self.flames[i%flame_no,j] - moth[j])
                    b = 1; t_param = (np.random.rand()*2-1)
                    self.moths[i,j] = distance*np.exp(b*t_param)*np.cos(2*np.pi*t_param) + self.flames[i%flame_no,j]
                self.moths[i] = clip_positions(self.moths[i], self.bounds)
                self.fits[i] = self.obj(self.moths[i])
            best_idx = np.argsort(self.fits)
            self.moths, self.fits = self.moths[best_idx], self.fits[best_idx]
            self.history.append(self.fits[0])
        return self.moths[0], self.history

class MayflyAlgorithm:
    def __init__(self, obj_func, dim, bounds, population_size=40, max_iter=100, alpha=0.5, beta=0.5, delta=0.1, gamma=1.0):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.male = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size//2, self.dim))
        self.female = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size//2, self.dim))
        self.male_v = np.zeros_like(self.male)
        self.female_v = np.zeros_like(self.female)
        self.male_f = np.apply_along_axis(self.obj_func, 1, self.male)
        self.female_f = np.apply_along_axis(self.obj_func, 1, self.female)
        self.best = self.male[np.argmin(self.male_f)]
        self.best_f = np.min(self.male_f)
        self.history = [self.best_f]
    def optimize(self):
        for t in range(self.max_iter):
            # update male velocities & positions
            for i in range(len(self.male)):
                attract = self.alpha * (self.best - self.male[i])
                self.male_v[i] = self.beta * self.male_v[i] + attract + self.delta * np.random.randn(self.dim)
                self.male[i] = np.clip(self.male[i] + self.male_v[i], self.bounds[:,0], self.bounds[:,1])
            self.male_f = np.apply_along_axis(self.obj_func, 1, self.male)
            # update female velocities & positions
            male_best = self.male[np.argmin(self.male_f)]
            for i in range(len(self.female)):
                attract = self.gamma * (male_best - self.female[i])
                self.female_v[i] = self.beta * self.female_v[i] + attract + self.delta * np.random.randn(self.dim)
                self.female[i] = np.clip(self.female[i] + self.female_v[i], self.bounds[:,0], self.bounds[:,1])
            self.female_f = np.apply_along_axis(self.obj_func, 1, self.female)
            # mating (crossover)
            idx_m = np.argsort(self.male_f)
            idx_f = np.argsort(self.female_f)
            for i in range(len(self.male)):
                if np.random.rand() < 0.3:
                    cross = np.random.randint(1, self.dim)
                    child = np.concatenate([self.male[idx_m[i], :cross], self.female[idx_f[i], cross:]])
                    child = np.clip(child, self.bounds[:,0], self.bounds[:,1])
                    if self.obj_func(child) < self.male_f[idx_m[-1]]:
                        self.male[idx_m[-1]] = child
                        self.male_f[idx_m[-1]] = self.obj_func(child)
            self.best = self.male[np.argmin(self.male_f)]
            self.best_f = np.min(self.male_f)
            self.history.append(self.best_f)
        return self.best, self.history

#-----------------Pufferfish Optimization Algorithm (PFA)-------------------------------------
class PFA:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100, a=0.2, b=1.5, c=1.5):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.a = a
        self.b = b
        self.c = c
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.positions[self.best_idx]
        self.best_f = self.fitness[self.best_idx]
        self.history = [self.best_f]
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                rand_idx = np.random.randint(self.pop_size)
                rand = self.positions[rand_idx]
                R = np.random.rand(self.dim)
                F = self.a * (self.best - self.positions[i]) + self.b * (rand - self.positions[i])
                if np.random.rand() < 0.5:
                    new_pos = self.positions[i] + F + self.c * R
                else:
                    new_pos = self.positions[i] - F + self.c * R
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fit = self.obj_func(new_pos)
                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_f:
                self.best = self.positions[idx]
                self.best_f = self.fitness[idx]
            self.history.append(self.best_f)
        return self.best, self.history

#-----------------------Hippopotamus Optimization Algorithm (HOA)-------------------------------------
class HOA:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.positions[self.best_idx]
        self.best_f = self.fitness[self.best_idx]
        self.history = [self.best_f]
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand()
                if r < 0.5:
                    # grazing
                    new_pos = self.positions[i] + r * (self.best - self.positions[i]) + 0.01 * np.random.randn(self.dim)
                else:
                    # searching
                    rand_idx = np.random.randint(self.pop_size)
                    new_pos = self.positions[i] + r * (self.positions[rand_idx] - self.positions[i]) + 0.01 * np.random.randn(self.dim)
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fit = self.obj_func(new_pos)
                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_f:
                self.best = self.positions[idx]
                self.best_f = self.fitness[idx]
            self.history.append(self.best_f)
        return self.best, self.history
#---------------------Arctic Puffin Optimization (APO)---------------------------
class APO:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100, alpha=1.5, beta=0.1):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.positions[self.best_idx]
        self.best_f = self.fitness[self.best_idx]
        self.history = [self.best_f]
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.rand(self.dim)
                move = self.alpha * r * (self.best - self.positions[i]) + self.beta * np.random.randn(self.dim)
                new_pos = self.positions[i] + move
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fit = self.obj_func(new_pos)
                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_f:
                self.best = self.positions[idx]
                self.best_f = self.fitness[idx]
            self.history.append(self.best_f)
        return self.best, self.history
    
#-----------------------Tiki-Taka Algorithm (TTA)-----------------------------------------
class TTA:
    def __init__(self, obj_func, dim, bounds, population_size=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.positions[self.best_idx]
        self.best_f = self.fitness[self.best_idx]
        self.history = [self.best_f]
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                teammate = self.positions[np.random.randint(self.pop_size)]
                pass_vector = (teammate - self.positions[i]) * np.random.rand()
                shot_vector = (self.best - self.positions[i]) * np.random.rand()
                move = pass_vector + shot_vector + 0.01 * np.random.randn(self.dim)
                new_pos = self.positions[i] + move
                new_pos = np.clip(new_pos, self.bounds[:,0], self.bounds[:,1])
                new_fit = self.obj_func(new_pos)
                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_f:
                self.best = self.positions[idx]
                self.best_f = self.fitness[idx]
            self.history.append(self.best_f)
        return self.best, self.history

#-----------------------Constrained Particle Swarm Optimization (CPSO)--------------------------------
class CPSO:
    def __init__(self, obj_func, dim, bounds, constraints, penalty=1e6,
                 population_size=50, max_iter=100, inertia=0.7, c1=1.5, c2=1.5):
        self.obj_func = obj_func
        self.constraints = constraints  # list of constraint functions, each returns True if feasible
        self.penalty = penalty
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w = inertia
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                                           size=(self.pop_size, self.dim))
        v_range = self.bounds[:,1] - self.bounds[:,0]
        self.velocities = np.random.uniform(-v_range, v_range,
                                            size=(self.pop_size, self.dim))
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.apply_along_axis(self._penalized_obj, 1, self.positions)
        gidx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[gidx].copy()
        self.gbest_val = self.pbest_val[gidx]
        self.history = [self.gbest_val]
    def _penalized_obj(self, x):
        if all(con(x) for con in self.constraints):
            return self.obj_func(x)
        else:
            return self.obj_func(x) + self.penalty
    def optimize(self):
        for _ in range(self.max_iter):
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.w*self.velocities 
                               + self.c1*r1*(self.pbest_pos - self.positions)
                               + self.c2*r2*(self.gbest_pos - self.positions))
            self.positions += self.velocities
            self.positions = clip_positions(self.positions, self.bounds)
            fitness = np.apply_along_axis(self._penalized_obj, 1, self.positions)
            mask = fitness < self.pbest_val
            self.pbest_pos[mask] = self.positions[mask]
            self.pbest_val[mask] = fitness[mask]
            gidx = np.argmin(self.pbest_val)
            if self.pbest_val[gidx] < self.gbest_val:
                self.gbest_val = self.pbest_val[gidx]
                self.gbest_pos = self.pbest_pos[gidx].copy()
            self.history.append(self.gbest_val)
        return self.gbest_pos, self.history
