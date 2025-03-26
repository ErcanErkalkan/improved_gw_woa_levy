import numpy as np
from scipy.special import gamma

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

import numpy as np
from scipy.special import gamma

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

