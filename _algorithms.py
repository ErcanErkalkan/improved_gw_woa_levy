import numpy as np
from scipy.special import gamma
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional
from utils import levy_flight, chaos_reinit, spiral_position


# ---------------------- Base Optimizer ----------------------
class BaseOptimizer(ABC):
    """
    Abstract base class for metaheuristic optimizers.

    Attributes:
        obj_func: Objective function to minimize.
        dim: Dimension of decision vector.
        bounds: ndarray of shape (dim,2) with lower and upper limits.
        pop_size: Number of agents in population.
        max_iter: Maximum number of iterations.
        seed: Random seed for reproducibility.
        positions: Population positions array.
        fitness: Fitness values corresponding to positions.
        history: Best fitness value at each iteration.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 50,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        # initialize population
        self.positions = self._initialize_population()
        self.fitness = self._evaluate(self.positions)
        best = np.min(self.fitness)
        self.history = [best]

    def _initialize_population(self) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return np.random.uniform(lows, highs, size=(self.pop_size, self.dim))

    def _clip(self, positions: np.ndarray) -> np.ndarray:
        return np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        return np.array([self.obj_func(pos) for pos in positions])

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        ...

# ---------------------- Genetic Algorithm ----------------------
class GA(BaseOptimizer):
    """
    Genetic Algorithm with tournament selection, one-point crossover, and mutation.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 50,
        max_iter: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.cr = crossover_rate
        self.mr = mutation_rate

    def _tournament_selection(self, k: int = 3) -> np.ndarray:
        idx = np.random.choice(self.pop_size, k, replace=False)
        best = idx[np.argmin(self.fitness[idx])]
        return self.positions[best].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.cr:
            pt = np.random.randint(1, self.dim)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
            return c1, c2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(self.dim):
            if np.random.rand() < self.mr:
                span = self.bounds[i,1] - self.bounds[i,0]
                individual[i] += np.random.randn() * span * 0.1
        return self._clip(individual)

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection()
                p2 = self._tournament_selection()
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            self.positions = np.array(new_pop)
            self.fitness = self._evaluate(self.positions)
            best = np.min(self.fitness)
            self.history.append(best)
        idx = np.argmin(self.fitness)
        return self.positions[idx], self.history

# ---------------------- Enhanced GWO-WOA Hybrid ----------------------
class GWWOA(BaseOptimizer):
    """
    Hybrid Gray Wolf & Whale Optimization Algorithm with Lévy flights and chaotic escape.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 50,
        max_iter: int = 100,
        levy_prob: float = 0.1,
        chaos_prob: float = 0.1,
        beta: float = 1.5,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.levy_prob = levy_prob
        self.chaos_prob = chaos_prob
        self.beta = beta
        # Initialize alpha, beta, delta leaders
        self._sort_population()

    def _sort_population(self):
        idx = np.argsort(self.fitness)
        self.positions = self.positions[idx]
        self.fitness = self.fitness[idx]
        self.alpha, self.beta_wolf, self.delta = self.positions[:3]

    def _levy(self) -> np.ndarray:
        sigma = (gamma(1+self.beta)*np.sin(np.pi*self.beta/2)/(gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1/self.beta))

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter
            omega_init, omega_final = 0.9, 0.4
            omega = omega_init * (omega_final / omega_init)**(t / self.max_iter)
            for i in range(self.pop_size):
                A = 2*a*np.random.rand(self.dim) - a
                C = 2*np.random.rand(self.dim)
                p = np.random.rand()
                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        # encircling
                        D1 = np.abs(C*self.alpha - self.positions[i])
                        X1 = self.alpha - A*D1
                        D2 = np.abs(C*self.beta_wolf - self.positions[i])
                        X2 = self.beta_wolf - A*D2
                        D3 = np.abs(C*self.delta - self.positions[i])
                        X3 = self.delta - A*D3
                        new = (X1 + X2 + X3)/3
                    else:
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        new = self.positions[i] - A*np.abs(r1*self.positions[i] - r2*self.alpha)
                else:
                    l = np.random.uniform(-1,1,self.dim)
                    D = np.abs(self.alpha - self.positions[i])
                    new = D*np.exp(0.5*l)*np.cos(2*np.pi*l) + self.alpha
                if np.random.rand() < self.levy_prob:
                    new += omega*self._levy()
                if np.random.rand() < self.chaos_prob:
                    chaos = np.random.uniform(self.bounds[:,0], self.bounds[:,1], self.dim)
                    mask = np.random.rand(self.dim) < 0.3
                    new[mask] = chaos[mask]
                self.positions[i] = self._clip(new)
                self.fitness[i] = self.obj_func(self.positions[i])
            self._sort_population()
            self.history.append(self.fitness[0])
        return self.alpha, self.history

# ---------------------- Whale Optimization Algorithm ----------------------
class WOA(BaseOptimizer):
    """
    Whale Optimization Algorithm (WOA) simulating humpback bubble-net hunting.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        idx = np.argmin(self.fitness)
        self.best = self.positions[idx]

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(self.max_iter):
            a = 2 - 2*t/self.max_iter
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A = 2*a*r1 - a
                C = 2*r2
                p = np.random.rand()
                if p < 0.5:
                    if np.all(np.abs(A) < 1):
                        D = np.abs(C*self.best - self.positions[i])
                        new = self.best - A*D
                    else:
                        rand = self.positions[np.random.randint(self.pop_size)]
                        D = np.abs(C*rand - self.positions[i])
                        new = rand - A*D
                else:
                    l = np.random.uniform(-1,1,self.dim)
                    D = np.abs(self.best - self.positions[i])
                    new = D*np.exp(1*l)*np.cos(2*np.pi*l) + self.best
                self.positions[i] = self._clip(new)
                self.fitness[i] = self.obj_func(self.positions[i])
            idx = np.argmin(self.fitness)
            self.best = self.positions[idx]
            self.history.append(self.fitness[idx])
        return self.best, self.history

# ---------------------- Harmony Search (HS) ----------------------
class HS(BaseOptimizer):
    """
    Harmony Search: memory consideration, pitch adjustment, random selection.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        hmcr: float = 0.95,
        par: float = 0.3,
        bw: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        # Harmony memory initialized in Base

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            new_h = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_h[j] = self.positions[np.argmin(self.fitness), j]
                    if np.random.rand() < self.par:
                        new_h[j] += self.bw * np.random.uniform(-1,1)
                else:
                    new_h[j] = np.random.uniform(self.bounds[j,0], self.bounds[j,1])
            new_h = self._clip(new_h)
            f_new = self.obj_func(new_h)
            worst = np.argmax(self.fitness)
            if f_new < self.fitness[worst]:
                self.positions[worst] = new_h
                self.fitness[worst] = f_new
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Particle Swarm Optimization (PSO) ----------------------
class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization with inertia, cognitive and social coefficients.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 50,
        max_iter: int = 100,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        v_range = self.bounds[:,1] - self.bounds[:,0]
        self.vel = np.random.uniform(-v_range, v_range, (self.pop_size, self.dim))
        self.pbest = self.positions.copy()
        self.pf = self.fitness.copy()
        gidx = np.argmin(self.pf)
        self.gbest = self.pbest[gidx]
        self.w, self.c1, self.c2 = inertia, c1, c2

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.vel = (self.w*self.vel + self.c1*r1*(self.pbest - self.positions)
                        + self.c2*r2*(self.gbest - self.positions))
            self.positions += self.vel
            self.positions = self._clip(self.positions)
            self.fitness = self._evaluate(self.positions)
            mask = self.fitness < self.pf
            self.pbest[mask] = self.positions[mask]
            self.pf[mask] = self.fitness[mask]
            gidx = np.argmin(self.pf)
            self.gbest = self.pbest[gidx]
            self.history.append(self.pf[gidx])
        return self.gbest, self.history

# ---------------------- Grey Wolf Optimizer (GWO) ----------------------
class GWO(BaseOptimizer):
    """
    Grey Wolf Optimizer with hierarchical alpha, beta, delta leadership.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        idx = np.argsort(self.fitness)
        self.alpha, self.beta_wolf, self.delta = [self.positions[i] for i in idx[:3]]

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(self.max_iter):
            a = 2 - 2*t/self.max_iter
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = np.abs(C1*self.alpha - self.positions[i])
                X1 = self.alpha - A1*D_alpha
                # beta
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2, C2 = 2*a*r1 - a, 2*r2
                D_beta = np.abs(C2*self.beta_wolf - self.positions[i])
                X2 = self.beta_wolf - A2*D_beta
                # delta
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3, C3 = 2*a*r1 - a, 2*r2
                D_delta = np.abs(C3*self.delta - self.positions[i])
                X3 = self.delta - A3*D_delta
                new = (X1 + X2 + X3)/3
                self.positions[i] = self._clip(new)
                self.fitness[i] = self.obj_func(new)
            idx = np.argsort(self.fitness)
            self.alpha, self.beta_wolf, self.delta = [self.positions[i] for i in idx[:3]]
            self.history.append(self.fitness[idx[0]])
        return self.alpha, self.history

# ---------------------- Bacterial Foraging Optimization (BFO) ----------------------
class BFO(BaseOptimizer):
    """
    Bacterial Foraging Optimization: chemotaxis, reproduction, elimination-dispersal.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        chem_steps: int = 5,
        swim_length: int = 4,
        repro_steps: int = 2,
        elim_disp_steps: int = 2,
        Ped: float = 0.25,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, chem_steps, seed)
        self.S = swim_length
        self.Nre = repro_steps
        self.Ned = elim_disp_steps
        self.Ped = Ped

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.Ned):
            for _ in range(self.Nre):
                for i in range(self.pop_size):
                    health = 0
                    for _ in range(self.S):
                        cost = self.obj_func(self.positions[i])
                        delta = 0.1 * np.random.randn(self.dim)
                        new = self._clip(self.positions[i] + delta)
                        new_cost = self.obj_func(new)
                        if new_cost < cost:
                            self.positions[i] = new
                            health += new_cost
                        else:
                            break
                    health += self.obj_func(self.positions[i])
            # reproduction: sort and split
            idx = np.argsort([self.obj_func(p) for p in self.positions])
            self.positions = self.positions[idx]
            self.positions[self.pop_size//2:] = self.positions[:self.pop_size//2]
            # elimination-dispersal
            for i in range(self.pop_size):
                if np.random.rand() < self.Ped:
                    self.positions[i] = self._initialize_population()[0]
            self.fitness = self._evaluate(self.positions)
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Fish School Search (FSS) ----------------------
class FSS(BaseOptimizer):
    """
    Fish School Search: individual, instinctive, and volitive movements.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        step_ind: float = 0.1,
        step_vol: float = 0.01,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.step_ind = step_ind
        self.step_vol = step_vol
        self.vol = np.ones(self.dim)

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            prev = self.positions.copy()
            prev_fit = self.fitness.copy()
            # individual
            for i in range(self.pop_size):
                direction = np.random.randn(self.dim)
                candidate = self._clip(self.positions[i] + self.step_ind * direction)
                f_new = self.obj_func(candidate)
                if f_new < self.fitness[i]:
                    self.positions[i] = candidate
                    self.fitness[i] = f_new
            # instinctive
            diff = prev_fit - self.fitness
            if diff.sum() > 0:
                move = ((self.positions - prev) * diff[:,None]).sum(axis=0) / diff.sum()
                self.positions += move
                self.positions = self._clip(self.positions)
            # volitive
            if self.fitness.min() < prev_fit.mean():
                self.vol -= self.step_vol
            else:
                self.vol += self.step_vol
            bary = self.positions.mean(axis=0)
            for i in range(self.pop_size):
                dir_v = self.positions[i] - bary
                norm = np.linalg.norm(dir_v)
                if norm > 0:
                    self.positions[i] += self.vol * dir_v / norm
                self.positions[i] = self._clip(self.positions[i])
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(self.fitness.min())
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Pufferfish Optimization Algorithm (PFA) ----------------------
class PFA(BaseOptimizer):
    """
    Pufferfish Optimization: inflation-deflation inspired exploration and exploitation.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        a: float = 0.2,
        b: float = 1.5,
        c: float = 1.5,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.a, self.b, self.c = a, b, c

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                j = np.random.randint(self.pop_size)
                best = self.positions[np.argmin(self.fitness)]
                F = self.a * (best - self.positions[i]) + self.b * (self.positions[j] - self.positions[i])
                if np.random.rand() < 0.5:
                    new = self.positions[i] + F + self.c * np.random.randn(self.dim)
                else:
                    new = self.positions[i] - F + self.c * np.random.randn(self.dim)
                self.positions[i] = self._clip(new)
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Hippopotamus Optimization Algorithm (HOA) ----------------------
class HOA(BaseOptimizer):
    """
    Hippopotamus Optimization: grazing and searching modes.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            best = self.positions[np.argmin(self.fitness)]
            for i in range(self.pop_size):
                r = np.random.rand()
                if r < 0.5:
                    move = r * (best - self.positions[i]) + 0.01 * np.random.randn(self.dim)
                else:
                    j = np.random.randint(self.pop_size)
                    move = r * (self.positions[j] - self.positions[i]) + 0.01 * np.random.randn(self.dim)
                self.positions[i] = self._clip(self.positions[i] + move)
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Arctic Puffin Optimization (APO) ----------------------
class APO(BaseOptimizer):
    """
    Arctic Puffin Optimization: flocking-based exploration.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        alpha: float = 1.5,
        beta: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.alpha, self.beta = alpha, beta

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            best = self.positions[np.argmin(self.fitness)]
            for i in range(self.pop_size):
                move = self.alpha * np.random.rand(self.dim) * (best - self.positions[i])
                move += self.beta * np.random.randn(self.dim)
                self.positions[i] = self._clip(self.positions[i] + move)
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Tiki-Taka Algorithm (TTA) ----------------------
class TTA(BaseOptimizer):
    """
    Tiki-Taka Algorithm: passing and shooting inspired moves.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            best = self.positions[np.argmin(self.fitness)]
            for i in range(self.pop_size):
                teammate = self.positions[np.random.randint(self.pop_size)]
                pass_v = (teammate - self.positions[i]) * np.random.rand(self.dim)
                shot_v = (best - self.positions[i]) * np.random.rand(self.dim)
                move = pass_v + shot_v + 0.01 * np.random.randn(self.dim)
                self.positions[i] = self._clip(self.positions[i] + move)
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.history

# ---------------------- Constrained PSO (CPSO) ----------------------
class CPSO(BaseOptimizer):
    """
    Constrained PSO with penalty for infeasible solutions.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        constraints: List[Callable[[np.ndarray], bool]],
        pop_size: int = 50,
        max_iter: int = 100,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        penalty: float = 1e6,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        v_range = self.bounds[:,1] - self.bounds[:,0]
        self.vel = np.random.uniform(-v_range, v_range, (self.pop_size, self.dim))
        self.constraints = constraints
        self.penalty = penalty
        self.pbest = self.positions.copy()
        self.pf = np.array([self._penalized(x) for x in self.positions])
        gidx = np.argmin(self.pf)
        self.gbest = self.pbest[gidx]
        self.w, self.c1, self.c2 = inertia, c1, c2

    def _penalized(self, x: np.ndarray) -> float:
        val = self.obj_func(x)
        if not all(con(x) for con in self.constraints):
            val += self.penalty
        return val

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.vel = (self.w*self.vel + self.c1*r1*(self.pbest - self.positions)
                        + self.c2*r2*(self.gbest - self.positions))
            self.positions += self.vel
            self.positions = self._clip(self.positions)
            vals = np.array([self._penalized(x) for x in self.positions])
            mask = vals < self.pf
            self.pbest[mask] = self.positions[mask]
            self.pf[mask] = vals[mask]
            gidx = np.argmin(self.pf)
            self.gbest = self.pbest[gidx]
            self.history.append(self.pf[gidx])
        return self.gbest, self.history
# ---------------------- Flower Pollination Algorithm (FPA) ----------------------
class FPA(BaseOptimizer):
    """
    Flower Pollination Algorithm: global and local pollination via Lévy flights.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        p_switch: float = 0.8,
        beta: float = 1.5,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.p = p_switch
        self.beta = beta

    def _levy(self) -> np.ndarray:
        sigma = (gamma(1+self.beta) * np.sin(np.pi*self.beta/2) /
                 (gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1/self.beta))

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        best = self.positions[np.argmin(self.fitness)]
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.p:
                    L = self._levy()
                    new = self.positions[i] + L * (best - self.positions[i])
                else:
                    j, k = np.random.choice(self.pop_size, 2, replace=False)
                    eps = np.random.rand()
                    new = self.positions[i] + eps * (self.positions[j] - self.positions[k])
                self.positions[i] = self._clip(new)
                self.fitness[i] = self.obj_func(self.positions[i])
            best = self.positions[np.argmin(self.fitness)]
            self.history.append(np.min(self.fitness))
        return best, self.history

# ---------------------- Cat Swarm Optimization (CSO) ----------------------
class CSO(BaseOptimizer):
    """
    Cat Swarm Optimization: seeking and tracing modes for smart search.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        mixture_ratio: float = 0.3,
        memory_pool: int = 5,
        c1: float = 2.0,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.mr = mixture_ratio
        self.pool = memory_pool
        self.c1 = c1
        self.velocities = np.zeros_like(self.positions)

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                if np.random.rand() < self.mr:
                    # seeking
                    candidates = np.tile(self.positions[i], (self.pool,1))
                    for j in range(self.pool):
                        idx = np.random.randint(self.dim)
                        candidates[j,idx] += np.random.randn() * 0.1 * (self.bounds[idx,1]-self.bounds[idx,0])
                    fits = [self.obj_func(p) for p in candidates]
                    best = candidates[np.argmin(fits)]
                    self.positions[i] = best
                else:
                    # tracing
                    gbest = self.positions[np.argmin(self.fitness)]
                    r = np.random.rand(self.dim)
                    self.velocities[i] += self.c1 * r * (gbest - self.positions[i])
                    self.positions[i] += self.velocities[i]
                self.positions[i] = self._clip(self.positions[i])
                self.fitness[i] = self.obj_func(self.positions[i])
            self.history.append(np.min(self.fitness))
        best = self.positions[np.argmin(self.fitness)]
        return best, self.history

# ---------------------- Harris Hawks Optimization (HHO) ----------------------
class HHO(BaseOptimizer):
    """
    Harris Hawks Optimization: rapid dives and surprise pounce phases.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.prey = self.positions[np.argmin(self.fitness)]

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(1, self.max_iter+1):
            E1 = 2 * (1 - t/self.max_iter)
            for i in range(self.pop_size):
                E0 = 2*np.random.rand() - 1
                E = E1 * E0
                J = 2 * (1 - np.random.rand())
                if abs(E) >= 1:
                    rand = self.positions[np.random.randint(self.pop_size)]
                    new = rand - np.random.rand() * abs(rand - 2*np.random.rand()*self.positions[i])
                else:
                    if abs(E) < 0.5:
                        new = self.prey - E * abs(J*self.prey - self.positions[i])
                    else:
                        Xm = self.positions.mean(axis=0)
                        new = (self.prey - self.positions[i]) - E * abs(J*Xm - self.positions[i])
                self.positions[i] = self._clip(new)
                f = self.obj_func(self.positions[i])
                if f < self.fitness[i]:
                    self.fitness[i] = f
                if f < self.obj_func(self.prey):
                    self.prey = self.positions[i]
            self.history.append(self.obj_func(self.prey))
        return self.prey, self.history

# ---------------------- Moth-Flame Optimization (MFO) ----------------------
class MFO(BaseOptimizer):
    """
    Moth-Flame Optimization: logarithmic spiral position updates.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 30,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        self.moths = self.positions.copy()
        self.fits = self.fitness.copy()

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(self.max_iter):
            n_flames = int(self.pop_size - t*((self.pop_size-1)/self.max_iter))
            idx = np.argsort(self.fits)
            flames = self.moths[idx][:n_flames]
            for i in range(self.pop_size):
                for j in range(self.dim):
                    D = abs(flames[i % n_flames,j] - self.moths[i,j])
                    l = (np.random.rand()*2 - 1)
                    self.moths[i,j] = D * np.exp(1*l) * np.cos(2*np.pi*l) + flames[i % n_flames,j]
                self.moths[i] = self._clip(self.moths[i])
                self.fits[i] = self.obj_func(self.moths[i])
            best_idx = np.argmin(self.fits)
            self.history.append(self.fits[best_idx])
        return self.moths[best_idx], self.history

# ---------------------- Mayfly Algorithm ----------------------
class MayflyAlgorithm(BaseOptimizer):
    """
    Mayfly Algorithm: mating and swarming for enhanced diversity.
    """
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 40,
        max_iter: int = 100,
        alpha: float = 0.5,
        beta: float = 0.5,
        delta: float = 0.1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter, seed)
        half = pop_size//2
        self.male = self.positions[:half].copy()
        self.female = self.positions[half:pop_size].copy()
        self.mv = np.zeros_like(self.male)
        self.fv = np.zeros_like(self.female)
        self.mf = np.apply_along_axis(obj_func,1,self.male)
        self.ff = np.apply_along_axis(obj_func,1,self.female)
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma

    def optimize(self) -> Tuple[np.ndarray, List[float]]:
        for t in range(self.max_iter):
            best_m = self.male[np.argmin(self.mf)]
            # male update
            for i in range(len(self.male)):
                self.mv[i] = self.beta*self.mv[i] + self.alpha*(best_m - self.male[i]) + self.delta*np.random.randn(self.dim)
                self.male[i] = self._clip(self.male[i] + self.mv[i])
            self.mf = np.apply_along_axis(self.obj_func,1,self.male)
            # female update
            best_m = self.male[np.argmin(self.mf)]
            for i in range(len(self.female)):
                self.fv[i] = self.beta*self.fv[i] + self.gamma*(best_m - self.female[i]) + self.delta*np.random.randn(self.dim)
                self.female[i] = self._clip(self.female[i] + self.fv[i])
            self.ff = np.apply_along_axis(self.obj_func,1,self.female)
            # mating
            male_idx = np.argsort(self.mf)
            female_idx = np.argsort(self.ff)
            for i in range(len(self.male)):
                if np.random.rand() < 0.3:
                    cross = np.random.randint(1,self.dim)
                    child = np.concatenate([self.male[male_idx[i],:cross], self.female[female_idx[i],cross:]])
                    child = self._clip(child)
                    f_val = self.obj_func(child)
                    if f_val < self.mf[male_idx[-1]]:
                        self.male[male_idx[-1]] = child
                        self.mf[male_idx[-1]] = f_val
            best = np.min(np.concatenate([self.mf,self.ff]))
            self.history.append(best)
        idx = np.argmin(np.concatenate([self.mf,self.ff]))
        if idx < len(self.male): return self.male[idx], self.history
        else: return self.female[idx-len(self.male)], self.history

