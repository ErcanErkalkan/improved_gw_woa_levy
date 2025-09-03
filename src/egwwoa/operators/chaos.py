
import numpy as np
def logistic_map_reseed(n:int, D:int, lb:np.ndarray, ub:np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = rng.random((n,D))
    for _ in range(3):
        x = 4*x*(1-x)
    return lb + x*(ub-lb)
