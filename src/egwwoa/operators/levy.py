
import numpy as np, math
def levy_flight(n:int, D:int, beta_L:float, rng: np.random.Generator) -> np.ndarray:
    sigma_u = ((math.gamma(1+beta_L)*math.sin(math.pi*beta_L/2)) /
               (math.gamma((1+beta_L)/2)*beta_L*2**((beta_L-1)/2))) ** (1/beta_L)
    u = rng.normal(0, sigma_u, size=(n,D))
    v = rng.normal(0, 1, size=(n,D))
    return u / (np.abs(v) ** (1/beta_L))
