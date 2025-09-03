
import numpy as np
def toy_fitness_factory(T_hor:int=24):
    D = 3*T_hor
    w = np.linspace(1.0, 2.0, D)
    def f(x: np.ndarray)->float:
        return float((w*x**2).sum() + 0.01*np.abs(x).sum())
    return f, D
