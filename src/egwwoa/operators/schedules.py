
import math
def omega_cosine(t:int, T_iter:int, omega_max:float, omega_min:float) -> float:
    return omega_min + (omega_max-omega_min)*(1+math.cos(math.pi*(t-1)/T_iter))/2.0
