from egwwoa.operators.levy import levy_flight
import numpy as np

def test_levy_shape():
    rng=np.random.default_rng(0); s=levy_flight(4,5,1.5,rng); assert s.shape==(4,5)
