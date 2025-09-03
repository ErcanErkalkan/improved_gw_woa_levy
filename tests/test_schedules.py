from egwwoa.operators.schedules import omega_cosine

def test_omega_range():
    w=omega_cosine(1,50,1.0,0.1); assert 0.1<=w<=1.0
