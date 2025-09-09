import numpy as np
from mfd4.gradient_descent import gd, quadratic

def test_gd_converges_on_quadratic():
    f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)
    _, path, losses = gd(f, grad, x0=[-3, 3], lr=0.3, steps=150)
    # monotone-ish last few losses and near the optimum
    assert losses[-1] <= losses[-5]
    assert np.linalg.norm(path[-1] - np.array([2.0, -2.0])) < 1e-2
