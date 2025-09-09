import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import gd, gd_momentum, quadratic

# Quadratic bowl
f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)

# Contour grid
xs = np.linspace(-4, 6, 300)
ys = np.linspace(-6, 4, 300)
XX, YY = np.meshgrid(xs, ys)
ZZ = (XX - 2.0)**2 + (0.5*YY + 1.0)**2

# Paths
_, path_gd, _  = gd(f, grad, x0=[-3, 3], lr=0.25, steps=120)
_, path_mo, _  = gd_momentum(f, grad, x0=[-3, 3], lr=0.25, beta=0.9, steps=120)

# Plot + save (headless-friendly)
plt.figure()
plt.contour(XX, YY, ZZ, levels=30)
plt.plot(path_gd[:,0], path_gd[:,1], marker='o', linewidth=1, label="GD")
plt.plot(path_mo[:,0], path_mo[:,1], marker='o', linewidth=1, label="Momentum")
plt.legend()
plt.title("Vanilla GD vs Momentum on Quadratic")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("figures/momentum_vs_gd.png", dpi=160, bbox_inches="tight")
