import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import gd, gd_momentum, rosenbrock

f, grad = rosenbrock(a=1.0, b=100.0)

# Contour grid (precomputed formula for speed)
xs = np.linspace(-2, 2, 400)
ys = np.linspace(-1, 3, 400)
XX, YY = np.meshgrid(xs, ys)
ZZ = (1 - XX)**2 + 100.0*(YY - XX**2)**2

# Vanilla GD (zig-zaggy)
_, path_gd, _ = gd(f, grad, x0=[-1.2, 1.0], lr=1e-3, steps=3500)

# Momentum GD (faster valley following)
_, path_mo, _ = gd_momentum(f, grad, x0=[-1.2, 1.0], lr=2e-3, beta=0.9, steps=2500)

# Save figures (square + equal aspect)
plt.figure(figsize=(6, 6))
plt.contour(XX, YY, ZZ, levels=40)
plt.plot(path_gd[:,0], path_gd[:,1], marker='o', linewidth=1)
plt.xlim(xs.min(), xs.max()); plt.ylim(ys.min(), ys.max())
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Rosenbrock Path — GD lr=0.001")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("figures/rosenbrock_gd_lr0.001.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()

plt.figure(figsize=(6, 6))
plt.contour(XX, YY, ZZ, levels=40)
plt.plot(path_mo[:,0], path_mo[:,1], marker='o', linewidth=1)
plt.xlim(xs.min(), xs.max()); plt.ylim(ys.min(), ys.max())
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Rosenbrock Path — Momentum lr=0.002, beta=0.9")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("figures/rosenbrock_momentum_lr0.002beta0.9.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()
