import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import logistic_loss, gd

# Tiny separable dataset in 2D
X = np.array([
    [-2.0,  0.5],
    [-1.0,  1.0],
    [ 0.0,  0.0],
    [ 1.0, -0.5],
    [ 2.0, -1.0],
])
y = np.array([0, 0, 0, 1, 1])

# Build loss/grad on (w1, w2, b)
f, grad = logistic_loss(X, y, lam=0.0)

# Visualize loss over (w1, w2); for each grid point, optimize b quickly
w1s = np.linspace(-6, 6, 120)
w2s = np.linspace(-6, 6, 120)
W1, W2 = np.meshgrid(w1s, w2s)
ZZ = np.zeros_like(W1, dtype=float)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w1 = float(W1[i, j])
        w2 = float(W2[i, j])

        def fb_only(b: float) -> float:
            params = np.array([w1, w2, float(b)], dtype=float)
            return f(params)

        def gb_only(b: float) -> float:
            params = np.array([w1, w2, float(b)], dtype=float)
            return float(grad(params)[-1])  # gradient w.r.t. b

        # One-dimensional GD on b (scalar)
        b0 = 0.0
        lr_b = 0.2
        for _ in range(40):
            b0 = b0 - lr_b * gb_only(b0)
        ZZ[i, j] = fb_only(b0)

# Save surface contour
plt.figure()
plt.contour(W1, W2, ZZ, levels=30)
plt.title("Logistic Regression Loss Surface (w1, w2) with b optimized")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_surface.png", dpi=160, bbox_inches="tight")

# Run full GD on (w1, w2, b) and overlay path projected onto (w1, w2)
x0 = np.array([0.0, 0.0, 0.0])
_, path, _ = gd(f, grad, x0=x0, lr=0.5, steps=200)
plt.figure()
plt.contour(W1, W2, ZZ, levels=30)
plt.plot(path[:,0], path[:,1], marker='o', linewidth=1)
plt.title("GD Path on Logistic Loss (projected onto w1-w2)")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_gd_path.png", dpi=160, bbox_inches="tight")
