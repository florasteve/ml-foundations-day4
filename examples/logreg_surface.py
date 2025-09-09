import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import logistic_loss, gd

# Tiny 2D dataset
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

def newton_opt_b(w1: float, w2: float, b0: float = 0.0, iters: int = 25):
    """
    Optimize scalar bias b for fixed w=(w1,w2) using 1-D Newton steps:
      g(b)  = sum(sigmoid(Xw + b) - y)
      g'(b) = sum(p*(1-p))
    """
    w = np.array([w1, w2], dtype=float)
    b = float(b0)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g  = float(np.sum(p - y))
        g2 = float(np.sum(p * (1.0 - p)) + 1e-12)
        b -= g / g2
        b = float(np.clip(b, -20.0, 20.0))
    return b

# Grid
w1s = np.linspace(-6, 6, 160)
w2s = np.linspace(-6, 6, 160)
W1, W2 = np.meshgrid(w1s, w2s)
ZZ = np.empty_like(W1, dtype=float)

# Compute loss surface with b optimized via Newton
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w1 = float(W1[i, j]); w2 = float(W2[i, j])
        b_star = newton_opt_b(w1, w2, b0=0.0, iters=25)
        ZZ[i, j] = f(np.array([w1, w2, b_star], dtype=float))

# Sanity: replace any NaNs/Infs; print range
ZZ = np.nan_to_num(ZZ, posinf=np.finfo(float).max/10, neginf=0.0)
zmin, zmax = float(np.min(ZZ)), float(np.max(ZZ))
print(f"logreg surface range: min={zmin:.4f}, max={zmax:.4f}")

# === Figure 1: FILLED contours ONLY (no isoline rings) ===
plt.figure(figsize=(7, 6))
cf = plt.contourf(W1, W2, ZZ, levels=30, cmap="viridis")
plt.colorbar(cf, shrink=0.85, label="loss")
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Logistic Regression Loss Surface (b optimized via Newton)")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_surface.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()

# === Figure 2: GD path projected on w1-w2 (keep thin contour lines here for context) ===
x0 = np.array([0.0, 0.0, 0.0])
_, path, _ = gd(f, grad, x0=x0, lr=0.5, steps=200)
plt.figure(figsize=(7, 6))
try:
    cs = plt.contour(W1, W2, ZZ, levels=30, colors="k", linewidths=0.5, alpha=0.6)
except Exception:
    cs = plt.contour(W1, W2, ZZ, levels=5, colors="k", linewidths=0.5, alpha=0.6)
plt.plot(path[:,0], path[:,1], marker='o', linewidth=1)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("GD Path on Logistic Loss (projected onto w1-w2)")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_gd_path.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()
