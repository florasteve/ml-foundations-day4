import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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

f, grad = logistic_loss(X, y, lam=0.0)

def newton_opt_b(w1: float, w2: float, b0: float = 0.0, iters: int = 25):
    """Optimize scalar bias b for fixed w=(w1,w2) using 1-D Newton steps."""
    w = np.array([w1, w2], dtype=float)
    b = float(b0)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g  = float(np.sum(p - y))                       # dNLL/db
        g2 = float(np.sum(p * (1.0 - p)) + 1e-12)       # d^2NLL/db^2
        b -= g / g2
        b = float(np.clip(b, -20.0, 20.0))
    return b

# Grid
w1s = np.linspace(-6, 6, 160)
w2s = np.linspace(-6, 6, 160)
W1, W2 = np.meshgrid(w1s, w2s)
ZZ = np.empty_like(W1, dtype=float)

# Loss surface with b optimized via Newton
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w1 = float(W1[i, j]); w2 = float(W2[i, j])
        b_star = newton_opt_b(w1, w2, b0=0.0, iters=25)
        ZZ[i, j] = f(np.array([w1, w2, b_star], dtype=float))

# Replace NaNs/Infs and compute contrast range
ZZ = np.nan_to_num(ZZ, posinf=np.finfo(float).max/10, neginf=0.0)
zmin, zmax = float(np.min(ZZ)), float(np.max(ZZ))
print(f"logreg surface raw range: min={zmin:.3f}, max={zmax:.3f}")

# Percentile-based contrast so mid-range structure is visible
vmin = float(np.percentile(ZZ, 5))
vmax = float(np.percentile(ZZ, 95))
if vmax - vmin < 1e-6:
    vmin, vmax = zmin, zmax  # fallback if too narrow
levels = np.linspace(vmin, vmax, 30)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Filled contours + isolines + colorbar (square + equal aspect)
plt.figure(figsize=(7, 6))
cf = plt.contourf(W1, W2, ZZ, levels=levels, norm=norm, cmap="viridis")
cs = plt.contour(W1, W2, ZZ, levels=levels, colors="k", linewidths=0.4, alpha=0.6)
plt.clabel(cs, inline=1, fontsize=7, fmt="%.1f")
plt.colorbar(cf, shrink=0.85, label="loss")
plt.xlim(w1s.min(), w1s.max()); plt.ylim(w2s.min(), w2s.max())
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Logistic Regression Loss Surface (b optimized via Newton)")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_surface.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()

# Projected GD path on the same surface
x0 = np.array([0.0, 0.0, 0.0])
_, path, _ = gd(f, grad, x0=x0, lr=0.5, steps=200)

plt.figure(figsize=(7, 6))
plt.contour(W1, W2, ZZ, levels=levels, norm=norm, colors="k", linewidths=0.6, alpha=0.7)
plt.plot(path[:,0], path[:,1], marker='o', linewidth=1)
plt.xlim(w1s.min(), w1s.max()); plt.ylim(w2s.min(), w2s.max())
plt.gca().set_aspect("equal", adjustable="box")
plt.title("GD Path on Logistic Loss (projected onto w1-w2)")
plt.xlabel("w1"); plt.ylabel("w2")
plt.savefig("figures/logreg_gd_path.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.show()
