import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import gd, quadratic

# Define a convex bowl
f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)

# Grid for contours
xs = np.linspace(-4, 6, 300)
ys = np.linspace(-6, 4, 300)
XX, YY = np.meshgrid(xs, ys)
ZZ = (XX - 2.0)**2 + (0.5*YY + 1.0)**2

# Run GD
_, path, losses = gd(f, grad, x0=[-3, 3], lr=0.3, steps=200)

# Plot contour + path
plt.figure()
plt.contour(XX, YY, ZZ, levels=30)
plt.plot(path[:, 0], path[:, 1], marker='o', linewidth=1)
plt.title("Gradient Descent Path on Quadratic Loss")
plt.xlabel("x"); plt.ylabel("y")
plt.savefig("figures/quadratic_gd_path.png", dpi=160, bbox_inches="tight")
plt.close()

# Loss vs iteration
plt.figure()
plt.plot(losses)
plt.title("Loss vs Iteration (Quadratic)")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.savefig("figures/quadratic_gd_loss.png", dpi=160, bbox_inches="tight")
plt.close()
