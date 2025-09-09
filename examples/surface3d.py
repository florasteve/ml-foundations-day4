import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mfd4.gradient_descent import quadratic

f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)

xs = np.linspace(-4, 6, 120)
ys = np.linspace(-6, 4, 120)
XX, YY = np.meshgrid(xs, ys)
ZZ = (XX - 2.0)**2 + (0.5*YY + 1.0)**2

fig = plt.figure(figsize=(8.5, 6.0))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True, alpha=0.85)
ax.set_title("Quadratic Loss Surface (3D)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("loss")
fig.savefig("figures/quadratic_surface3d.png", dpi=160, bbox_inches="tight", pad_inches=0.3)
plt.show()
