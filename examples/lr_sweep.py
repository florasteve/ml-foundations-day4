import sys, pathlib
# Add the repo root so "from src.gradient_descent ..." works regardless of CWD
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from src.gradient_descent import gd, quadratic

f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)

xs = np.linspace(-4, 6, 200)
ys = np.linspace(-6, 4, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = (XX - 2.0)**2 + (0.5*YY + 1.0)**2

for lr in [0.05, 0.1, 0.3, 0.9, 1.5]:
    _, path, _ = gd(f, grad, x0=[-3, 3], lr=lr, steps=80)
    plt.figure()
    cs = plt.contour(XX, YY, ZZ, levels=30)
    plt.plot(path[:, 0], path[:, 1], marker='o', linewidth=1)
    plt.title(f"GD Path (lr={lr})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"figures/quadratic_lr_{str(lr).replace('.','p')}.png", dpi=160, bbox_inches="tight")
    plt.show()
