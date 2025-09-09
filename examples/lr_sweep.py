import numpy as np
import matplotlib.pyplot as plt
from mfd4.gradient_descent import gd, quadratic

# Quadratic bowl
f, grad = quadratic(a=1.0, b=2.0, c=0.5, d=-1.0)

# A baseline grid range (used if paths stay inside it)
BASE_X = (-4.0, 6.0)
BASE_Y = (-6.0, 4.0)

def quad_surface(xmin, xmax, ymin, ymax, n=200):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = (XX - 2.0)**2 + (0.5*YY + 1.0)**2
    return xs, ys, XX, YY, ZZ

def trim_path_for_display(path, limit=30.0, fallback_n=30):
    """
    Keep the path until it leaves a reasonable window (±limit).
    If it leaves immediately, show the first fallback_n points.
    """
    mask = (np.abs(path[:, 0]) <= limit) & (np.abs(path[:, 1]) <= limit)
    if np.any(mask):
        last_ok = np.where(mask)[0].max()
        return path[: last_ok + 1]
    return path[: min(fallback_n, len(path))]

def dynamic_bounds(path, base_x=BASE_X, base_y=BASE_Y, pad=0.08):
    """
    Compute axis limits that include both the baseline window and the (trimmed) path.
    Add a small padding so points aren’t on the edge.
    """
    x_min = min(base_x[0], np.nanmin(path[:, 0]))
    x_max = max(base_x[1], np.nanmax(path[:, 0]))
    y_min = min(base_y[0], np.nanmin(path[:, 1]))
    y_max = max(base_y[1], np.nanmax(path[:, 1]))

    dx = x_max - x_min or 1.0
    dy = y_max - y_min or 1.0
    x_min -= pad * dx; x_max += pad * dx
    y_min -= pad * dy; y_max += pad * dy
    return float(x_min), float(x_max), float(y_min), float(y_max)

# Run the sweep
for lr in [0.05, 0.1, 0.3, 0.9, 1.5]:
    _, path, _ = gd(f, grad, x0=[-3, 3], lr=lr, steps=80)

    # For divergent cases, trim the view to the informative region
    shown = trim_path_for_display(path, limit=30.0, fallback_n=40)

    # Autoscale around the (trimmed) path + baseline window
    x_min, x_max, y_min, y_max = dynamic_bounds(shown)

    # Build a contour surface over the chosen window (so background isn’t blank)
    _, _, XX, YY, ZZ = quad_surface(x_min, x_max, y_min, y_max, n=220)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.contour(XX, YY, ZZ, levels=30)
    plt.plot(shown[:, 0], shown[:, 1], marker='o', linewidth=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"GD Path (lr={lr})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(f"figures/quadratic_lr_{str(lr).replace('.','p')}.png",
                dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.show()
