import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mfd4.gradient_descent import gd, gd_momentum, rosenbrock

# Use headless backend if not set by caller
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"

# Rosenbrock function (classic narrow valley)
f, grad = rosenbrock(a=1.0, b=100.0)

# Precompute contours for background
xs = np.linspace(-2, 2, 400)
ys = np.linspace(-1, 3, 400)
XX, YY = np.meshgrid(xs, ys)
ZZ = (1 - XX)**2 + 100.0*(YY - XX**2)**2

def run_and_thin(stepper, thin=10):
    _, path, _ = stepper()
    if thin > 1:
        path = path[::thin]
    return path

# Paths (tuned so the GIF lengths are reasonable)
path_gd  = run_and_thin(lambda: gd(f, grad, x0=[-1.2, 1.0], lr=1e-3, steps=3500), thin=10)       # ~350 frames
path_mom = run_and_thin(lambda: gd_momentum(f, grad, x0=[-1.2, 1.0], lr=2e-3, beta=0.9, steps=2500), thin=7)  # ~357 frames

def make_anim(path, title, outfile, fps=24):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # background contours
    ax.contour(XX, YY, ZZ, levels=40)

    # pre-create artists
    (line,) = ax.plot([], [], lw=2)
    (pt,)   = ax.plot([], [], "o", markersize=5)

    # nice view limits
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())

    def init():
        line.set_data([], [])
        pt.set_data([], [])
        return line, pt

    def update(i):
        seg = path[: i + 1]
        line.set_data(seg[:, 0], seg[:, 1])
        # IMPORTANT: pass sequences to set_data, not scalars
        pt.set_data([seg[-1, 0]], [seg[-1, 1]])
        return line, pt

    frames = len(path)
    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=frames, interval=int(1000 / fps), blit=True, repeat=False
    )

    # Save as GIF via Pillow
    writer = animation.PillowWriter(fps=fps)
    ani.save(outfile, writer=writer, dpi=160)
    plt.close(fig)

# Generate both GIFs headlessly
make_anim(path_gd,  "Rosenbrock Path — GD (lr=0.001)",              "figures/rosenbrock_gd.gif", fps=24)
make_anim(path_mom, "Rosenbrock Path — Momentum (lr=0.002, β=0.9)", "figures/rosenbrock_momentum.gif", fps=24)
print("Saved:", "figures/rosenbrock_gd.gif", "figures/rosenbrock_momentum.gif")
