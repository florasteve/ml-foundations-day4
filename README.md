# ML Foundations — Day 4: Gradient Descent

Implement gradient descent from scratch in NumPy and visualize loss surfaces (2D/3D), learning-rate effects, convergence, and optimization intuition.

## Status
Project complete – Documentation in progress

## Roadmap
- Vanilla GD on convex/quadratic
- Logistic regression GD
- 2D/3D loss surface plots
- Learning-rate sweep + divergence
- Batch vs stochastic vs mini-batch (preview)

## Setup
~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## Run demos (headless)
~~~bash
export MPLBACKEND=Agg
PYTHONPATH=src python examples/quadratic_contours.py
PYTHONPATH=src python examples/lr_sweep.py
PYTHONPATH=src python examples/surface3d.py
PYTHONPATH=src python examples/logreg_surface.py
~~~

## Results

### Quadratic Loss (Contour + GD Path)
![Quadratic GD Path](figures/quadratic_gd_path.png)

### Loss vs Iteration
![Quadratic GD Loss](figures/quadratic_gd_loss.png)

### Learning Rate Sweep
![LR 0.05](figures/quadratic_lr_0p05.png)
![LR 0.1](figures/quadratic_lr_0p1.png)
![LR 0.3](figures/quadratic_lr_0p3.png)
![LR 0.9](figures/quadratic_lr_0p9.png)
![LR 1.5](figures/quadratic_lr_1p5.png)

### 3D Surface of Quadratic Bowl
![Quadratic 3D Surface](figures/quadratic_surface3d.png)

### Logistic Regression Loss Surface
![Logistic Loss Surface](figures/logreg_surface.png)

### GD Path on Logistic Loss
![Logistic GD Path](figures/logreg_gd_path.png)
