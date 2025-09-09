import numpy as np

def gd(f, grad_f, x0, lr=0.1, steps=200, tol=1e-8, callback=None):
    x = np.array(x0, dtype=float)
    trace = [x.copy()]
    losses = [float(f(x))]
    for t in range(1, steps + 1):
        g = np.array(grad_f(x), dtype=float)
        x_new = x - lr * g
        fx_new = float(f(x_new))
        if callback is not None:
            callback(t, x_new, fx_new)
        trace.append(x_new.copy())
        losses.append(fx_new)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, np.vstack(trace), np.array(losses)

def quadratic(a=1.0, b=0.0, c=1.0, d=0.0):
    def f(z):
        x, y = z
        return (a * x - b) ** 2 + (c * y - d) ** 2
    def grad(z):
        x, y = z
        return np.array([2 * a * (a * x - b), 2 * c * (c * y - d)])
    return f, grad
