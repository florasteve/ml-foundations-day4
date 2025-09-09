import numpy as np

def gd(f, grad_f, x0, lr=0.1, steps=200, tol=1e-8, callback=None):
    """
    Basic gradient descent with trace.
    Returns:
        x (np.ndarray): final point
        trace (np.ndarray): iterates (T+1, d)
        losses (np.ndarray): f(x_t) (T+1,)
    """
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
    """
    f(x,y) = (a*x - b)^2 + (c*y - d)^2
    """
    def f(z):
        x, y = z
        return (a*x - b)**2 + (c*y - d)**2
    def grad(z):
        x, y = z
        return np.array([2*a*(a*x - b), 2*c*(c*y - d)])
    return f, grad

def logistic_loss(X, y, lam=0.0):
    """
    Binary logistic regression negative log-likelihood with L2 (lam) on w (not bias).
    X: (n, d), y: (n,) in {0,1}
    Returns f(theta), grad(theta) where theta = [w1,...,wd,b]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    def f(params):
        w = params[:-1]
        b = params[-1]
        z = X @ w + b
        # stable log(1+exp(z))
        pos = np.maximum(z, 0)
        neg = z - pos
        logexp = pos + np.log1p(np.exp(neg))
        nll = np.sum(logexp - y * z)
        reg = 0.5 * lam * np.sum(w * w)
        return float(nll + reg)

    def grad(params):
        w = params[:-1]
        b = params[-1]
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = X.T @ err + lam * w
        gb = np.sum(err)
        return np.concatenate([gw, np.array([gb])])

    return f, grad
