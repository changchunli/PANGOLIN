from __future__ import absolute_import, division, print_function

import numpy as np


def adam(grad, x, m, v, i, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.
    """
    m = (1 - b1) * grad + b1 * m  # First  moment estimate.
    v = (1 - b2) * (grad**2) + b2 * v  # Second moment estimate.

    # mhat = m / (1 - b1**(i + 1))  # Bias correction.
    # vhat = v / (1 - b2**(i + 1))
    # x = x - step_size * mhat / (np.sqrt(vhat) + eps)

    step_size = step_size * np.sqrt(1 - b2**(i + 1)) / (1 - b1**(i + 1))
    x = x - step_size * m / (np.sqrt(v) + eps)

    return x, m, v


def adamax(grad, x, m, v, i, step_size=0.002, b1=0.9, b2=0.999, eps=10**-8):
    m = (1 - b1) * grad + b1 * m
    v = np.maximum(b2 * v, np.abs(grad))

    # mhat = m / (1 - b1**(i + 1))
    # x = x - step_size * mhat / (v + eps)

    step_size = step_size / (1 - b1**(i + 1))
    x = x - step_size * m / (v + eps)

    return x, m, v


def adagrad(grad, x, sum_sq_grad, step_size=0.01, eps=10**-8):
    """Adagrad as described in http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    It adapts the learning rate to the parameters, performing larger updates for
    infrequent and smaller updates for frequent parameters.
    """
    sum_sq_grad = sum_sq_grad + grad**2
    x = x - step_size * grad / (np.sqrt(sum_sq_grad) + eps)

    return x, sum_sq_grad


def adadelta(grad, x, avg_sq_grad, avg_sq_update, gamma=0.9, eps=10**-8):
    """Adadelta as described in https://arxiv.org/pdf/1212.5701.pdf.
    It is an extension of Adagrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    """
    avg_sq_grad = avg_sq_grad * gamma + grad**2 * (1 - gamma)
    update = (np.sqrt(avg_sq_update) + eps) * grad / (
        np.sqrt(avg_sq_grad) + eps)
    x = x - update
    avg_sq_update = avg_sq_update * gamma + update**2 * (1 - gamma)

    return x, avg_sq_grad, avg_sq_update


def nadam(grad, x, m, v, i, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Nadam ad described in `Incorporating Nesterov Momentum into Adam`.
    It combines Adam and NAG.
    """
    m = (1 - b1) * grad + b1 * m
    v = (1 - b2) * (grad**2) + b2 * v
    mhat = m / (1 - b1**(i + 1))
    vhat = v / (1 - b2**(i + 1))
    x = x - step_size * (b1 * mhat + (1 - b1) * grad /
                         (1 - b1**(i + 1))) / (np.sqrt(vhat) + eps)

    return x, m, v


def amsgrad(grad,
            x,
            m,
            v,
            v_hat_prev,
            i,
            step_size=0.001,
            b1=0.9,
            b2=0.999,
            eps=10**-8,
            bias_correction=True):
    """AMSGrad described as in `On the Convergence of Adam and Beyond`.
    It uses the maximum of past squared gradients `v_t` rather than
    the exponential average to update the parameters.
    """

    if bias_correction:
        step_size = step_size * np.sqrt(1 - b2**(i + 1)) / (1 - b1**(i + 1))

    m = (1 - b1) * grad + b1 * m
    v = (1 - b2) * (grad**2) + b2 * v
    v_hat = np.maximum(v_hat_prev, v)
    x = x - step_size * m / (np.sqrt(v_hat) + eps)

    return x, m, v, v_hat


def sgd(grad, x, velocity, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = mass * velocity - (1.0 - mass) * grad
    x = x + step_size * velocity

    return x, velocity


def rmsprop(grad, x, avg_sq_grad, step_size=0.001, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = avg_sq_grad * gamma + grad**2 * (1 - gamma)
    x = x - step_size * grad / np.sqrt(avg_sq_grad + eps)

    return x, avg_sq_grad
