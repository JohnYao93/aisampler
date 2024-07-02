import jax
import jax.numpy as jnp
import numpy as np


def lag_s_autocorrelation(samples, s, mu, sigma):
    N = samples.shape[0]
    c = 1 / ((sigma**2) * (N - s))
    summand = (samples[s:] - mu) * (samples[:-s] - mu)
    return c * np.sum(summand)


def ess(samples, mu, sigma):
    samples = np.array(samples)
    N = samples.shape[0]

    summ = 0
    for s in range(1, N):
        rho = lag_s_autocorrelation(samples, s, mu, sigma)
        if rho < 0.05:
            break
        summ += (1 - s / N) * rho

    return N / (1 + 2 * summ)


def gelman_rubin_r(chains):

    m = chains.shape[0]
    n = chains.shape[1]
    psi = np.sqrt(np.mean(chains**2, axis=-1))
    psi_bar = np.mean(psi)
    psi_j_bar = np.mean(psi, axis=1)
    B = (n / (m - 1)) * np.sum((psi_j_bar - psi_bar) ** 2)
    W = (1 / (m * (n - 1))) * np.sum(np.sum((psi - psi_j_bar[:, None]) ** 2, axis=-1))
    sigma_hat_squared = (((n - 1) / n) * W) + (B / n)
    V = sigma_hat_squared + (B / (n * m))
    R_hat = V / W
    # R_hat = ((m+1)/m) * (sigma_hat_squared / W) - ((n-1) / (m * n))
    return R_hat


def auto_correlation_time(x, s, mu, var):
    b, t, d = x.shape
    act_ = np.zeros([d])
    for i in range(0, b):
        y = x[i] - mu
        p, n = y[:-s], y[s:]
        act_ += np.mean(p * n, axis=0) / var
    act_ = act_ / b
    return act_


def effective_sample_size(x, mu, var):
    """
    Calculate the effective sample size of sequence generated by MCMC.
    Make sure that `mu` and `var` are correct!

    Args:
        x: np.ndarray, shape=(batch_size, time, dimension)
        mu: np.ndarray, shape=(dimension,)
        var: np.ndarray, shape=(dimension,)
    Returns:
        np.ndarray, shape=(dimension,)
    """
    # batch size, time, dimension
    b, t, d = x.shape
    ess_ = np.ones([d])
    for s in range(1, t):
        p = auto_correlation_time(x, s, mu, var)
        if np.sum(p > 0.05) == 0:
            break
        else:
            for j in range(0, d):
                if p[j] > 0.05:
                    ess_[j] += 2.0 * p[j] * (1.0 - float(s) / t)
    return t / ess_
