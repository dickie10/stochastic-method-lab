import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



"""
    The log return of GBM without noise is not auto-correlated;
    The log return of GBM with Gaussian noise is not auto-correlated when lags > 1, if lags = 1, it shows a slight auto-correlation;
    The log return of GBM with a high frequency perturbation noise is auto-correlated, showing a periodic pattern.
"""

def GBM(N, mu, sigma, seed):
    np.random.seed(seed)
    dt = 1 / N
    S0 = 1
    ds = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(loc=0, scale=np.sqrt(dt), size=N - 1))
    path = S0 * ds.cumprod()
    return np.insert(path, 0, S0)


k = 5
N = 2 ** k
dt = 1 / 2 ** k
mu = 0.3
sigma = 0.5
seed = 5
gbm0 = GBM(N=N + 1, mu=mu, sigma=sigma, seed=seed)
g_noise = np.sqrt(dt) * np.random.normal(0, 1, N + 1)
f = 0.3
p_noise = np.sqrt(dt) * np.sin(2 * np.pi * f * np.arange(N + 1))
gbm1 = gbm0 + g_noise
gbm2 = gbm0 + p_noise
ax0 = plt.subplot(311)
plot_acf(np.diff(np.log(gbm0)), ax=ax0)
ax1 = plt.subplot(312)
plot_acf(np.diff(np.log(gbm1)), ax=ax1)
ax2 = plt.subplot(313)
plot_acf(np.diff(np.log(gbm2)), ax=ax2)
plt.tight_layout()
plt.show()
