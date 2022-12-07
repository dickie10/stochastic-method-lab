import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(1)


"""
   Sigma and Mu in GBM with Gaussian noise and a periodic noise do not converge to the correct value
"""



def sigma_mu_est(k, gbm, N):
    s_coarse = gbm[::int((1/2**k)*N)]
    r_arr = np.diff(np.log(s_coarse))
    r_mean = np.mean(r_arr)
    if len(r_arr) > 1:
        sigma_r = np.std(r_arr, ddof=1) 
    else:
        sigma_r = 0
    sigma_estimator = sigma_r / np.sqrt(1/2**k)
    mu_estimator = r_mean / (1/2**k) + sigma_estimator ** 2 / 2
    return sigma_estimator, mu_estimator 

def GBM(N, mu, sigma, seed):
    np.random.seed(seed)
    dt = 1 / N
    S0 = 1
    ds = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(loc=0, scale=np.sqrt(dt), size=N - 1))
    path = S0 * ds.cumprod()
    return np.insert(path, 0, S0)


sme_v = np.vectorize(sigma_mu_est, excluded=['gbm'])
k = 5
N = 2 ** k
dt = 1 / 2 ** k
mu = 0.3
sigma = 0.5
seed = 5
gbm0 = GBM(N=N + 1, mu=mu, sigma=sigma, seed=seed)
g_noise = np.sqrt(dt) * np.random.normal(0, 1, N + 1)
f = 0.3
p_noise = np.sqrt(dt) * np.sin(0.02 * np.pi * np.arange(N + 1))
plt.rc('figure', figsize=(10, 5))
plt.subplot(121)
gbm = gbm0 + g_noise
sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)
sigma_est_lst = sigma_mu[0]
mu_est_lst = sigma_mu[1]
plt.semilogx(2 ** np.arange(k + 1), sigma_est_lst, label='$\sigma$')
plt.semilogx(2 ** np.arange(k + 1), mu_est_lst, label='$\mu$')
plt.xlabel('$dt$')
plt.ylabel('value')
plt.title('GBM with Gaussian Noise')
plt.legend()
plt.subplot(122)
gbm = gbm0 + p_noise
sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)
sigma_est_lst = sigma_mu[0]
mu_est_lst = sigma_mu[1]
plt.semilogx(2 ** np.arange(k + 1), sigma_est_lst, label='$\sigma$')
plt.semilogx(2 ** np.arange(k + 1), mu_est_lst, label='$\mu$')
plt.xlabel('$dt$')
plt.ylabel('value')
plt.title('GBM with High Freq Periodic Perturbation')
plt.legend()
plt.tight_layout()
plt.show()