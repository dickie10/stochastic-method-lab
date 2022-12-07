
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(1)


"""
    sigma converges; mu does not.
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
mu = 0.3
sigma = 0.5
seed = 1
gbm = GBM(N=N + 1, mu=mu, sigma=sigma, seed=seed)
sigma_mu = sme_v(np.array(range(k + 1)), gbm=gbm, N=N)
plt.semilogx(2 ** np.arange(k + 1), sigma_mu[0], '+', label='$\sigma$')
plt.semilogx(2 ** np.arange(k + 1), sigma_mu[1], 'o', label='$\mu$')
plt.xlabel('dt')
plt.ylabel('value')
plt.title('Semilogx graph')
plt.legend()
plt.show()



