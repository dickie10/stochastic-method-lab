import sys
import numpy as np
import matplotlib.pyplot as plt

'''
 The variance of the estimate for µ stay nearly constant as N increases.

'''


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

gbm_v = np.vectorize(GBM, otypes=[list])

def gbm_sigma_mu(k, gbm, N):
    return sme_v(np.array(range(k + 1)), gbm=gbm, N=N)


gsm_v = np.vectorize(gbm_sigma_mu, otypes=[list])


k = 5
N = 2 ** k
mu = 0.3
sigma = 0.5
seeds = np.array(range(3000))
gbms = gbm_v(N=N + 1, mu=mu, sigma=sigma, seed=seeds)
gbms_sigma_mu = np.concatenate(gsm_v(k, gbms, N))
gbms_sigma = gbms_sigma_mu[::2]
gbms_mu = gbms_sigma_mu[1::2]
sigma_mean = np.mean(gbms_sigma, axis=0)
mu_mean = np.mean(gbms_mu, axis=0)
sigma_std = np.std(gbms_sigma, axis=0)
mu_std = np.std(gbms_mu, axis=0)
sigma_var_sqrt = np.sqrt(sigma_mean ** 2 / (2 * 2 ** np.arange(k + 1)))

plt.loglog(2 ** np.arange(k + 1), sigma_mean, '*', label='$\hat{\sigma}$ mean')
plt.loglog(2 ** np.arange(k + 1), mu_mean, 'X', label='$\hat{\mu}$ mean')
plt.loglog(2 ** np.arange(k + 1), sigma_std, 'o', label='$\hat{\sigma}$ std')
plt.loglog(2 ** np.arange(k + 1), mu_std, 'v', label='$\hat{\mu}$ std')
plt.loglog(2 ** np.arange(k + 1), sigma_var_sqrt, 'D', label='$\hat{\sigma}$ std (formula)')
plt.xlabel('$N$')
plt.title('$\hat{\sigma}\ &\ \hat{\mu}$')
plt.legend(loc=3)
plt.show()
