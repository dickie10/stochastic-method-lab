import numpy as np
import matplotlib.pyplot as plt

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

np.random.seed(1)
sme_v2 = np.vectorize(sigma_mu_est)


k = 5
N = 2 ** k
mu = 0.3
sigma = 0.5
seed = 1
gbm = GBM(N=N + 1, mu=mu, sigma=sigma, seed=seed)
sigma_est, mu_est = sigma_mu_est(k, gbm=gbm, N=N)
seeds = range(1000)
gbms = gbm_v(N=N + 1, mu=mu_est, sigma=sigma_est, seed=seeds)
sigma_mu_est_lst = sme_v2(k, gbm=gbms, N=N)
sigma_est_lst = sigma_mu_est_lst[0]
mu_est_lst = sigma_mu_est_lst[1]
plt.rc('figure', figsize=(14, 10))
plt.subplot(121)
plt.hist(sigma_est_lst, 40)
plt.axvline(x=np.mean(sigma_est_lst), linestyle='--', color='y', label='$\sigma$ mean')
plt.axvline(x=np.mean(sigma_est_lst + np.std(sigma_est_lst)), linestyle='--', color='c', label='$\sigma$ std')
plt.axvline(x=np.mean(sigma_est_lst - np.std(sigma_est_lst)), linestyle='--', color='c')
plt.axvline(x=sigma, linestyle='--', color='r', label='$\sigma$ true')
plt.xlabel('$\sigma$')
plt.legend()
plt.subplot(122)
plt.hist(mu_est_lst, 40)
plt.axvline(x=np.mean(mu_est_lst), linestyle='--', color='y', label='$\mu$ mean')
plt.axvline(x=np.mean(mu_est_lst + np.std(mu_est_lst)), linestyle='--', color='c', label='$\mu$ std')
plt.axvline(x=np.mean(mu_est_lst - np.std(mu_est_lst)), linestyle='--', color='c')
plt.axvline(x=mu, linestyle='--', color='r', label='$\mu$ true')
plt.xlabel('$\mu$')
plt.legend()
plt.tight_layout()
plt.show()