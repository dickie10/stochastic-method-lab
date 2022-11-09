from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime,date 
import matplotlib.pyplot as plt
import numpy as np 
import math 





def x(S, K, r, sigma, T): 
    x = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T)) 
    return x


def call(S, K, r, sigma, T): #blackes scholes method 
    price = S * norm.cdf(x(S, K, r, sigma, T)) - K * (math.e**(-r * T)) * norm.cdf(x(S, K, r, sigma, T)-sigma*np.sqrt(T))
    return price






S = np.linspace(1,5,num=5)
K = np.linspace(1.2,2.4,num=5)
sigma = np.linspace(0.5,1.5,num=5)
T = np.linspace(1,5,num=5)
r = np.linspace(0.03,0.08,num=5) 

S_MIN, S_MAX = 1, 5
R_MIN, R_MAX = 0.01, 0.30
SIGMA_MIN, SIGMA_MAX = 0.1, 0.9
steps = 100
K, T = 1.2, 4
s_sample, r_sample, sigma_sample = 1, 0.2, 0.5 
S_arr = np.linspace(S_MIN, S_MAX, steps)
rate_arr = np.linspace(R_MIN, R_MAX, steps)
sigma_arr = np.linspace(SIGMA_MIN, SIGMA_MAX,steps)
stock_arr = [call(s, K, r_sample, sigma_sample, T) for s in S_arr]
rated_arr = [call(s_sample, K, r, sigma_sample, T) for r in rate_arr]
price_arr = [call(s_sample, K, r_sample, sigma, T) for sigma in sigma_arr]



#below graph code
axis_1 = plt.subplot(1, 3, 1)
axis_1.plot(S_arr, stock_arr)
axis_1.set_title("varying S")
axis_1.set_xlabel("S")
axis_1.set_ylabel("option price")
axis_2 = plt.subplot(1, 3, 2)
axis_2.plot(rate_arr, rated_arr)
axis_2.set_title("varying rate")
axis_2.set_xlabel("rate")
axis_3 = plt.subplot(1, 3, 3)
axis_3.plot(sigma_arr, price_arr)
axis_3.set_title("varying sigma")
axis_3.set_xlabel("sigma")
plt.tight_layout()
plt.show()
