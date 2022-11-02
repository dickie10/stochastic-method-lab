import numpy as np
import matplotlib.pyplot as plt   
from scipy.stats import norm  
import math 
from scipy import stats

def monte_carlo(S0,sigma,r,N,T,k):  
    W = np.random.normal(0,1,size=N) #sampling using browian motion  
    xx = (r-((sigma)**2/2)*T+sigma*W) 
    val = -r * T
    ST = S0*np.exp(xx)   
    payoff_call = np.maximum(ST-k,0) 
    return (np.mean(payoff_call)*np.exp(val)) 


def black_scholes_methods(S, K, r, sigma, T):
    """
    uses black scholes method 
    """
    x = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    price = S * norm.cdf(x) - K * (math.e**(-r * T)) * \
        norm.cdf(x-sigma*np.sqrt(T))
    return price

S0 = 1
k = 0.8 
T = 1 
r = 0.3 
sigma = 0.7   
N = 10000
black_scholes = black_scholes_methods(S0,k,r,sigma,T)
M = 1.2**np.arange(10,30,1) 
mean_val = np.array([monte_carlo(S0,sigma,0.3,int(n),T,k)  for n in M])
diff = abs(mean_val - black_scholes)   
slope, intercept, r, p, se = stats.linregress(np.log(M), np.log(diff))   
print("The converange rate is the slope in the log plot") 
print(slope)
plt.loglog(M,diff)  
plt.xlabel("samples") 
plt.ylabel("difference between monte carlo and blackes scholes") 
plt.legend() 
plt.show()