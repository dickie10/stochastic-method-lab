from scipy.stats import norm
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy


def payoff(S, K):
    """
    use payoff
    """
    payoff_data = np.maximum((S - K), 0)
    return payoff_data


def binomialtree(S, K, T, r, sigma, N):
    """
    uses binomial tree
    """
    u = np.exp(sigma*np.sqrt(T/N))
    d = 1/u
    arr = np.zeros(N+1)
    arr[0] = S*np.power(u, N)

    for i in range(1, N+1):
        arr[i] = arr[i-1] * (d/u)
    payoff_arr = payoff(arr, K)

    p = (np.exp(r*(T/N))-d)/(u-d)
    coeff = np.exp(-r*(T/N))
    for i in range(0, N):
        upper_vector = np.delete(payoff_arr, N - i)
        lower_vector = np.delete(payoff_arr, 0)
        payoff_arr = coeff*(p*upper_vector + (1-p)*lower_vector)
    return payoff_arr[0]


def black_scholes_methods(S, K, r, sigma, T):
    """
    uses black scholes method 
    """
    x = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    price = S * norm.cdf(x) - K * (math.e**(-r * T)) * \
        norm.cdf(x-sigma*np.sqrt(T))
    return price



S, K, sigma, r, T = 1, 1.2, 0.5, 0.03, 1

black_scholes_method_value = black_scholes_methods(S, K, r, sigma, T)
N = np.arange(1,100,1)
binomialprice = [abs((binomialtree(S, K, T, r, sigma, n) - black_scholes_method_value)/(binomialtree(S, K, T, r, sigma, n))) for n in N]
plt.plot(np.log(N), np.log(binomialprice),label="blackes scholes method logarithm error")
plt.xlabel("$Capital N$")
plt.ylabel("$Logarithmic\ error\ $")
plt.legend()
plt.tight_layout()
plt.show()



def f(x,m,c):
    return m*x+c

a,b = scipy.optimize.curve_fit(f,np.log(N),np.log(binomialprice))
print(a[0])