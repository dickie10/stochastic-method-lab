
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

def get_payoff(S, K): 
        payoff = np.maximum((S - K), 0)
        return payoff 
    
    
def binomial_tree(payoff,S,K,T,r,sigma,N):
    u = np.exp(sigma*np.sqrt(T/N))
    d = 1/u
    p = (np.exp(r*(T/N))-d)/(u-d)  
    coeff = np.exp(-r*(T/N))
    for i in range(0, N):
        upper_vector = np.delete(payoff, N - i)
        lower_vector = np.delete(payoff, 0)
        payoff = coeff*(p*upper_vector + (1-p)*lower_vector)   
    return payoff[0]


def main():
    # all the values to be used to test run the program
    T = 1
    N = 1000
    sigma = 0.5
    r = 0.02
    S = 1
    K = 0.7
    u = np.exp(sigma*np.sqrt(T/N))
    d = 1/u
    
    arr = np.zeros(N+1)    
    arr[0] = S*np.power(u,N)
      
    for i in range(1, N+1):
        arr[i] = arr[i-1] * (d/u)   
    
    payoff = get_payoff(arr, K)

    print("The arbitrage free option price is :", round(binomial_tree(payoff,S,K,T,r,sigma,N), 4))
    
if __name__ == "__main__":
    main()


