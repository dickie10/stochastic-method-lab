import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 





def bw_mtn(T, n):
    walk = T/n
    dW = norm.rvs(0, 1, n) * np.sqrt(walk)
    W = np.cumsum(dW)
    return W

  


def GBM(Bm,mu,sigma,S,T,N):
    t = np.linspace(0,T,N) 
    return S * np.exp((mu-(sigma**2)/2)*t + sigma * Bm)  

def euler(bm,sigma,mu,S,T,n):
    s = np.zeros(n) 
    s[0] = S 
    for i in range(n-1):
        s[i + 1] = s[i] + s[i]*((T/n) * mu + sigma * (bm[i+1]-bm[i])) 
    return s 

def coster(x,t): 
    return (1+t)** 2 * np.cos(x) 

mu,sigma,S, T = 0.5,3,1,1 
N = 1000 
t = np.linspace(0,T,N) 
Bm = bw_mtn(T, N)
Geometric_Bm = GBM(Bm,mu,sigma,S,T,N) 
Euler = euler(Bm,sigma,mu,S,T,N)
sol = coster(Geometric_Bm,t) 

plt.plot(t, Euler, 'g', label="Euler-Mouryana")
plt.plot(t, sol, 'r', label="True solution")
plt.ylabel("S")
plt.xlabel("time")
plt.title("Values")
plt.tight_layout()
plt.legend()
plt.show()
