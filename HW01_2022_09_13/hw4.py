import numpy as np  
from scipy import optimize

P = 50000 
C = 120.0*np.arange(42,52) 
n=10
def fun_opt(val):
    rate = val**np.arange(1,n+1,1) 
    ans = np.dot(C,rate)  
    ret_val = ans-P
    return ret_val

root = optimize.brentq(fun_opt,0,1) 
print(1/root-1)
