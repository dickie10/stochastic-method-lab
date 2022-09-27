

import numpy as np 
from scipy.optimize import brentq 

def fun_pv(irr,c,p,n,m): 
    F = 1 
    C = F*c/m 
    val_i = np.arange(1,(n*m)+1) 
    ans = np.sum(C / (1 + irr / m) ** val_i) + (F/ (1 + irr / m) ** (n * m)) - F*p
    return ans 

c = 0.1
m = 2 
n = 10
p = 0.75

irr = brentq(fun_pv,0,1,args=(c,p,n,m)) 
print(irr) 