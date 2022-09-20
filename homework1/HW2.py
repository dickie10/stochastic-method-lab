import numpy as np

def pv(C,r,m,n):
    x = 1/(1+(r/m)) 
    ar = np.arange(0,(n*m),1) 
    val = x**ar
    res = C*np.sum(val) 
    return res 

def fv(res,r,m,n):
    x = 1+(r/m) 
    ar = np.arange((n*m),0,-1) 
    val = x**ar  
    val1 = np.sum(val)
    ans = res/val1
    return ans 

res = pv(2000,0.02,12,30)
final = fv(res,0.02,12,40)
print(final)