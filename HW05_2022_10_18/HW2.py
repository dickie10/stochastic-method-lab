import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial 


def striling(x):
    val = np.sqrt(2* np.pi *x) * (x / np.e) **x 
    return val  

def relative_error(val1,val2):
    return (val1 - val2)/ val2

N = 100 
lister = np.arange(1, N+1)
stirling_lister = striling(lister)
factorial_lister = factorial(lister)
error_list = relative_error(factorial_lister, stirling_lister)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Relative Error vs. N")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("n")
ax.set_ylabel("Relative Error")
ax.plot(lister, error_list,'*')
ax.plot(lister,1./(12*lister), label = 'line 1/(12n)')
plt.legend()
plt.show() 
