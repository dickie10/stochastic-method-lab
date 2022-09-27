

import numpy as np  
from scipy import optimize 
import timeit 


n = 300
C = 120 * np.arange(10, n + 10) 
P = 15000

def fun_opt(val):
    rate = val**np.arange(1,n+1,1) 
    ans = np.dot(C,rate)  
    ret_val = ans-P
    return ret_val  

def irr(root):
    return 1 / root - 1

def bisection(x0,x1,e):
    step = 1
    condition = True
    while condition: 
        x2 = (x0 + x1)/2
        if fun_opt(x0) * fun_opt(x2) < 0:
            x1 = x2
        else:
            x0 = x2
        
        step = step + 1
        condition = abs(fun_opt(x2)) > e

    print('\nRequired Root is : %0.8f' % x2) 
    irr_value = irr(x2)


def derive(x): 
    c_arr = C[::-1]
    index = np.arange(1, len(C) + 1)[::-1]
    derivative = np.dot(c_arr, index * x ** (index - 1))
    return derivative
    
def newtonRaphson( x ):
    h = fun_opt(x) / derive(x)
    while abs(h) >= 0.0001: 
        h = fun_opt(x)/derive(x)
        x = x - h
     
    print("The value of the root is : ","%.4f"% x) 
    irr_value = irr(x)

def secant(x0,x1,e,N):
    step = 1
    condition = True
    while condition: 
        if fun_opt(x0) == fun_opt(x1):
            print('Divide by zero error!')
            break
        
        x2 = x0 - (x1-x0)*fun_opt(x0)/( fun_opt(x1) - fun_opt(x0) ) 
        x0 = x1
        x1 = x2
        step = step + 1
        
        if step > N:
            print('Not Convergent!')
            break
        
        condition = abs(fun_opt(x2)) > e
    print('\n Required root is: %0.8f' % x2) 
    irr_value = irr(x2)

def brenter():
    root = optimize.brentq(fun_opt,0,1) 
    irr_value = irr(root)
    
def irr(root):
    return 1 / root - 1

starttime = timeit.default_timer()
bisection(0.5, 0.9, 1e-8) 
print("The time difference for bisection :", timeit.default_timer() - starttime)  

starttime = timeit.default_timer() 
newtonRaphson(0.5) 
print("The time difference for newtonRaphson :", timeit.default_timer() - starttime)
  
starttime = timeit.default_timer()
secant(0.5,0.9,0.000001,500)  
print("The time difference for secant :", timeit.default_timer() - starttime)

starttime = timeit.default_timer()
brenter()
print("The time difference for brentq :", timeit.default_timer() - starttime)




