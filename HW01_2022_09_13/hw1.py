import numpy as np  
import time


def looper(c,r): 
    pv = 0
    for x in range(len(c)): 
        pv += c[x]/(1+r)**(x+1) 
    return pv , 'for loop'

def poly_val(c,r):
    val = 1.0/(1.0+r) 
    c = np.flip(c)
    c = np.append(c,0) 
    pv = np.polyval(c,val) 
    return pv , "Polyval function"

def Horn_scheme(c,r):
    x = 1.0/(1.0+r) 
    res = 0 
    c = np.flip(c)
    for i in range(len(c)):
        res = res * x + c[i]
    return res*x, "Horners scheme" 

def dot_product(c,r):
    x = 1/(1+r)
    ar = np.arange(1,len(c)+1,1)
    ar1 = x**ar
    res = np.dot(c,ar1) 
    return res, "Dot product"
    

C = 120 * np.arange(500,1200) 
r = 0.01   
#val = looper(C,r) 
#val_poly = poly_val(C,r)  
#val_horn = Horn_scheme(C,r)  
val_rate = 1.0/(1.0+r) 
value = [(val_rate)** x for x in range(len(C))]
#val_dot = dot_product(C,value)

'Working with time'  
lis_fun = [looper(C,r),poly_val(C,r), Horn_scheme(C,r),dot_product(C,r)] 

for fun_val in lis_fun:
    start_time = time.time()    
    for x in range(1000):
        res,word=fun_val
    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print("#-----------------------------------------------#")  
    print("the value of {words} is {result} ".format(words=word,result=res)) 
    print("the time for {words} is:{timer}".format(words=word,timer=elapsed_time))
    print("#-----------------------------------------------#") 
        