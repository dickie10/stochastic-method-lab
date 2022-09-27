import numpy as np
import matplotlib.pyplot as plt


def fun_pv(irr,n,F): 
    ans = F*(1+irr)**(n-15) 
    return ans 

n = np.arange(0,16) 
F = 1
m = 2  
c = 0.08
coup_list = [0.06,0.08,0.1] 

for yeild in coup_list: 
         price = [fun_pv(yeild,year,F) for year in n]  
         plt.plot(n, price,label='c ={value}'.format(value=yeild))  
         
plt.xlabel("Maturity years")
plt.ylabel("Forward value") 
plt.title("Forward value vs. Time to maturity")
plt.legend() 