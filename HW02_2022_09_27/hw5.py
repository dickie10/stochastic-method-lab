import numpy as np
import matplotlib.pyplot as plt

def volatility(n,m,y,F,c):
    C = F*c/m   
    ans =  -((C / y) * n * m - (C / (y ** 2)) * ((1 + y) ** (n * m + 1) - (1 + y)) - n * m * F) /((C / y) * ((1 + y) ** (n * m + 1) - (1 + y)) + F * (1 + y))
    return ans


n = np.arange(0,100) 
irr = 0.06/2
F = 1000  
m = 2 
coup_list = [0.02,0.06,0.12] 

for c in coup_list: 
        price = [volatility(year, m, irr, F, c) for year in n]  
        plt.plot(n, price,label='c ={value}'.format(value=c))  
        
plt.xlabel("Volatility")
plt.ylabel("time") 
plt.title("Volatility vs. Time to Maturity")
plt.legend() 