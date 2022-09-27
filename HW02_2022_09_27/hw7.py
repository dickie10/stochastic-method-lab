import matplotlib.pyplot as plt
import numpy as np  
from scipy.optimize import brent

def fun_pv(irr,c,n,m,F,horizon): 
    C = F*c/m 
    val_i = np.arange(1,(n*m)+1) 
    ans = np.sum(C / (1 + irr / m) ** val_i) + (F/ (1 + irr / m) ** (n * m))
    return ans*(1+irr/m)**(horizon*m) 

coup =  np.arange(0, 0.3,0.01)
c = 0.1
F=1
m = 1
n = 30 
horizon=10

price = [fun_pv(val,c,n,m,F,horizon) for val in coup]  
yeild = brent(fun_pv,args=(c,n,m,F,horizon)) 
yeild_min = fun_pv(yeild,c,n,m,F,horizon)
        
plt.plot(coup, price)
plt.xlabel("Yield")
plt.ylabel("Horizon Price")
plt.title("Horizon Price vs. Yield")
plt.scatter(yeild, yeild_min, marker='x')
plt.show() 
print("When yield = {value} and par value = 1, minimum horizon price = {data}".format(value=yeild, data=yeild_min))