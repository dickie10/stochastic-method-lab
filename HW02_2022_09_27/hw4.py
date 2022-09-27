import matplotlib.pyplot as plt
import numpy as np 

def fun_pv(irr,c,n,m,F): 
    C = F*c/m 
    val_i = np.arange(1,(n*m)+1) 
    ans = np.sum(C / (1 + irr / m) ** val_i) + (F/ (1 + irr / m) ** (n * m))
    return ans 

coup = np.arange(0,1,0.01) 
c = 0.08 
F=1000 
m = 1
n = 10 


price = [fun_pv(val,c,n,m,F) for val in coup]  
plt.plot(coup, price)  
        
plt.xlabel("Yields")
plt.ylabel("Price") 
plt.title("Price of bonds vs. Yields")
plt.legend() 