import numpy as np
import matplotlib.pyplot as plt 

def converter(stock,strike_price,opt,position):
    
    zero = np.zeros(len(stock))
    if opt == 'call' and position == 'long':
        return np.fmax(zero, stock - strike_price) # as max(0,s-x)
    elif opt == 'put' and position == 'long': 
        return np.fmax(zero, strike_price - stock) #as max(0,x-s)
    elif opt == 'call' and position == 'short':
        return np.fmin(zero, strike_price - stock) #as min(0,x-s)
    elif opt == 'put' and position == 'short':
        return np.fmin(zero, stock - strike_price) #as min(0,s-x)
    else:
        pass 



stock = np.arange(0, 200)
strike_price = 100
opt = ['call', 'put'] #array for call or put
position = ['long', 'short'] #array for position
i = 0
for opts in opt:
    for pos in  position:
        i += 1
        plt.subplot(len(opt), len(position), i) #subplot for 4 figures
        plt.plot(stock, converter(stock, strike_price, opts, pos))
        plt.title(opts +" and "+ pos)
        plt.ylabel('Payoff value')
        plt.xlabel('Prices')
plt.tight_layout()
plt.show()