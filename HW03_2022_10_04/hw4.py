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


stock = np.arange(30, 110) 
striker = [50,90,70]
butterfly = converter(stock,striker[0],'call','long') +converter(stock,striker[1],'call','long') + 2*converter(stock,striker[2],'call','short')
plt.plot(stock,butterfly)
plt.xlabel('Stock Price [$]')
plt.ylabel('Payoff [$]')
plt.title('Butterfly Spread')
plt.show()