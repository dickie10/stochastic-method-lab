import numpy as np
import matplotlib.pyplot as plt

def checker(x): 
    val = (x-np.mean(x)) / np.std(x)  
    return val

N=10000
p = 0.5 

sample1 = np.sort(checker(np.random.binomial(n=N, p=p, size=N))) 
sample2 = np.sort(checker(np.random.normal(loc=N, scale=p, size=N))) 
plt.plot(sample2, sample2, label='Diagonal')
plt.scatter(sample2, sample1, s=5, c='g', label='Q-Q Plot')
plt.title('Q-Q Plot')
plt.xlabel('normal distribution')
plt.ylabel('binomial distribution') 
plt.legend() 
plt.show()
