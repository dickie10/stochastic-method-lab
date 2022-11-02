import numpy as np
import matplotlib.pyplot as plt 


def bw_mtn(walk,M,N):  #function for the standard browian motion
    ans = np.random.normal(0,1,size=(M,N))* np.sqrt(walk)  
    answer = np.cumsum(ans,axis=1)
    return answer

 
M = 1000
N = 600 
walk = 1/N  
steps = np.linspace(0, 1, N)  #steps to plot graph for 10 samples
path = np.asarray(bw_mtn(walk,M,N))  #making sure that matrix is in arrays
mean_lst = np.mean(path, axis=0) #taking mean
std_lst = np.std(path, axis=0)  #taking standard deviation
plt.plot(steps, mean_lst, c='green', label='empirical mean')
plt.plot(steps,std_lst, c='yellow', label='empirical standard deviation')
samples = 10  
print(path[0])
for pather in enumerate(path[0:samples - 1]):  
        plt.plot(steps, pather[1], c='blue')


plt.xlabel('time in years')
plt.ylabel('moves')
plt.legend()  
plt.show()