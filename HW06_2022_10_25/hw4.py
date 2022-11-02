import numpy as np
import matplotlib.pyplot as plt


def bw_mtn(walk,N):  #function for the standard browian motion
    ans = np.random.normal(0,1,size=N)* np.sqrt(walk)  
    answer = np.cumsum(ans)
    return answer 


N = 10000  
T = 1
t = np.linspace(0,T,N) 
walk = 1/N  
path = bw_mtn(walk,N)

diff = path[1:]-path[:-1]#taking each difference between each values in vector

value = path[:-1] * diff   
ITO = np.cumsum(value)

sum1 = (path[:-1]+path[:1])/2 
value1 = sum1*diff
Sta = np.cumsum(value1) #Stratonovich integrals  

inte_diff = Sta - ITO #difference

plt.plot(np.linspace(0, T, N), path, label='Brownian Motion')
plt.plot(np.linspace(0, T, N-1), ITO, label='Ito Integral')
plt.plot(np.linspace(0, T, N-1), Sta, label='Stratonovich Integral')  
plt.legend()  
plt.show()  

