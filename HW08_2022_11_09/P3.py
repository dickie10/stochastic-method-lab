import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

def compare(x,y):
    if x > 1 or x  < -1:
        return y 

def bw_mtn(walk,M,N):  #function for the standard browian motion 
    timer = np.cumsum(np.full(M, walk))
    ans = np.random.normal(0,1,size=(N))* np.sqrt(walk)  
    answer = np.cumsum(ans)
    return answer, timer


M = 1000
N = 1000
arr = [] 
walk = 1/N  
path,timer = bw_mtn(walk,M,N)
pather = path.flatten()  
df = pd.DataFrame(pather, columns = ['bm'])  
df['timer'] = pd.DataFrame(timer)  
df = df[(df['bm'] > 1) | (df['bm']< -1)]  
arr = df['timer'].to_list()
mean_lst = np.mean(arr) #taking mean
std_lst = np.var(arr) 
print(mean_lst) 
print(std_lst) 
print(df)