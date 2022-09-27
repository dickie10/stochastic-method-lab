#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:50:31 2022

@author: suyashthapa
"""

import matplotlib.pyplot as plt
import numpy as np 

def fun_pv(irr,c,n,m,F): 
    C = F*c/m 
    val_i = np.arange(1,(n*m)+1) 
    ans = np.sum(C / (1 + irr / m) ** val_i) + (F/ (1 + irr / m) ** (n * m))
    return ans 

coup = [0.02,0.06,0.12] 
irr = 0.06 
F=1000 
m = 2 
n = np.arange(0,21) 

for coup_list in coup: 
        price = [fun_pv(irr,coup_list,year,m,F) for year in n]  
        plt.plot(n, price,label='c ={value}'.format(value=coup_list))  
        
plt.xlabel("Time")
plt.ylabel("Price") 
plt.title("Price vs. Time to Maturity")
plt.legend() 


    
        




