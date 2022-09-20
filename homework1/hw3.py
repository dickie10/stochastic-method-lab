import numpy as np 
import pandas as pd
''' 
P = 500000 
r = 0.02
m = 12
n = 20
'''
def Mont(P,r,m,n): 
    mon = P*((r/m)/ (1-(1+r/m)**(-n*m))) 
    print(mon)  
    df = pd.DataFrame(columns=["Monthy_pay","Interest","Principal_paid","Remaining Principal"])
    payment = np.zeros(m * n + 1)
    interest = np.zeros(m * n + 1)
    Rem_principal = np.ones(m * n + 1) * P
    principal = np.zeros(m * n + 1)
    for x in range(1,n*m+1): 
        Rem_principal[x] = mon * (1 - (1 + r / m) ** (-m * n + x)) / (r / m)
        interest[x] = Rem_principal[x - 1] * r / m
        principal[x] = mon - interest[x] 
        payment[x] = mon
    
    df["Monthy_pay"] = pd.DataFrame(payment) 
    df["Interest"] = pd.DataFrame(interest) 
    df["Remaining Principal"] = pd.DataFrame(Rem_principal) 
    df["Principal_paid"] = pd.DataFrame(principal)  
    df["Interest"]= df["Interest"].apply(lambda x: format(x,"2f"))
    return df
    



final_df = Mont(500000,0.02,12,20) 
print(final_df)