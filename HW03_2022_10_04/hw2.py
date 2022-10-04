import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

'''
while exporting the data from ECB two bonds were found AAA rated bond and normal bond issued G_N_A is the AAA rated bond
and G_N_C is the normal bond. the symbol is represented instrument_FM, This information is found on ECB website.OBS_VALUE is the 
price of the instrument.DATA_TYPE_FM seems to have different products but we are interested only in SR it is the spot rate. The OBS_VALUE 
with the SR on column DATA_TYPE_FM is the corresponding spot rate. reference taken from https://www.datacareer.de/blog/accessing-ecb-exchange-rate-data-in-python/,
https://sdw.ecb.europa.eu/browse.do?node=9691126 

''' 

def converter(lister):
    print(type) 
    if 'M' not in lister and 'Y' in lister: 
        return int(lister[:-1])
    elif 'Y' not in lister and 'M' in lister: 
        return int(lister[:-1])/12
    else: 
        print("here")
        print(lister.split("Y")[1])
        return int(lister.split('Y')[0]) + int(lister.split('Y')[1][:-1]) / 12
    



def pre_process(df,bond_type): 
    df = df[df['INSTRUMENT_FM'].apply(lambda x: x == bond_type)] 
    df = df[df['DATA_TYPE_FM'].str.contains("SR_")] 
    print(df['DATA_TYPE_FM'])
    df['DATA_TYPE_FM'] = df['DATA_TYPE_FM'].map(lambda x: converter(str(x)[3:])) 
    df = df.sort_values(by=['DATA_TYPE_FM'])
    return df







df = pd.read_csv("data.csv") 
df = df[['INSTRUMENT_FM','DATA_TYPE_FM','OBS_VALUE']]  
df_AAA = pre_process(df,'G_N_A') 
df_common = pre_process(df,'G_N_C')  

plt.plot(df_AAA['DATA_TYPE_FM'], df_AAA['OBS_VALUE'], label='AAA bonds')
plt.plot(df_common['DATA_TYPE_FM'], df_common['OBS_VALUE'], label='All bonds')  
plt.title("Spot rate vs maturity year") 
plt.xlabel("Years") 
plt.ylabel("Spot price") 
plt.legend()
plt.show()


