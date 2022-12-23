import yfinance as yf 
import datetime as dt
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import date 
from datetime import datetime 
from scipy.stats import norm  
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import math 
from scipy import stats


def get_data(stock_name,time_period): 
    #function genrate data from yahoo finance
    data = yf.Ticker(stock_name) 
    get_value = data.history(period=time_period)  
    print(data.options) 
    opt = data.option_chain('2023-02-17').calls #first maturity date 
    print(opt)
    opt2 = data.option_chain('2023-01-20').calls #second maturity date 
    return get_value,opt,opt2



def stand_fun(val,norm=True):  
    #standard normalizing function
    if norm == True:
        val_data = np.random.normal(loc=0,scale=1,size=val-1) 
    else:
        val_data = val  
    ans = (val_data - np.mean(val_data))/np.std(val_data) 
    return np.sort(ans) 

def get_volatility(df,his_diff):  
    #function to get volatility
    vola_est =  np.std(his_diff, ddof=1)/ np.sqrt(1/len(df["Close"]))
    return vola_est 

def cal_time(d1,d2):  
    #function to calculate difference between time
    delta = d2-d1 
    return delta.days/365

def black_scholes_methods(S, K, r, sigma, T):
    """
    uses black scholes method and function calculates black scholes method

    """
    x = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return (S * norm.cdf(x) - K * (math.e**(-r * T)) *  norm.cdf(x-sigma*np.sqrt(T)))


def get_optprice(S,K,r1,vola_esti,T1):   
    #function gets the option price
    opt_price = []
    for k_value in K: 
        opt_price.append(black_scholes_methods(S,k_value,r1,vola_esti,T1)) 
    return opt_price 

ticker = "AAPL"
period = "1y" 
df,mat1,mat2 = get_data(ticker,period) 
df = df.reset_index()
print(df["Close"])  
his_diff = np.diff(np.log(df["Close"]))  
scal_norm = stand_fun(len(df["Close"]),norm=True) 
scal_log = stand_fun(his_diff,norm=False)  
vola_esti = get_volatility(df,his_diff) 
print(vola_esti)
plot_acf(df["Close"], lags=12)
plt.show()
plt.plot(scal_norm, scal_norm, label='Standard Normal')
plt.scatter(scal_norm, scal_log, s=5, label='logged apple stock')
plt.title('Q-Q Plot')
plt.xlabel('scaled normal distribution')
plt.ylabel('scaled log-returns')
plt.legend()
plt.show()   
"""
looking at the QQ plot it seems the stock for APPL duration of 1 year it has thin tails which makes it perfect for Normal distribution.Which will make a 
negligiable deviation,when we took the lag as 12 we found that through autocorrelation that time series is not seasonal,we also see that data are postiviley correlated when we took lag as 12.
The reference was taken from https://towardsdatascience.com/q-q-plots-explained-5aa8495426c0 (QQ-plot) and 
The reference taken for volatility https://www.learnpythonwithrune.org/calculate-the-volatility-of-historic-stock-prices-with-pandas-and-python/, The reference for autocorrelation https://www.alpharithms.com/autocorrelation-time-series-python-432909/, ref to choose lags https://stats.stackexchange.com/questions/140778/what-is-the-best-lag-length-for-auto-correlation

"""

#The interest free rate for US for maturity 2 months and 1 months is  4.05% and 3.7%, as the stock we have is calculated in usd
r1 = 0.045
r2 = 0.037
#calcuting maturating time 
current = date.today() 
print(current)   
d1 = date(2023, 2, 17) 
d2 = date(2023,1,20) 
T1 = cal_time(current,d1) 
T2 = cal_time(current,d2) 
S = df["Close"].iloc[-1] 
#df = df.set_index("Date") 
#print(df["Close"]) 

#for first maturity at 2023-02-17 with 57 days   

print(mat1["strike"])
K = mat1["strike"].iloc[:30].values.tolist() #strike price   
opt_price_list1 = get_optprice(S,K,r1,vola_esti,T1)  
print(opt_price_list1) 
ana_df = pd.DataFrame(opt_price_list1, columns = ["option price black-scholes"]) 
ana_df["mat1_option"] = mat1["lastPrice"].iloc[:30].values.tolist()   
ana_df["strike"] = K 
print(ana_df)

#for second maturity at 2023-01-20  
print(mat2["strike"])
K = mat2["strike"].iloc[:30].values.tolist() #strike price   
opt_price_list2 = get_optprice(S,K,r2,vola_esti,T2)  
ana_df["black_scholes_second_maturity"] = opt_price_list2 
ana_df["mat2_option"] = mat2["lastPrice"].iloc[:30].values.tolist()   
ana_df["strike2"] = K 
print(opt_price_list2) 

#ploting to see how does black scholes behave with option prices
plt.subplot(1, 2, 1)
plt.scatter(ana_df["strike"], ana_df["mat1_option"], label="option_Price")
plt.scatter(ana_df["strike"], ana_df["option price black-scholes"], label='black-scholes')
plt.title('First maturity')
plt.xlabel('strike-price')
plt.ylabel('price-values')
plt.legend() 
plt.subplot(1, 2, 2)
plt.scatter(ana_df["strike"], ana_df["mat2_option"], label="option_Price")
plt.scatter(ana_df["strike"], ana_df["black_scholes_second_maturity"], label='black-scholes')
plt.title('Second maturity')
plt.xlabel('strike-price')
plt.ylabel('price-values')
plt.legend() 
plt.show()  
"""
 From the plot we see that for both the maturity and 30 strike prices the black-scholes overall matches the opton price, that could be because of implied volatilty that we used, deviations might be found a little if the period of stock
 is increased.
"""

"""
Black Scholes Model

Advantages:
1) Black Scholes Model is very flexible it is used in not only stocks but also in bonds,commodities etc.
2) Black Scholes Model can price the options contracts very quickly
3) Black Scholes Model takes time to expiration, volatility, interest rates, and other important variables which in turn leads to better or accurate pricing of options prices

Disadvantages:
1) There will be mismatch between the pricing of options at expiration for American style options
2) Many important factors are ignored like dividends,interest rates, change in volatility etc.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Finite Difference method 

Advantages: 
1) It can used for both European and American style options 
2) FDM does not take any precautions for convergence

Disadvantages:
1) It is computationally time consuming as more variables are taken

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Binomial Tree method: 

Advantages: 
1) Binomial Option is useful for American options, where the holder has the right to exercise at any time until expiration
2) The model is mathematically simple to calculate

Disadvantage: 
1) A notable disadvantage is that the computational complexity rises a lot in multi-period models 
2) Creating an accurate binomial model tree for a single stock option consumes vast amounts of time

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Monte-Carlo Simulation: 

Advantages: 
1) Monte-Carlo simulation tackles the payoff issues in which the derivative depends on different market varibales
2) It is very flexible to price the path dependent options

Disadavanatges: 
1) Early exercise oppurtunites are not handled well in Monte carlo simulations
2) It is a time-consuming process. It is because it requires the generation of a huge amount of sampling so as to retrieve the desired output.


"""






