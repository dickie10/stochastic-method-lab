import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import yfinance as yf
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta 
from math import *

N_prime = norm.pdf
N = norm.cdf


def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call

def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    #see hull derivatives chapter on greeks for reference
    vega = S * N_prime(d1) * np.sqrt(T)
    return vega



def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    
    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            break

        ### use newton rapshon to update the estimate 
        if vega(S, K, T, r, sigma) != 0:
            sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma




def option_checker(input_value): 
    if input_value == "Call":
        call_option= st.option_chain(next_year).calls
        K = call_option["strike"]  
        obs_price = call_option["lastPrice"]
    elif input_value == "Put":
        put_option= st.option_chain(next_year).puts
        K = put_option["strike"]  
        obs_price = put_option["lastPrice"]
    else: 
        print("Enter correct option") 
        return None,None,False
    return K,obs_price,True






T, r = 1,0.0378
dater = datetime.date(2022,6,16)
next_year = (dater+relativedelta(years=1)). strftime("%Y-%m-%d")
print(next_year) 
st = yf.Ticker("CRM") 
S0 = st.history(period="1d")["Close"][0]   
while True:

    print('Enter either Call or Put:') 
    input_val = input()  
    K,obs_price,bool = option_checker(input_val) 
    if bool == True: 
        break 

sigma = np.zeros(len(K))
sigma = [implied_volatility_call(obs_price[i],S0, K[i], T, r) for i in range(len(K))]
print(sigma)
plt.plot(K, sigma, 'r')
plt.ylabel("Implied volatility")
plt.xlabel("Strike Price")
plt.title("Implied Volatility vs. Strike Price")
plt.legend()
plt.show()

