
from pylab import *
from numpy import *

def binomial_tree_american_option(K,T,S0,r,N,u,d,opttype='P'):

    dt = T/N
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)
    
    S = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
        

    if opttype == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        C[:i+1] = disc * ( q*C[1:i+2] + (1-q)*C[0:i+1] )
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)
                
    return C[0]

def main():
    S0 = 100      
    K = 100       
    T = 1        
    r = 0.06      
    N = 3        
    u = 1.1       
    d = 1/u      
    
    print("the value of american put option  ")
    print("with Put option :", round(binomial_tree_american_option(K,T,S0,r,N,u,d,opttype='P'), 4))
    print("with Call option :", round(binomial_tree_american_option(K,T,S0,r,N,u,d,opttype='C'), 4))
    
   
if __name__ == "__main__":
    main()