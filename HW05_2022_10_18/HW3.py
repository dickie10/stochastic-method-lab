from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np



def normal_pdf(x):
    """
        Compute the pdf of the Gaussian distribution with mean 0 and var 1
    """ 
    value = 1 / np.sqrt(2* np.pi) * np.e ** (-1/2*(x**2)) 
    return value





def convert_x(n,p): 
    '''
        Convert x corrdinate of binomial distribution
    '''
    q = 1 - p  
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n  
    print(p)
    print((j - n * p) / np.sqrt(n * p * q))
    return (j - n * p) / np.sqrt(n * p * q)


def convert_y(n,p): 
    """
        convert y corrdinate of binomial distribution
    """ 
    j = np.arange(0, n + 1)
    n = np.ones(n + 1) * n
    q = 1 - p
    return np.sqrt(n * p * q) * binom.pmf(j, n, p)


def ploter(ax,n,p):

    ax.plot(convert_x(n,p), convert_y(n,p),'gx',label="binomial")
    ax.plot(convert_x(n,p),normal_pdf(convert_x(n,p)), label="Gaussian")  
    ax.set_title('$n = {}, p = {}$'.format(n, p))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend()  


plt.rc('figure', figsize=(20, 10))

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ploter(ax1, n=10, p=0.5)
ax2 = fig.add_subplot(2, 2, 2)
ploter(ax2, n=10, p=0.2)
ax3 = fig.add_subplot(2, 2, 3)
ploter(ax3, n=100, p=0.5)
ax4 = fig.add_subplot(2, 2, 4)
ploter(ax4, n=100, p=0.2)
fig.tight_layout()

plt.show()