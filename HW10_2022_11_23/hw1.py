import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_banded
import time






## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    '''
    nf = len(a)     # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
 
    ac[-1] = dc[-1]/bc[-1]
 
    for il in range(nf-2, -1, -1):
        ac[il] = (dc[il]-cc[il]*ac[il+1])/bc[il]
 
    return ac




n = 10
a = np.array([-1 for x in range(1,n)])
b = np.array([2 for x in range(1,n+1)])
c = np.array([-1 for x in range(1,n)])
d =  np.random.rand(10)
print(a) 
print(b) 
print(c) 
print(d)


arr = diags(diagonals=[a, b, c], offsets=[-1,0,1],shape=(n,n)).toarray() #creating an array
print(arr)
x = np.linalg.solve(arr, d)
print("Test results:") 
t0 = time.time()
for i in range(100000):
    TDMAsolver(a, b, c, d)
t1 = time.time()
print("Python method  {}".format(t1-t0)) 

t00 = time.time()
for i in range(100000):
    np.linalg.solve(arr, d)
t11 = time.time()
print("Scipy {}".format(t11-t00))



