__doc__ = """ Code for question 1(c). """

from MCMC import MCMC
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def func1(x, xi, delta, C):
    return delta * np.exp(-x / xi) + C

## Initiliaztion
N, q = 16, 3

Ts = np.linspace(.95, 1.2, 26)
ksis = []
# Outer Loop
for T in tqdm(Ts):
    beta = 1. / T
    tmp = MCMC(beta, 0., q, N, 
        [lambda sigma, H, k=k: 
            np.sum(
                sigma * (sigma[:, np.arange(k, N + k) % N] + sigma[np.arange(k, N + k) % N, :])
            )
         for k in range(9)], [1000000, 10000]) / (2 * N**2)
    
    popt, _ = curve_fit(func1, list(range(9)), tmp, p0=[2.0, 1, 4.0])
    ksis.append(popt[0])
plt.plot(Ts, ksis)