__doc__ = """ Code for question 1(a). """

from MCMC import MCMC
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def stat1(sigma, H):
    """ Return energy """
    return H 

def stat2(sigma, H):
    """ Return energy squared """
    return H * H 

def func(x, T0, gamma, C):
    return C * abs(1 - x / T0)**(-gamma)

np.random.seed(100)
h, q, N = 0., 3, 16
Ts = np.linspace(0.04, 2, 50)
us = []
cs = []
for T in tqdm(Ts):
    beta = 1. / T
    tmp = MCMC(beta, h, q, N, [stat1, stat2], [50000, 10000])

    us.append(tmp[0] / N**2)
    cs.append((tmp[1] - tmp[0]**2) / N**4 * beta**2)

plt.plot(Ts, us)

plt.show()
plt.plot(Ts, cs)

plt.show()

popt, _ = curve_fit(func, Ts, cs, p0=[1.0, -0.5, 1e-3])
print(popt)