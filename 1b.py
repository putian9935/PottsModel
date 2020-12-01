__doc__ = """ Code for question 1(b). """

from MCMC import MCMC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def stat(sigma, H):
    """ Return energy """
    return np.sum(sigma)

q, N = 3, 16
hs = np.linspace(0.04, 2, 50)
for T in [.5,1.,2]:
    beta = 1. / T   
    ms = []
    for h in tqdm(hs):
        tmp = MCMC(beta, h, q, N, [stat], [1000000, 10000])
        ms.append(tmp[0] / N**2)

    plt.plot(hs, ms, label=r'$T=%.1f$'%T) 
plt.legend()
plt.savefig('1b.png')