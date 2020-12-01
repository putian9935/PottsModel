__doc__ = """ Framework for MCMC """

from flip import Gibbs
import numpy as np

def MCMC(beta, h, q, N, func_array, args):
    # MCMC framework

    # Initialize the state
    sigma = (q // 2 + 1) * np.ones((N, N))
    horz = sigma == sigma[:, list(range(1, N)) + [0]]
    vert = sigma == sigma[list(range(1, N)) + [0], :]
    H = -np.sum(horz) -np.sum(vert) - h * np.sum(sigma)

    # Initialize the parameters 
    mc_iter, burn_in = args
    num_array = len(func_array)
    sum_array = [0.] * num_array

    # Burn-in  
    for _ in range(burn_in):
        H = Gibbs(sigma, beta, H, h, q, N)
    # Main loop
    from tqdm import tqdm
    for _ in range(mc_iter):
        for k in range(num_array):
            sum_array[k] += func_array[k](sigma, H)
        H = Gibbs(sigma, beta, H, h, q, N)
    return np.array(sum_array) / mc_iter

    