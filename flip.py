__doc__ = """ Code for flipping """

import numpy as np

# Flip the old spin to a new distinct one
def flip(q, old):
    new = np.random.randint(q-1) + 1
    return new + (1 if new >= old else 0)

# Gibbs sampling 
def Gibbs(sigma, beta, H, h, q, N):
    # randomly choose a spin to flip
    i, j = np.random.randint(N), np.random.randint(N)
    old = sigma[i, j]
    new = flip(q, old)

    DeltaH = h * (old - new)
    for dir in [(-1,0), (1,0), (0,1), (0,-1)]:
        adjacent = ((i+dir[0]) % N, (j+dir[1]) % N)
        DeltaH -= 1 if new == sigma[adjacent] else old == sigma[adjacent]
    if DeltaH <= 0 or np.random.random() < np.exp(-beta * DeltaH):
        sigma[i, j] = new
        return DeltaH + H
    return H
