import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from Models import*

num_regions = 3
betas = [1/2, 1/2, 1/2]
gammas = [1/7, 1/7, 1/7]
rhos = [1, 1, 1]

B = np.zeros((num_regions, num_regions))
np.fill_diagonal(B, betas)

P = np.zeros((num_regions, num_regions))
np.fill_diagonal(P, rhos)

C = np.zeros((num_regions, num_regions))
np.fill_diagonal(C, gammas)


def SIR_SEG(z, t):
    r, c = z.shape
    l = t.size
    S = np.zeros((r, l+1))
    I = np.zeros((r, l+1))
    R = np.zeros((r, l+1))
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    for i in range(1, t.size):
        dSdt = -B.dot(S[:, i-1] * I[:, i-1]) + P.dot(S[:, i-1])
        dIdt = B.dot(S[:, i-1] * I[:, i-1]) - C.dot(I[:, i-1]) + P.dot(I[:, i-1])
        dRdt = C.dot(I[:, i-1])
        print(dSdt, dIdt, dRdt)
        S[:, i] = S[:, i-1] + dSdt
        I[:, i] = I[:, i-1] + dIdt
        R[:, i] = R[:, i-1] + dRdt

    return S, I, R


I0 = np.ones(num_regions)
R0 = np.zeros(num_regions)
S0 = np.ones(num_regions) * (N - 1)
Z0 = np.transpose(np.array([S0, I0, R0]))
print(Z0)
S, I, R = SIR_SEG(Z0, t)


