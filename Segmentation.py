import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from Models import*

num_regions = 3
betas = [1, 2, 3]
gammas = [1, 2, 3]
rhos = [1, 2, 3]

B = np.zeros((num_regions, num_regions))
np.fill_diagonal(B, betas)

R = np.zeros((num_regions, num_regions))
np.fill_diagonal(R, rhos)

C = np.zeros((num_regions, num_regions))
np.fill_diagonal(C, gammas)


def SIR_SEG(z, t):
    r, c = z.shape
    l = t.size
    S = np.zeros((r, l+1))
    I = np.zeros((r, l+1))
    R = np.zeros((r, l+1))
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 0]
    R[:, 0] = z[:, 0]

    for i in range(1, t.size):
        dt = t[i] - t[i-1]
        dSdt = -B.dot(S[:, i-1] * I[:, i-1]) + R.dot(S[:, i-1])
        dIdt = B.dot(S[:, i-1] * I[:, i-1]) - C.dot(I[:, i-1]) + R.dot(I[:, i-1])
        dRdt = C.dot(I[:, i-1])
        S[:, i] = S[:, i-1] + dSdt * dt
        I[:, i] = I[:, i-1] + dIdt * dt
        R[:, i] = R[:, i-1] + dRdt * dt

    return S, I, R


I0 = np.ones(num_regions)
R0 = np.zeros(num_regions)
S0 = np.ones(num_regions) * (N - 1)
Z0 = np.transpose(np.array([S0, I0, R0]))

S, I, R = SIR_SEG(Z0, t)


