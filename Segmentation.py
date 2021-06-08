#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from Models import*
from Kom_dat import *

# num_regions = 3
# population = np.array([N, N, N])
# gammas = np.array([1/7, 1/7, 1/7])
# travel_out = np.array([0.1, 0.3, 0.0])
# travel_in = np.array([[0, 0.8, 0.2],
#                       [0.3, 0, 0.7],
#                       [0.5, 0.5, 0]])

num_regions = len(travel_out_2)
population = np.array(population_2)
gammas = np.ones(num_regions)
travel_out = np.array(travel_out_2)
travel_in = np.array(travel_in_2)
beta = population/N

def SIR_SEG(z, t):
    r, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((r, l+1))
    I = np.zeros((r, l+1))
    R = np.zeros((r, l+1))
    P = np.zeros((r,l))
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    P[:, 0 ] = S[:, 0] + I[: , 0] + R[:, 0]
    for i in range(1, t.size):
        #Standard form:
        dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
        dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
        dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])

        #Homecoming form:
        # if k%2:
        #     dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
        #     dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
        #     dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])
        # else:
        #     dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population + travel_out * S[:, i - 2] - (travel_out * travel_in.T).dot(S[:, i - 2])
        #     dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 2] + travel_out * I[:, i - 1] - (travel_out * travel_in.T).dot(I[:, i - 2])
        #     dRdt = gammas * I[:, i - 1] + travel_out * R[:, i - 2] - (travel_out * travel_in.T).dot(R[:, i - 2])

        k += 1

        S[:, i] = S[:, i-1] + dSdt
        I[:, i] = I[:, i-1] + dIdt
        R[:, i] = R[:, i-1] + dRdt
        P[:, i] += S[:, i] + I[: , i] + R[:, i]

        

    return S, I, R, P


I0 = np.ones(num_regions)
R0 = np.zeros(num_regions)
S0 = population - I0 #np.ones(num_regions) * (N - 1)
Z0 = np.transpose(np.array([S0, I0, R0]))
ts = t.size

S, I, R, P = SIR_SEG(Z0, t)
fig1 = plt.figure(1)
fig1.add_subplot(3, 1, 1)
plt.plot(t, S[0, 0:ts], color = '#00BFFF', label='S')
plt.plot(t, I[0, 0:ts], color = '#228B22', label='I')
plt.plot(t, R[0, 0:ts], color = '#B22222', label='R')
plt.plot(t, P[0, 0:ts], label='P')
plt.ylabel('# Mennesker')
#plt.legend(loc='best')
plt.grid()

fig1.add_subplot(3, 1, 2)
plt.plot(t, S[1, 0:ts], color = '#00BFFF', label='S')
plt.plot(t, I[1, 0:ts], color = '#228B22', label='I')
plt.plot(t, R[1, 0:ts], color = '#B22222', label='R')
plt.plot(t, P[1, 0:ts], label='P')
plt.ylabel('# Mennesker')
#plt.legend(loc='best')
plt.grid()

fig1.add_subplot(3, 1, 3)
plt.plot(t, S[2, 0:ts], color = '#00BFFF', label='S')
plt.plot(t, I[2, 0:ts], color = '#228B22', label='I')
plt.plot(t, R[2, 0:ts], color = '#B22222', label='R')
plt.plot(t, P[2, 0:ts], label='P')
plt.ylabel('# Mennesker')
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()

plt.show()
