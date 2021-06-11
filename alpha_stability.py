#%%
from Models import *
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# time span
t0 = 0
tf = 120

Ii0 = 1
S0 = N - Ii0
Iq0 = 0
R0 = 0
z0SIS = [S0, Ii0, R0]
z0Co = [S0, Iq0, Ii0, R0]
alpha = np.linspace(0, 1/300, 100)
steady = np.zeros((len(alpha), 4))

def SIS(t, z, alpha=alpha):
    dSdt = -beta * z[0] * z[1] / N + alpha * z[2]
    dIdt = beta*z[0]*z[1] / N - gamma * z[1]
    dRdt = gamma * z[1] - alpha * z[2]
    return [dSdt, dIdt, dRdt]

def co(t, z, alpha=alpha, r = r):
    dSdt = - (beta * z[0] * z[2] / N) + alpha * z[3]
    dIqdt = beta * z[0] * z[2] * r / N - gamma * z[1]
    dIidt = beta * z[0] * z[2] * (1-r) / N - gamma * z[2]
    dRdt = gamma * z[1] + gamma * z[2] - alpha * z[3]
    return [dSdt, dIqdt, dIidt, dRdt]

for i in range(len(alpha)):
    z = solve_ivp(co, (t0, tf), z0Co, vectorized=True, args=(alpha[i],), rtol=2.220446049250313e-14)
    steady[i, :] = z.y[:, -1]

plt.figure()
plt.plot(alpha, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(alpha, steady[:, 1], color = '#228B22', label='I')
plt.plot(alpha, steady[:, 2], color = '#FF8C00', label='Q')
plt.plot(alpha, steady[:, 3], color = '#B22222', label='R')
plt.xlabel(r'$\alpha$, [$1/dage$]')
plt.ylabel('co-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $\alpha$ varierer')
plt.grid()

print('her')

steady = np.zeros((len(alpha), 3))
for i in range(len(alpha)):
    z = solve_ivp(SIS, (t0, tf), z0SIS, vectorized=True, args=(alpha[i],), rtol=2.220446049250313e-14)
    steady[i, :] = z.y[:, -1]

plt.figure()
plt.plot(alpha, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(alpha, steady[:, 1], color = '#228B22', label='I')
plt.plot(alpha, steady[:, 2], color = '#B22222', label='R')
plt.xlabel(r'$\alpha$, [$1/dage$]')
plt.ylabel('SIRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIRS når $\alpha$ varierer')
plt.grid()
plt.show()