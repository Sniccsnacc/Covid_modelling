#%%
from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# time span
num_days = 269
t = np.linspace(0, num_days, num_days)

Ii0 = 1
S0 = N - Ii0
Iq0 = 0
R0 = 0
z0SIS = [S0, Ii0, R0]
z0Co = [S0, Iq0, Ii0, R0]
alpha = np.linspace(0, 1/300, 100)
steady = np.zeros((len(alpha), 4))

for i in range(len(alpha)):
    z = odeint(co, z0Co, t, args=(alpha[i],))
    steady[i, :] = z[-1, :]

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



steady = np.zeros((len(alpha), 3))
for i in range(len(alpha)):
    z = odeint(SIS, z0SIS, t, args=(alpha[i],))
    steady[i, :] = z[-1, :]

plt.figure()
plt.plot(alpha, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(alpha, steady[:, 1], color = '#228B22', label='I')
plt.plot(alpha, steady[:, 2], color = '#B22222', label='R')
plt.xlabel(r'$\alpha$, [$1/dage$]')
plt.ylabel('SIRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIRS når $\alpha$ varierer')
plt.grid()








