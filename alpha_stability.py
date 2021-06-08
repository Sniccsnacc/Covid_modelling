#%%
from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# time span
num_days = 5000
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
plt.title(r'Stabilitet for Co når $\alpha$ varierer')
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






z = odeint(SIS, z0SIS, t, args=(alpha[0],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha = 1')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[10],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[10]')
plt.legend(loc='best')
plt.grid()

idx = (5806000/7/2 - 1000< z[:,0]) * (5806000/7/2 + 1000 > z[:,0])
result = np.where(idx)
print(result)
k = np.array(result)

for i in range (k.size):
    print(z[k[0][i],1])






z = odeint(SIS, z0SIS, t, args=(alpha[20],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[20]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[30],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[30]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[40],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[40]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[50],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[50]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[60],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[60]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[70],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[70]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[80],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[80]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[90],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha[90]')
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0SIS, t, args=(alpha[99],))

fig3 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIRS-values')
plt.title('SIRS for alpha = 1/300')
plt.legend(loc='best')
plt.grid()
