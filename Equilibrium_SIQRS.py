# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:35:32 2021

@author: Magnus
"""
### Equilibrium point analysis for SIQRS


import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models import*

print('\nSIQRS - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n',
      '  quarantine rate = ', r, '\n')


# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# region SIQRS
Q0 = 0
z0 = [S0, Q0, I0, R0]


z = odeint(co, z0, t)

plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='Q')
plt.plot(t, z[:, 2], color = '#FF8C00', label='I')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title('SIQRS')
plt.grid()
plt.tight_layout(h_pad=-1)





idx = (5080250/9 - 200 < z[:,0]) * (5080250/9 + 200 > z[:,0])
t = np.where(idx == False) # 406


