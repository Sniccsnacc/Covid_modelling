# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:31:26 2021

@author: Magnus
"""

### Equilibrium point analysis


# region SIS model
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models import*

print('\nSIRS - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n')


# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

z = odeint(SIS, z0, t)


plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.legend(loc='best')
plt.title('SIRS')
plt.grid()
plt.tight_layout(h_pad=0.02)



idx = (483830 - 1000 < z[:,0]) * (z[:,0] < 483830 + 1000)
t = np.where(idx == False) # dag 269


