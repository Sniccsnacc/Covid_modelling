#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from Models import*

# region printing parameters
print('\n','figure(1)')

print('SIR - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n')

print('\nSIR with birth and death rate - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  birth rate = ', tau, '\n',
      '  death rate = ', psi, '\n')

print('\nSIRS - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n')

print('--------------------------------------------')
print('\n','figure(2)',)

print('\nSIQRS - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n',
      '  quarantine rate = ', mu, '\n')

print('\nSIQRSV - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n',
      '  quarantine rate = ', mu, '\n',
      '  Vaccination rate = ', zeta, '\n')

print('--------------------------------------------')
print('\n','figure(3)',)

print('\nQuar - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  resusceptible rate = ', alpha, '\n',
      '  quarantine rate = ', mu, '\n')
# endregion

#number of plots
numplot = 2

# region SIR model

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(4, 1, 1)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.title('SIR')
plt.legend(loc='best')
plt.grid()



#endregion

# region SIR model with birth and death rate
z = odeint(SIRBD, z0, t)


fig1.add_subplot(4, 1, 2)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.legend(loc='best')
plt.title('SIR med fødsels- og dødsrate')
plt.grid()
plt.tight_layout(h_pad=0.02)

# endregion

# region SIS model

z = odeint(SIS, z0, t)


fig1.add_subplot(4, 1, 3)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.legend(loc='best')
plt.title('SIRS')
plt.grid()
plt.tight_layout(h_pad=0.02)



# endregion

# region SIQRS
Q0 = 0
z0 = [S0, Q0, I0, R0]

z = odeint(co, z0, t)


fig1.add_subplot(4, 1, 4)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='$I_q$')
plt.plot(t, z[:, 2], color = '#FF8C00', label='$I_i$')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.ylabel('# mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title('SIQRS')
plt.grid()
plt.tight_layout(h_pad=-1)

# endregion
#
# # region SIQRSV
# V0 = 0
# z0 = [S0, I0, Q0, R0, V0]
# z = odeint(SIQRSV, z0, t)
# fig2 = plt.figure(2)
# fig2.add_subplot(1, 1, 1)
# plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
# plt.plot(t, z[:, 1], color = '#228B22', label='I')
# plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
# plt.plot(t, z[:, 3], color = '#B22222', label='R')
# plt.plot(t, z[:, 4], '-', label='V', color='pink')
# plt.xlabel('time')
# plt.ylabel('SIQRSV-værdier')
# plt.legend(loc='best')
# plt.title('SIQRSV')
# plt.grid()
# plt.tight_layout()
#
#
# # endregion
#
# # region Quar, SIRS with quarantine after a given number of infected
# z0 = [S0, I0, Q0, R0]
# z = odeint(Quar, z0, t)
#
# plt.figure(3)
# plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
# plt.plot(t, z[:, 1], color = '#228B22', label='I')
# plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
# plt.plot(t, z[:, 3], color = '#B22222', label='R')
# plt.ylabel('SIQRS-værdier')
# plt.legend(loc='best')
# plt.title('Quar')
# plt.grid()
# plt.show()
#
#
#
# # endregion
#
