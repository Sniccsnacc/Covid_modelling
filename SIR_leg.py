#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, solve_ivp
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
      '  quarantine rate = ', r, '\n')

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

# region SIR model

# initial conditions
I0 = 0.5
S0 = N - I0
R0 = 0
z0 = np.array([S0, I0, R0])
t = np.linspace(0, 120, 200)

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
plt.figure(1)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.title('SIR')
plt.legend(loc='best')
plt.grid()
print(z[199, 1])


#endregion

# region SIR model with birth and death rate
z = odeint(SIRBD, z0, t)


plt.figure(2)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title('SIR med fødsels- og dødsrate')
plt.grid()
plt.tight_layout(h_pad=0.02)

# endregion

# region SIS model
t = np.linspace(0, 200, 300)
z = odeint(SIS, z0, t)


plt.figure(3)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title('SIRS')
plt.grid()





# endregion

# region SIQRS
Q0 = 0
z0 = [S0, Q0, I0, R0]


z = odeint(co, z0, t)


plt.figure(4)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 2], color = '#228B22', label='$I_i$')
plt.plot(t, z[:, 1], color = '#FF8C00', label='$I_q$')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title('SIQRS')
plt.grid()
plt.show()




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
