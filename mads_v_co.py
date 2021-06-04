import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models import*

print('\n','figure(1)')

print('mads - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  r = ', r, '\n',
      '  alpha = ', '\n')


print('--------------------------------------------')
print('\n','figure(2)',)

print('mads - parameters')
print('   beta = ', beta, '\n',
      '  gamma = ', gamma, '\n',
      '  r = ', r, '\n',
      '  alpha = ', '\n')

I0 = 1
S0 = N - I0
Q0 = 0
R0 = 0
z0 = [S0, I0, Q0, R0]

# solving the SIR ODE
z = odeint(mads, z0, t)

# plotting the SIR model
plt.figure(1)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.ylabel('SIQRS')
plt.xlabel('t [dage]')
plt.title('Mads model')
plt.legend(loc='best')
plt.grid()


Ii0 = 1
S0 = N - Ii0
Iq0 = 0
R0 = 0
z0 = [S0, Iq0, Ii0, R0]

# solving the SIR ODE
z = odeint(co, z0, t)

# plotting the SIR model
plt.figure(2)
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 2], color = '#228B22', label='Ii')
plt.plot(t, z[:, 1], color = '#FF8C00', label='Iq')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.ylabel('SIQRS')
plt.xlabel('t [dage]')
plt.title('co model')
plt.legend(loc='best')
plt.grid()
plt.show()