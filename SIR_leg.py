#%% How to solve ODE


import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models import*



#number of plots
numplot = 4

# region SIR model

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
fig, ax = plt.subplots(nrows = 3, ncols = 1)
plt.tight_layout()
plt.subplot(numplot, 1, 1)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='R')
plt.ylabel('SIR-values')
plt.title('SIR')
plt.legend(loc='best')



#endregion

# region SIRS model
'''
Here the SIR model has been modified such that some removed individuals can be susceptible again
'''



z = odeint(SIS, z0, t)


plt.subplot(numplot, 1, 2)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='R')
plt.ylabel('SIS-values')
plt.legend(loc='best')
plt.title('SIS')


# endregion

# region SIQRS
Q0 = 0
z0 = [S0, I0, Q0, R0]

z = odeint(SIQRS, z0, t)

plt.subplot(numplot, 1, 3)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='Q')
plt.plot(t, z[:, 3], '--', label='R')
plt.ylabel('SIQRS-values')
plt.legend(loc='best')
plt.title('SIQRS')

# endregion

# region SIQRSV
V0 = 0
z0 = [S0, I0, Q0, R0, V0]

z = odeint(SIQRSV, z0, t)

plt.subplot(numplot, 1, 4)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='Q')
plt.plot(t, z[:, 3], '--', label='R')
plt.plot(t, z[:, 4], '--', label='V', color='pink')
plt.xlabel('time')
plt.ylabel('SIQRSV-values')
plt.legend(loc='best')
plt.title('SIQRSV')


# endregion

# region Quar, SIRS with quarantine after a given number of infected
z0 = [S0, I0, Q0, R0]
x0 = [S0, I0, R0]
z = odeint(Quar, z0, t)
x = odeint(SIS, x0, t)


plt.figure(2)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='Q')
plt.plot(t, z[:, 3], '--', label='R', color='pink')
plt.plot(t, x[:, 0], 'b-', label='S')
plt.plot(t, x[:, 1], 'g-', label='I')
plt.plot(t, x[:, 2], '-', label='R', color='pink')
plt.ylabel('SIQRS-values')
plt.legend(loc='best')
plt.title('Quar')
plt.show()



# endregion
