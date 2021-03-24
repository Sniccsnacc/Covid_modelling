import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models import*



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
fig1.add_subplot(2, 1, 1)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='R')
plt.ylabel('SIR-values')
plt.title('SIR')
plt.legend(loc='best')
plt.grid()



#endregion

# region SIS model

z = odeint(SIS, z0, t)


fig1.add_subplot(2, 1, 2)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='R')
plt.ylabel('SIS-values')
plt.legend(loc='best')
plt.title('SIS')
plt.grid()
plt.tight_layout()



# endregion

# region SIQRS
Q0 = 0
z0 = [S0, I0, Q0, R0]

z = odeint(SIQRS, z0, t)

fig2 = plt.figure(2)
fig2.add_subplot(2, 1, 1)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='Q')
plt.plot(t, z[:, 3], '-', label='R')
plt.ylabel('SIQRS-values')
plt.legend(loc='best')
plt.title('SIQRS')
plt.grid()

# endregion

# region SIQRSV
V0 = 0
z0 = [S0, I0, Q0, R0, V0]
z = odeint(SIQRSV, z0, t)

fig2.add_subplot(2, 1, 2)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='Q')
plt.plot(t, z[:, 3], '-', label='R')
plt.plot(t, z[:, 4], '-', label='V', color='pink')
plt.xlabel('time')
plt.ylabel('SIQRSV-values')
plt.legend(loc='best')
plt.title('SIQRSV')
plt.grid()
plt.tight_layout()


# endregion

# region Quar, SIRS with quarantine after a given number of infected
z0 = [S0, I0, Q0, R0]
z = odeint(Quar, z0, t)

plt.figure(3)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='Q')
plt.plot(t, z[:, 3], '-', label='R', color='pink')
plt.ylabel('SIQRS-values')
plt.legend(loc='best')
plt.title('Quar')
plt.grid()
plt.show()



# endregion
