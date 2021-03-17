#%% How to solve ODE

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd


# region Explixit euler just for fun

def ExplicitEuler(fun, x0, tspan):
    nx = len(x0)
    X = np.zeros((nx, tspan.size), dtype='float')
    T = np.zeros(tspan.size, dtype='float')

    X[:, 0] = x0
    for k in range(tspan.size - 1):
        f = np.array(fun(X[:, k], tspan), dtype=float)
        dt = tspan[k+1] - tspan[k]
        X[:, k+1] = X[:, k] + f * dt

    return X.T
# endregion

# region loading data and modifying
mat = pd.read_csv('Data/Test_pos_over_time.csv', sep=';')
cases = np.array([mat.NewPositive[0:-2]]).squeeze().astype(int)
time = np.linspace(0, cases.size, cases.size)
sick = np.zeros(cases.size)


# finding the total number of sick people
for i in range(cases.size):
    if i < 14:
        sick[i] = cases[i] + cases[i - 1]
    else:
        sick[i] = sick[i-1] + cases[i] - cases[i-14]

# endregion

#region Parameters and number of plots
N = 6e6
beta = 0.0000045
gamma = 0.147059



# new parameter, for how quickly you can become susceptible again
# used in SIS model
alpha = 0.01


# new parameter, for how many infected go to quarantine
# used in SIQRS model
mu = 0.0011

# new parameter, for how many quarantined and removed individuals get vaccinated / become immune
zeta = 0.02

# time span
t = np.linspace(0, 100, 1000)

#number of plots
numplot = 4
# endregion

# region SIR model
'''
The regular SIR model
'''

# SIR as an ODE
def SIR(z, t):
    dSdt = -beta *z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt


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

# SIS model
def SIS(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[2]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1] - alpha * z[2]
    return [dSdt, dIdt, dRdt]

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
'''
Here the SIRS model has been extended, such that some infected individuals go to quarantine and dose not come out before 
the are susceptible again
'''

def SIQRS(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * (z[2] + z[3])
    dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[2]
    dQdt = mu * z[1] - gamma * z[2]
    dRdt = gamma * z[1] - alpha * z[3]
    return [dSdt, dIdt, dQdt, dRdt]


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
def SIQRSV(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[3]
    dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[1]
    dQdt = mu * z[1] - gamma * z[2] - zeta * z[2]
    dRdt = gamma * (z[1] + z[2]) - alpha * z[3] - zeta * z[3]
    dVdt = zeta * (z[2] + z[3])
    return [dSdt, dIdt, dQdt, dRdt, dVdt]

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

def Quar(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[3]
    dIdt = beta * z[0] * z[1] - gamma * z[1]
    dQdt = 0
    dRdt = gamma * z[1] - alpha * z[3]

    if z[1] >= 1600:
        dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[2]
        dQdt = mu * z[1] - gamma * z[2]
        dRdt = gamma * z[1] - alpha * z[3]
    elif z[2] > 0:
        dQdt = - gamma * z[2]


    return [dSdt, dIdt, dQdt, dRdt]


z0 = [S0, I0, Q0, R0]
x0 = [S0, I0, R0]
z = ExplicitEuler(Quar, z0, t)
x = ExplicitEuler(SIS, x0, t)


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
