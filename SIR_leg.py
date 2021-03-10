#%% How to solve ODE

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# region loading data and modifing
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

#region Parameters
beta = 0.000055
gamma = 0.25
N = 16000


# new parameter, for how quickly you can become susceptible again
# used in SIRS model
alpha = 0.04


# new parameter, for how many infected go to quarantine
# used in SIQRS model
eps = 0.02

# new parameter, for how many quarantined and removed individuals get vaccinated / become immune


# endregion

# region SIR model
'''
The regular SIR model
'''

# SIR as an ODE
def SIR(z, t):
    dSdt = -beta*z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt


# time span
t = np.linspace(0, 100, 1000)

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
plt.figure()
plt.subplot(3, 1, 1)
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


plt.subplot(3, 1, 2)
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
    dIdt = beta * z[0] * z[1] - gamma * z[1] - eps * z[2]
    dQdt = eps * z[1] - gamma * z[2]
    dRdt = gamma * z[1] - alpha * z[3]
    return [dSdt, dIdt, dQdt, dRdt]


Q0 = 0
z0 = [S0, I0, Q0, R0]

z = odeint(SIQRS, z0, t)

plt.subplot(3, 1, 3)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='Q')
plt.plot(t, z[:, 3], '--', label='R')
plt.xlabel('time')
plt.ylabel('SIQRS-values')
plt.legend(loc='best')
plt.title('SIQRS')
plt.show()

# endregion

# region SIQRSV

# endregion