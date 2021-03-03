#%% How to solve ODE

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# region loading data
mat = pd.read_csv('Data/Test_pos_over_time.csv', sep=';')
cases = np.array([mat.NewPositive[0:-2]]).squeeze().astype(int)
time = np.linspace(0, cases.size, cases.size)
sick = np.zeros(cases.size)

plt.figure()
plt.plot(time, cases, 'b', label='cases pr. day')
plt.title('NOT modified')

# finding the total number of sick people
for i in range(cases.size):
    if i < 14:
        sick[i] = cases[i] + cases[i - 1]
    else:
        sick[i] = sick[i-1] + cases[i] - cases[i-14]


# end region

sum = 0
for i in range(1, cases.size):
    sum += 1/2*(cases[i] + cases[i -1])

print(sum)

plt.plot(time, sick, 'r', label='total cases')
plt.title('modified')
plt.legend()
plt.show()
#region SIR model

# Parameteres
beta = 0.000055
gamma = 0.25
N = 16000



# SIR as an ODE
def SIR(z, t):
    dSdt = -beta*z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt

# time span
t = np.linspace(0, 500, 1000)

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
plt.figure(4)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='R')
plt.xlabel('time')
plt.ylabel('SIR-values')
plt.legend(loc='best')



#endregion

# region SIRS model

# new parameter
alpha = 0.005

# SIS model
def SIRS(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[2]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1] - alpha * z[2]
    return [dSdt, dIdt, dRdt]

z = odeint(SIRS, z0, t)

plt.figure(5)
plt.plot(t, z[:, 0], 'b--', label='S')
plt.plot(t, z[:, 1], 'g--', label='I')
plt.plot(t, z[:, 2], 'r--', label='R')
plt.xlabel('time')
plt.ylabel('SIR-values')
plt.legend(loc='best')
plt.title('SIS')
plt.show(block=False)

# endregion