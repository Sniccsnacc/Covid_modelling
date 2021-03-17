#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
from Parameters import *

beta = np.linspace(0, 1, 1500)
t = np.linspace(0, len(new_pos)-1, len(sick[200:-1]))

Q0 = 0
I0 = 100 - Q0
S0 = N - I0 - Q0
R0 = 0

######## SIQR MODEL ##############
z0 = [S0, I0, Q0, R0]
def SIQR(x, t, beta, k = None):
    S = -beta/(N) * x[1]*x[0] #+ alpha * x[3]
    I = -S -gamma*x[1] - mu*x[1]
    Q = mu*x[1] - gamma*x[2]
    R = gamma*x[1] + gamma * x[2] #- alpha * x[3]

    return S, I, Q, R

top_as_beta_SIQR = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i], None)
    z = odeint(SIQR, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIQR[i] = np.max(z_tot)

######### SIQRSV MODEL ##############
V0 = 0
z0 = [S0, I0, Q0, R0, V0]
def SIQRSV(x, t, beta, k = None):
    dSdt = -beta/N * x[0] * x[1] + alpha * x[3]
    dIdt = beta/N * x[0] * x[1] - gamma * x[1] - mu * x[1]
    dQdt = mu * x[1] - gamma * x[2] - zeta * x[2]
    dRdt = gamma * (x[1]+x[2]) - alpha * x[3] - zeta * x[3]
    dVdt = zeta * (x[2] + x[3])
    
    return dSdt, dIdt, dQdt, dRdt, dVdt

top_as_beta_SIQRSV = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i], None)
    z = odeint(SIQRSV, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIQRSV[i] = np.max(z_tot)
#%%
######## Thresholded quarantine #########
# z0 = [S0, I0, Q0, R0]
# def Quar(x, t, beta, k = None):
#     dSdt = -beta/N * x[0] * x[1] + alpha * x[3]
#     dIdt = beta/N * x[0] * x[1] - gamma * x[1]
#     dQdt = 0
#     dRdt = gamma * x[1] - alpha * x[3]

#     if x[1] >= 1600:
#         dIdt = beta * x[0] * x[1] - gamma * x[1] - mu * x[1]
#         dQdt = mu * x[1] - gamma * x[2]
#         dRdt = gamma * (x[1]+x[2]) - alpha * x[3]
#     elif x[2] > 0:
#         dQdt = - gamma * x[2]


#     return dSdt, dIdt, dQdt, dRdt

# top_as_beta_QUAR = np.empty(len(beta))

# for i in range(len(beta)):
#     args = (beta[i], None)
#     z = odeint(Quar, z0, t, args)
#     z_tot = z[:,1] + z[:,2]

#     top_as_beta_QUAR[i] = np.max(z_tot)

#%% Plotting

plt.figure()
plt.plot(beta, top_as_beta_SIQR)
plt.plot(beta, top_as_beta_SIQRSV)
#plt.semilogy(beta, top_as_beta_QUAR)
plt.xlabel('Beta values')
plt.ylabel('Max number of infected at one day')
plt.title('Top of infection curve as function of beta')
plt.legend(('SIQR', 'SIQRSV', 'Thresholded Quarantine'))
plt.show()


