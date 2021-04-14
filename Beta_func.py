#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
from Parameters import *

beta = np.linspace(0, 1, 1500)/N
t = np.linspace(0, len(new_pos)-1, len(sick[200:-1]))

Q0 = 0
I0 = 100 - Q0
S0 = N - I0 - Q0
R0 = 0

######## SIQR MODEL ##############
z0 = [S0, I0, Q0, R0]
def SIQR(x, t, beta):
    S = -beta * x[1]*x[0] #+ alpha * x[3]
    I = -S -gamma*x[1] - mu*x[1]
    Q = mu*x[1] - gamma*x[2]
    R = gamma*x[1] + gamma * x[2] #- alpha * x[3]

    return [S, I, Q, R]

top_as_beta_SIQR = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i],)
    z = odeint(SIQR, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIQR[i] = np.max(z_tot)

######### SIQRSV MODEL ##############
V0 = 0
z0 = [S0, I0, Q0, R0, V0]
def SIQRSV(x, t, beta):
    dSdt = -beta * x[0] * x[1] + alpha * x[3]
    dIdt = beta * x[0] * x[1] - gamma * x[1] - mu * x[1]
    dQdt = mu * x[1] - gamma * x[2] - zeta * x[2]
    dRdt = gamma * (x[1]+x[2]) - alpha * x[3] - zeta * x[3]
    dVdt = zeta * (x[2] + x[3])
    
    return dSdt, dIdt, dQdt, dRdt, dVdt

top_as_beta_SIQRSV = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i],)
    z = odeint(SIQRSV, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIQRSV[i] = np.max(z_tot)

######## Thresholded quarantine #########
z0 = [S0, I0, R0]
def SIR(z, t, beta):
    dSdt = -beta *z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt



top_as_beta_SIR = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i],)
    z = odeint(SIR, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIR[i] = np.max(z_tot)

#%% Plotting

plt.figure()
plt.plot(beta*N, top_as_beta_SIQR)
plt.plot(beta*N, top_as_beta_SIQRSV)
plt.plot(beta, top_as_beta_SIR)
plt.xlabel('Beta values')
plt.ylabel('Max number of infected at one day')
plt.title('Top of infection curve as function of beta')
plt.legend(('SIQR', 'SIQRSV', 'Thresholded Quarantine'))
plt.show()


#%% Interpolation

from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x+b

fr = 594

p1 = curve_fit(f, beta[fr:]*N, top_as_beta_SIQR[fr:])
p2 = curve_fit(f, beta[fr:]*N, top_as_beta_SIQRSV[fr:])


fig, ax = plt.subplots(nrows = 2, ncols = 1)
plt.tight_layout(h_pad=3.0)
plt.subplot(2,1,1)
plt.plot(beta[fr:]*N, top_as_beta_SIQR[fr:])
plt.plot(beta[fr:]*N, f(beta[fr:]*N, p1[0][0], p1[0][1]))
plt.xlabel('Beta values')
plt.title('Top of infection curve as function of beta')
plt.legend(('SIQR', str(round(p1[0][0],2)) + 'x + ' + str(round(p1[0][1],2))))

plt.subplot(2,1,2)
plt.plot(beta[fr:]*N, top_as_beta_SIQRSV[fr:])
plt.plot(beta[fr:]*N, f(beta[fr:]*N, p2[0][0], p2[0][1]))
plt.xlabel('Beta values')
plt.ylabel('Max number of infected at one day')
plt.title('Top of infection curve as function of beta')
plt.legend(('SIQRSV', str(round(p2[0][0],2)) + 'x + ' + str(round(p2[0][1],2))))
plt.show()
