#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
from Latatouille import new_pos
import SIR_leg as sir

beta = np.linspace(0, 1, 1500)
t = np.linspace(0, len(new_pos)-1, len(sir.sick[200:-1]))

p = {"beta": 0.55,
    "gamma": 1/14,
    "mu": 0.3,
    "alpha": 0.0005,
    "N": 5806000}

Iso0 = 0
I0 = 100 - Iso0
S0 = p["N"] - I0 - Iso0
R0 = 0
z0 = [S0, I0, Iso0, R0]

def SIQR(x, t, beta, k = None):
    S = -beta/(p["N"]) * x[1]*x[0] #+ p["alpha"] * x[3]
    I = -S -p["gamma"]*x[1] - p["mu"]*x[1]
    Q = p["mu"]*x[1] - p["gamma"]*x[2]
    R = p["gamma"]*x[1] + p["gamma"] * x[2] #- p["alpha"] * x[3]

    return S, I, Q, R

top_as_beta_SIQR = np.empty(len(beta))

for i in range(len(beta)):
    args = (beta[i], None)
    z = odeint(SIQR, z0, t, args)
    z_tot = z[:,1] + z[:,2]

    top_as_beta_SIQR[i] = np.max(z_tot)


#%% Plotting

plt.figure()
plt.loglog(beta, top_as_beta_SIQR)
plt.xlabel('Beta values')
plt.ylabel('Max number of infected at one day')
plt.title('Top of infection curve as function of beta (SIQR)')
plt.show
