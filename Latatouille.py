#%% Load in data

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
import numpy as np
from ipywidgets import interactive
from SIR_leg import sick

D_O_T = pd.read_csv('Data/Deaths_over_time.csv',sep = ';')
Cases_sex = pd.read_csv('Data/Cases_by_sex.csv', sep = ';')
lsick = pd.read_csv('Data/Test_pos_over_time.csv', sep = ';')
new_pos = lsick.NewPositive

plt.figure(1)
plt.plot(D_O_T.Dato[0:-1], D_O_T.Antal_døde[0:-1])
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.xticks(rotation=45)
plt.title('Deaths per day')
plt.ylabel('Number of deaths')
plt.show()


#%% Solve equations

p = {"beta": 0.55,
    "gamma": 1/14,
    "mu": 0.55,
    "N": 5806000}
t = np.linspace(0, len(new_pos)-1, len(sick[200:-1]))

Iso0 = 500
I0 = 1717 - Iso0
S0 = p["N"] - I0 - Iso0
R0 = 13858
z0 = [S0, I0, Iso0, R0]
#gamma = p["gamma"]

def sliderplot(beta, gamma):

    def SIR(x, t):
        S = -p["beta"]/(beta*p["N"]) * x[1]*x[0]
        I = -S -1/gamma*x[1] - p["mu"]*x[1]
        Iso = p["mu"]*x[1] - 1/gamma*x[2]
        R = 1/gamma*x[1] + 1/gamma * x[2]

        return S, I, Iso, R
    
    # solving the SIR ODE
    z = odeint(SIR, z0, t)

    plt.figure()
    plt.plot(t, z)
    plt.plot(range(len(sick[200:-1])), sick[200:-1])
    plt.legend(['S', 'I', 'Iso', 'R', 'DK'])
    plt.xlim(50, 200)
    plt.ylim(0, 500000)
    plt.show()

    return z

interactive_plot = interactive(sliderplot, beta = (0.001, 3, 0.0001), gamma = (1, 29, 1))
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot


#%% Not interactive

gamma = 1/p["gamma"]

def SIR(x, t):
    S = -p["beta"]/(3.71*p["N"]) * x[1]*x[0]
    I = -S -1/gamma*x[1] - p["mu"]*x[1]
    Iso = p["mu"]*x[1] - 1/gamma*x[2]
    R = 1/gamma*x[1] + 1/gamma * x[2]

    return S, I, Iso, R

# solving the SIR ODE
z = odeint(SIR, z0, t)

plt.figure()
plt.plot(t, z[:,2])
plt.plot(range(len(sick[200:-1])), sick[200:-1])
#plt.legend(['S', 'I', 'Iso', 'R', 'DK'])
#plt.xlim(50, 200)
#plt.ylim(0, 500000)
plt.show()

#%% 

#Bastian sutter stor penis
