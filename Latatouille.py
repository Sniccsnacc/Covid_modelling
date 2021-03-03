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
    "N": 5806000}
t = np.linspace(0, len(new_pos)-1, len(sick[200:-1]))
I0 = 1717
S0 = p["N"] - I0
R0 = 13858
z0 = [S0, I0, R0]
#gamma = p["gamma"]

def sliderplot(beta, gamma):

    def SIR(x, t):
        S = -p["beta"]/(beta*p["N"]) * x[1]*x[0]
        I = -S -1/gamma*x[1]
        R = 1/gamma*x[1]

        return S, I, R
    
    # solving the SIR ODE
    z = odeint(SIR, z0, t)

    plt.figure()
    plt.plot(t, z)
    plt.plot(range(len(sick[200:-1])), sick[200:-1])
    plt.legend(['S', 'I', 'R', 'DK'])
    plt.xlim(50, 200)
    plt.ylim(0, 500000)
    plt.show()

interactive_plot = interactive(sliderplot, beta = (0.001, 10, 0.001), gamma = (1, 29, 1))
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot