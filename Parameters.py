#%% 
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# region loading data and modifying
mat = pd.read_csv('Data2/Test_pos_over_time.csv', sep=';')
cases = np.array([mat.NewPositive[0:-2]]).squeeze().astype(int)
time = np.linspace(0, cases.size, cases.size)
sick = np.zeros(cases.size)
new_pos = mat.NewPositive

# Parameters
N = 5.806e6
quar_thresshold_procentage = 0.25
quar_thresshold = N * (1 - quar_thresshold_procentage)
beta = 2
gamma = 1/6

# finding the total number of sick people
for i in range(1, cases.size):
    if i < int(1/gamma):
        sick[i] = cases[i] + sick[i - 1]
    else:
        sick[i] = sick[i-1] + cases[i] - cases[i-int(1/gamma)]

# endregion

# region The parameters used in the different models:

# new parameter, for how quickly you can become susceptible again
# used in SIS model
#alpha = 0.00020202020202020205
#alpha = 0.0002356902356902357
alpha = 1/240

# new parameter, for how many infected go to quarantine
# used in SIQRS model
mu = 1/7
r = 1/7

# new parameter, for how many quarantined and removed individuals get vaccinated / become immune
zeta = 0.02

# new parameter, for birth rate (tau) and death rate (xi)
tau = (60937 / 365)
psi = (54645 / 365)

# time span
num_days = 8000
t = np.linspace(0, num_days, num_days)

# endregion
