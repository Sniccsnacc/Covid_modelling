import numpy as np
import pandas as pd

# region loading data and modifying
mat = pd.read_csv('Data/Test_pos_over_time.csv', sep=';')
cases = np.array([mat.NewPositive[0:-2]]).squeeze().astype(int)
time = np.linspace(0, cases.size, cases.size)
sick = np.zeros(cases.size)
new_pos = mat.NewPositive


# finding the total number of sick people
for i in range(cases.size):
    if i < 14:
        sick[i] = cases[i] + cases[i - 1]
    else:
        sick[i] = sick[i-1] + cases[i] - cases[i-14]

# endregion

# region The parameters used in the different models:

# Parameters
N = 5.806e6
quar_thresshold_procentage = 0.25
quar_thresshold = N * (1 - quar_thresshold_procentage)
beta = 1 / N
gamma = 1/7

# new parameter, for how quickly you can become susceptible again
# used in SIS model
alpha = 1/14


# new parameter, for how many infected go to quarantine
# used in SIQRS model
mu = 1/4

# new parameter, for how many quarantined and removed individuals get vaccinated / become immune
zeta = 0.02

# time span
num_days = 200
t = np.linspace(0, num_days, num_days)

# endregion