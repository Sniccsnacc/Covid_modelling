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
N = 5806000
beta = 0.1 / N
gamma = 0.147059

# new parameter, for how quickly you can become susceptible again
# used in SIS model
alpha = 0.01


# new parameter, for how many infected go to quarantine
# used in SIQRS model
mu = 0.11

# new parameter, for how many quarantined and removed individuals get vaccinated / become immune
zeta = 0.02

# time span
t = np.linspace(0, 0.1, 1000)

# endregion