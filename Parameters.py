import numpy as np
# The parameters used in the different models:

# Parameters
N = 5806000
beta = 0.0000045
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
t = np.linspace(0, 100, 1000)
