from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
Q0 = 0
z0 = [S0, I0, Q0, R0]
mu = np.linspace(0, 1 - gamma, 100)

# allocation space
R_total = np.zeros((len(mu)))
Q_total = np.zeros((len(mu)))
dt = mu[1] - mu[0]

# calculating total number of recovered and infected for a given mu
for i in range(len(mu)):
    z = odeint(SIQRS, z0, t, args=(mu[i],))
    R_total[i] = np.sum(z[3]) * dt
    Q_total[i] = np.sum(z[2]) * dt

# plotting
fig = plt.figure()
plt.plot(mu, R_total - Q_total)
plt.show()