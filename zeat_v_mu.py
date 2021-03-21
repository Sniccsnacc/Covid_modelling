from Models import*
from scipy.integrate import odeint
import matplotlib.pyplot as plt


I0 = 1
S0 = N - I0
R0 = 0
Q0 = 0
z0 = [S0, I0, Q0, R0]
mu = np.linspace(0, 1 - gamma, 100)


Z = np.zeros((len(mu)))
dt = mu[1] - mu[0]


for i in range(len(mu)):
    z = odeint(SIQRS, z0, t, args=(mu[i], ))
    I_total = np.sum(z[3]) * dt
    q_total = np.sum(z[2]) * dt

fig = plt.figure()
plt.plot(mu, Z)
plt.xlabel('$\mu$')
plt.show()