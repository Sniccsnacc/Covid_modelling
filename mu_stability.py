from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

I0 = 1
S0 = N - I0
Q0 = 0
R0 = 0
z0SIS = [S0, I0, R0]
z0SIQRS = [S0, I0, Q0, R0]
mu = np.linspace(0, 1, 100)
steady = np.zeros((len(mu), 4))

for i in range(len(mu)):
    z = odeint(SIQRS, z0SIQRS, t, args=(1/14, mu[i],))
    steady[i, :] = z[-1, :]



plt.figure()
plt.plot(mu, steady[:, 0], 'b-', label='S')
plt.plot(mu, steady[:, 1], 'g-', label='I')
plt.plot(mu, steady[:, 2], '-', label='Q')
plt.plot(mu, steady[:, 3], 'r-', label='R')
plt.xlabel(r'$\mu$')
plt.ylabel('SIQRS-stable')
plt.legend(loc='best')
plt.title(r'stability for SIQRS when varying $\mu$')
plt.grid()
plt.show()