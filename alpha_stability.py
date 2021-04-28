from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


I0 = 1
S0 = N - I0
Q0 = 0
R0 = 0
z0SIS = [S0, I0, R0]
z0SIQRS = [S0, I0, Q0, R0]
alpha = np.linspace(0, 1, 100)
steady = np.zeros((len(alpha), 4))

for i in range(len(alpha)):
    z = odeint(SIQRS, z0SIQRS, t, args=(alpha[i],))
    steady[i, :] = z[-1, :]

plt.figure()
plt.plot(alpha, steady[:, 0], 'b-', label='S')
plt.plot(alpha, steady[:, 1], 'g-', label='I')
plt.plot(alpha, steady[:, 2], '-', label='Q')
plt.plot(alpha, steady[:, 3], 'r-', label='R')
plt.xlabel(r'$\alpha$')
plt.ylabel('SIQRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $\alpha$ variere')
plt.grid()


steady = np.zeros((len(alpha), 3))
for i in range(len(alpha)):
    z = odeint(SIS, z0SIS, t, args=(alpha[i],))
    steady[i, :] = z[-1, :]

plt.figure()
plt.plot(alpha, steady[:, 0], 'b-', label='S')
plt.plot(alpha, steady[:, 1], 'g-', label='I')
plt.plot(alpha, steady[:, 2], 'r-', label='R')
plt.xlabel(r'$\alpha$')
plt.ylabel('SIRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIRS når $\alpha$ variere')
plt.grid()
plt.show()