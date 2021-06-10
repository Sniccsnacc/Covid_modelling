#%%
from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

num_days = 406
t = np.linspace(0, num_days, num_days)

I0 = 1
S0 = N - I0
Q0 = 0
R0 = 0
z0SIS = [S0, I0, R0]

z0co = np.array([S0, Q0, I0, R0])
r = np.linspace(0, 1, 100)
steady = np.zeros((len(r), 4))


### alpha = 1/2 og mu varierer
for i in range(len(r)):
    z = odeint(co, z0co, t, args=(1/240, r[i],))
    steady[i, :] = z[-1, :]

plt.figure()
plt.plot(r, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(r, steady[:, 1], color = '#228B22', label='$I_i$')
plt.plot(r, steady[:, 2], color = '#FF8C00', label='$I_q$')
plt.plot(r, steady[:, 3], color = '#B22222', label='R')
plt.xlabel(r'$\mu$')
plt.ylabel('SIQRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $\mu$ varierer')
plt.grid()

### alpha varierer og mu = 1/7

alpha = np.linspace(0, 1/300, 100)
steady = np.zeros((len(alpha), 4))
for i in range(len(alpha)):

    # S, Iq, Ii, R = ExplicitEuler_co(z0co, t, alpha=alpha[i], r = 1/7)
    z = odeint(co, z0co, t, args=(alpha[i], 1/7))
    steady[i, :] = z[-1, :]
    # steady[i, 0] = S[-1]
    # steady[i, 1] = Iq[-1]
    # steady[i, 2] = Ii[-1]
    # steady[i, 3] = R[-1]

plt.figure()
plt.plot(alpha, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(alpha, steady[:, 1], color = '#228B22', label='I')
plt.plot(alpha, steady[:, 2], color = '#FF8C00', label='Q')
plt.plot(alpha, steady[:, 3], color = '#B22222', label='R')
plt.xlabel(r'$\alpha$, [$1/dage$]')
plt.ylabel('co-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $\alpha$ varierer')
plt.grid()

plt.show()