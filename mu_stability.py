#%%
from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

num_days = 500
t = np.linspace(0, num_days, num_days)

I0 = 1
S0 = N - I0
Q0 = 0
R0 = 0
z0SIS = [S0, I0, R0]
z0SIQRS = [S0, I0, Q0, R0]
mu = np.linspace(0, 1, 500)
steady = np.zeros((len(mu), 4))


### alpha = 1/2 og mu varierer
for i in range(len(mu)):
    z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[i],))
    steady[i, :] = z[-1, :]
plt.figure()
plt.plot(mu, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(mu, steady[:, 1], color = '#228B22', label='I')
plt.plot(mu, steady[:, 2], color = '#FF8C00', label='Q')
plt.plot(mu, steady[:, 3], color = '#B22222', label='R')
plt.xlabel(r'$\mu$')
plt.ylabel('SIQRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $\mu$ varierer')
plt.grid()


### alpha varierer og mu = 1/7

alpha = np.linspace(0, 1/300, 100)
steady = np.zeros((len(alpha), 4))
for i in range(len(alpha)):
    z = odeint(SIQRS, z0SIQRS, t, args=(alpha[i], 1/7))
    steady[i, :] = z[-1, :]
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


### test for hvornår endemisk fase opnås, alpha = 1/2 og mu varieres gradvist
z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[0],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIQRS-values')
plt.title('SIQRS for $\mu$ = 0')
plt.legend(loc='best')
plt.grid()

z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[100],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIQRS-values')
plt.title('SIQRS for $\mu$ = 0.5025')
plt.legend(loc='best')
plt.grid()

z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[130],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIQRS-values')
plt.title('SIQRS for $\mu$ = 0.6532')
plt.legend(loc='best')
plt.grid()

z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[170],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIQRS-values')
plt.title('SIQRS for $\mu$ = 0.3407')
plt.legend(loc='best')
plt.grid()

z = odeint(SIQRS, z0SIQRS, t, args=(1/2, mu[190],))

fig2 = plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#FF8C00', label='Q')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
plt.xlabel('t')
plt.ylabel('SIQRS-values')
plt.title('SIQRS for $\mu$ = 1.0')
plt.legend(loc='best')
plt.grid()


plt.show()