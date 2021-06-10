#%%
from Models import *
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

t0 = 0
tf = 60

Ii0 = 1
S0 = N - Ii0
Iq0 = 0
R0 = 0
z0SIS = [S0, Ii0, R0]
z0Co = [S0, Iq0, Ii0, R0]
r = np.linspace(0, 1, 100)
steady = np.zeros((len(r), 4))

def co(t, z, alpha=alpha, r = r):
    dSdt = - (beta * z[0] * z[2] / N) + alpha * z[3]
    dIqdt = beta * z[0] * z[2] * r / N - gamma * z[1]
    dIidt = beta * z[0] * z[2] * (1-r) / N - gamma * z[2]
    dRdt = gamma * z[1] + gamma * z[2] - alpha * z[3]
    return [dSdt, dIqdt, dIidt, dRdt]

### alpha = 1/2 og mu varierer
for i in range(len(r)):
    z = solve_ivp(co, (t0, tf), z0Co, vectorized=True, args=(alpha, r[i]), rtol=2.220446049250313e-14)
    steady[i, :] = z.y[:, -1]

plt.figure()
plt.plot(r, steady[:, 0], color = '#00BFFF', label='S')
plt.plot(r, steady[:, 2], color = '#228B22', label='$I_i$')
plt.plot(r, steady[:, 1], color = '#FF8C00', label='$I_q$')
plt.plot(r, steady[:, 3], color = '#B22222', label='R')
plt.xlabel('$r$')
plt.ylabel('SIQRS-stabil')
plt.legend(loc='best')
plt.title(r'Stabilitet for SIQRS når $r$ varierer')
plt.grid()
plt.show()