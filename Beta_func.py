#%% Beta stability

from Models import *
from scipy.integrate import odeint

S0 = N-1
I0 = 1
R0 = 0
z0 = [S0, I0, R0]

S_end = []
I_end = []
R_end = []

betas = np.linspace(0, 1, 10000)

for i in betas:
    Z = odeint(SIR, z0, t, args = (i, ))
    S_end.append(Z[-1,0])
    I_end.append(Z[-1,1])
    R_end.append(Z[-1,2])


#%% Plotting
import matplotlib.pyplot as plt

plt.figure(1)
#plt.plot(betas, S_end, color = '#00BFFF', label='S')
plt.plot(betas, I_end, color = '#228B22', label='I')
#plt.plot(betas, R_end, color = '#B22222', label='R')
plt.ylabel('Antal mennesker')
plt.xlabel(r'$\beta$')
plt.title('SIR')
plt.legend(loc='best')
plt.grid()

