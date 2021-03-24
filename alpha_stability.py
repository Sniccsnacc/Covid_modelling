from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]
alpha = np.linspace(0, 1, 100)
Tspan = np.linspace(0, 200, 200)


for i in range(len(alpha)):
    z = odeint(SIS, z0, Tspan, args=(alpha[i], ))
    plt.figure(1)
    plt.plot(Tspan, z[:, 1])

plt.show()