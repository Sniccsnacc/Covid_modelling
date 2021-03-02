# How to solve ODE

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math


#region ODE_1: dy/dt = -y(t) + 1, y(0) = 0

# making funciton of RHS of ODE
def ODE1(y,t):
    dydt = -y + 1
    return dydt


# making time
t = np.linspace(0, 20)

# initial condition
y0 = 0

# solving ODE_1
y = odeint(ODE1, y0, t)

#plotting the graph
plt.figure(1)
plt.plot(t, y)
plt.xlabel('time')
plt.ylabel('y(t)')
#plt.show()

#endregion


#region ODE_2: 5* dy/dt = -y(t) + u(t), y(0) = 1, u = 0 for t<10 and u = 2 for t>=10

# defintion the RHS of ODE_2
def ODE2(y,t):
    if(t<10):
        u=0
    else:
        u=2

    dydt = (-y+u)/5.0
    return dydt

# making time
t = np.linspace(0, 40)

# intial condition
y0 = 1

# solving ODE_2
y = odeint(ODE2,y0,t)

# plot
plt.figure(2)
plt.plot(t, y, 'r-', label='Output (y(t))')
plt.plot([0,10,10,40],[0,0,2,2],'b-',label='Input (u(t))')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
#plt.show()

#endregion


#region ODE_3:_ dx/dt = 3*exp(-t), dy/dt = 3 - y(t), x(0)=0, y(0)=0

def ODE3(z,t):
    dxdt = 3.0 * math.exp(-t)
    dydt = 3.0 - z[1]
    dzdt = [dxdt, dydt]
    return dzdt

t = np.linspace(0,5)

z0 = [0,0]

z = odeint(ODE3, z0, t)

plt.figure(3)
plt.plot(t,z[:,0],'b-',label=r'$\frac{dx}{dt}=3 \; \exp(-t)$')
plt.plot(t,z[:,1],'r--',label=r'$\frac{dy}{dt}=-y+3$')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
#plt.show()


#endregion


#region SIR model

# Parameteres
beta = 0.000055
gamma = 0.25
N = 16000



# SIR as an ODE
def SIR(z, t):
    dSdt = -beta*z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt

# time span
t = np.linspace(0, 30)

# initial conditions
I0 = 1
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

# solving the SIR ODE
z = odeint(SIR, z0, t)

# plotting the SIR model
plt.figure(4)
plt.plot(t, z[:, 0], 'b-', label='S')
plt.plot(t, z[:, 1], 'g-', label='I')
plt.plot(t, z[:, 2], 'r-', label='R')
plt.xlabel('time')
plt.ylabel('SIR-values')
plt.legend(loc='best')
plt.show()


#endregion
