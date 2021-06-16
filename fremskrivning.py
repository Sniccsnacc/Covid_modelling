import numpy as np

from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Kom_dat import *

I0 = 49
S0 = N - I0
R0 = 0
z0 = [S0, I0, R0]

z = odeint(SIR, z0, t, args=(0.192,))

plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
#plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.title("Fremskrivning af SIR med \n {}".format(r"$\beta = 0.192$"))
plt.legend(loc='best')
plt.grid()

z = odeint(SIS, z0, t, args=(alpha, 0.194,))

plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 1], color = '#228B22', label='I')
plt.plot(t, z[:, 2], color = '#B22222', label='R')
#plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title("Fremskrivning af SIRS med \n {}".format(r"$\beta = 0.194$"))
plt.grid()

Q0 = 0
z0 = [S0, Q0, I0, R0]
z = odeint(co, z0, t, args=(alpha, r, 0.224))

plt.figure()
plt.plot(t, z[:, 0], color = '#00BFFF', label='S')
plt.plot(t, z[:, 2], color = '#228B22', label='$I$')
plt.plot(t, z[:, 1], color = '#FF8C00', label='$Q$')
#plt.plot(t, z[:, 1] + z[:, 2], label='$I + Q$')
plt.plot(t, z[:, 3], color = '#B22222', label='R')
#plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title("Fremskrivning af SIQRS med \n {}".format(r"$\beta = 0.224$"))
plt.grid()


num_regions = len(travel_out)
population = np.array(population)
gammas = np.ones(num_regions) / 6
g = np.array(travel_out)
MT = np.array(travel_in).T
def SIR_SEG(z, t, beta=beta):
    rows, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((rows, l + 1))
    I = np.zeros((rows, l + 1))
    R = np.zeros((rows, l + 1))
    P = np.zeros((rows, l))

    Sa = np.zeros((rows, rows))
    Ia = np.zeros((rows, rows))
    Ra = np.zeros((rows, rows))

    Ny_smit = np.zeros((rows, l + 1))

    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    P[:, 0] = S[:, 0] + I[:, 0] + R[:, 0]

    L = g * MT

    for i in range(1, l):
        if k % 2:  # Being home step
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - 2 * g * S[:, i - 1]
            dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - 2 * g * I[:, i - 1]
            dRdt = gammas * I[:, i - 1] - 2 * g * R[:, i - 1]

            Sa = (L * S[:, i - 1]).T
            Ia = (L * I[:, i - 1]).T
            Ra = (L * R[:, i - 1]).T

            Ny_smit[:, i] = beta * S[:, i - 1] * I[:, i - 1] / population

        else:  # Being on work step
            Ia_tot = np.sum(Ia, 0)  # Folk på arbejde

            dSa = -beta * Sa * (I[:, i - 1] + Ia_tot) / population
            dIa = beta * Sa * (I[:, i - 1] + Ia_tot) / population - gammas * Ia
            dRa = gammas * Ia

            Sa += 1 / 2 * dSa;
            Ia += 1 / 2 * dIa;
            Ra += 1 / 2 * dRa

            dSdt = -beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population + 2 * np.sum(Sa, 1)
            dIdt = beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population - gammas * I[:, i - 1] + 2 * np.sum(Ia, 1)
            dRdt = gammas * I[:, i - 1] + 2 * np.sum(Ra, 1)

            Ny_smit[:, i] = beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population

        k += 1
        S[:, i] = S[:, i - 1] + 1 / 2 * dSdt
        I[:, i] = I[:, i - 1] + 1 / 2 * dIdt
        R[:, i] = R[:, i - 1] + 1 / 2 * dRdt
        P[:, i] = S[:, i] + I[:, i] + R[:, i]

    return S, I, R, P, Ny_smit


def SIQRS_SEG(z, t, beta=beta):
    rows, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((rows, l + 1))
    I = np.zeros((rows, l + 1))
    Q = np.zeros((rows, l + 1))
    R = np.zeros((rows, l + 1))
    P = np.zeros((rows, l))

    Sa = np.zeros((rows, rows))
    Ia = np.zeros((rows, rows))
    Qa = np.zeros((rows, rows))
    Ra = np.zeros((rows, rows))

    Ny_smit = np.zeros((rows, l + 1))

    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    Q[:, 0] = z[:, 2]
    R[:, 0] = z[:, 3]
    P[:, 0] = S[:, 0] + I[:, 0] + Q[:, 0] + R[:, 0]

    L = g * MT

    for i in range(1, l):
        if k % 2:  # Being home step
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population + alpha * R[:, i - 1] - 2 * g * S[:, i - 1]
            dIdt = (1 - r) * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - 2 * g * I[:, i - 1]
            dQdt = r * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * Q[:, i - 1]
            dRdt = gammas * I[:, i - 1] + gammas * Q[:, i - 1] - alpha * R[:, i - 1] - 2 * g * R[:, i - 1]

            Sa = (L * S[:, i - 1]).T
            Ia = (L * I[:, i - 1]).T
            Qa = Qa * 0.0
            Ra = (L * R[:, i - 1]).T

            Ny_smit[:, i] = beta * S[:, i - 1] * I[:, i - 1] / population

        else:  # Being on work step
            Ia_tot = np.sum(Ia, 0)  # Folk på arbejde

            dSa = -beta * Sa * (I[:, i - 1] + Ia_tot) / population + alpha * Ra
            dIa = (1 - r) * beta * Sa * (I[:, i - 1] + Ia_tot) / population - gammas * Ia
            dQa = r * beta * Sa * (I[:, i - 1] + Ia_tot) / population - gammas * Qa
            dRa = gammas * Ia + gammas * Qa - alpha * Ra

            Sa += 1 / 2 * dSa;
            Ia += 1 / 2 * dIa;
            Qa += 1 / 2 * dQa;
            Ra += 1 / 2 * dRa

            dSdt = -beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population + alpha * R[:, i - 1] + 2 * np.sum(Sa, 1)
            dIdt = (1 - r) * beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population - gammas * I[:,
                                                                                                 i - 1] + 2 * np.sum(Ia,
                                                                                                                     1)
            dQdt = r * beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population - gammas * Q[:, i - 1] + 2 * np.sum(Qa,
                                                                                                                    1)
            dRdt = gammas * I[:, i - 1] + gammas * Q[:, i - 1] - alpha * R[:, i - 1] + 2 * np.sum(Ra, 1)

            Ny_smit[:, i] = beta * S[:, i - 1] * (I[:, i - 1] + Ia_tot) / population

        k += 1
        S[:, i] = S[:, i - 1] + 1 / 2 * dSdt
        I[:, i] = I[:, i - 1] + 1 / 2 * dIdt
        Q[:, i] = Q[:, i - 1] + 1 / 2 * dQdt
        R[:, i] = R[:, i - 1] + 1 / 2 * dRdt
        P[:, i] = S[:, i] + I[:, i] + Q[:, i] + R[:, i]

    return S, I, Q, R, P, Ny_smit

z0 = [S0, I0, R0]
beta = np.array([0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333,
                0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333,
                0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333,
                0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333,
                0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333,
                0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.21666667,
                0.21666667, 0.21666667, 0.21666667, 0.21666667, 0.21666667,
                0.21666667, 0.21666667, 0.21666667, 0.21666667, 0.21666667,
                0.21666667, 0.21666667, 0.21666667, 0.21666667, 0.21666667,
                0.21666667, 0.175     , 0.175     , 0.175     , 0.175     ,
                0.175     , 0.175     , 0.175     , 0.175     , 0.175     ,
                0.175     , 0.175     , 0.175     , 0.175     , 0.175     ,
                0.175     , 0.175     , 0.175     , 0.175     , 0.175     ,
                0.175     , 0.175     , 0.175     , 0.05      , 0.05      ,
                0.05      , 0.05      , 0.05      , 0.05      , 0.05      ,
                0.05      , 0.05      , 0.05      , 0.05      , 0.05      ,
                0.05      , 0.05      , 0.05      , 0.05      , 0.05      ,
                0.05      , 0.05      , 0.21666667, 0.21666667, 0.21666667,
                0.21666667, 0.21666667, 0.21666667, 0.21666667, 0.21666667,
                0.21666667, 0.21666667, 0.21666667])

I0 = np.zeros(len(beta))
I0[0] = 23; I0[53] = 1; I0[78] = 25
S0 = population - I0
R0 = np.zeros(len(beta))
z0 = np.array([S0, I0, R0]).T
t = np.linspace(0,2000,2000)

S, I, R, P, ny = SIR_SEG(z0, t, beta=beta)

SS = np.sum(S, 0)
II = np.sum(I, 0)
RR = np.sum(R, 0)
ts = len(t)

plt.figure()
plt.plot(range(int(ts/2)), SS[0:ts:2], color = '#00BFFF', label='S')
plt.plot(range(int(ts/2)), II[0:ts:2], color = '#228B22', label='I')
plt.plot(range(int(ts/2)), RR[0:ts:2], color = '#B22222', label='R')
#plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.title("Fremskrivning af Segmenteret SIR med \n {}".format(r"$\beta = \{0.133, 0.217, 0.175, 0.05, 0.217 \}$"))
plt.legend(loc='best')
plt.grid()


beta = np.array([0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
       0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
       0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
       0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
       0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
       0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.15416667,
       0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
       0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
       0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
       0.15416667, 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
       0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
       0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
       0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
       0.2375    , 0.2375    , 0.2375    , 0.19583333, 0.19583333,
       0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
       0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
       0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
       0.19583333, 0.19583333, 0.09166667, 0.09166667, 0.09166667,
       0.09166667, 0.09166667, 0.09166667, 0.09166667, 0.09166667,
       0.09166667, 0.09166667, 0.09166667])
Q0 = np.zeros(len(beta))
z0 = np.array([S0, I0, Q0, R0]).T
S, I, Q, R, P, ny = SIQRS_SEG(z0, t, beta=beta)
SS = np.sum(S, 0)
II = np.sum(I, 0)
QQ = np.sum(Q, 0)
RR = np.sum(R, 0)

plt.figure()
plt.plot(range(int(ts/2)), SS[0:ts:2], color = '#00BFFF', label='S')
plt.plot(range(int(ts/2)), II[0:ts:2], color = '#228B22', label='I')
plt.plot(range(int(ts/2)), QQ[0:ts:2], color = '#FF8C00', label='Q')
#plt.plot(range(int(ts/2)), QQ[0:ts:2] + II[0:ts:2], label='I + Q')
plt.plot(range(int(ts/2)), RR[0:ts:2], color = '#B22222', label='R')
#plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
plt.ylabel("Antal mennesker")
plt.xlabel("t [Dage]")
plt.title("Fremskrivning af Segmenteret SIQRS med \n {}".format(r"$\beta = \{0.113, 0.092, 0.238, 0.154, 0.196 \}$"))
plt.legend(loc='best')
plt.grid()
plt.show()