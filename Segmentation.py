#%%
import matplotlib.pyplot as plt
from Models import*
from Kom_dat import *

# num_regions = 3
# population = np.array([N, N, N])
# gammas = np.array([1/7, 1/7, 1/7])
# g = np.array([0.1, 0.3, 0.0])
# MT = np.array([[0, 0.8, 0.2],
#                       [0.3, 0, 0.7],
#                       [0.5, 0.5, 0]])

num_regions = len(travel_out)
population = np.array(population)
gammas = np.ones(num_regions) / 6
g = np.array(travel_out)
MT = np.array(travel_in).T
beta = 0.204 #Beta værdi fundet ved MSE mellem 0.1 og 0.3 (singulær)
beta = 0.210 #Beta værdi fundet ved MMSE mellem 0.1 og 0.3 (Thresholded) 
beta = 0.222

## Beta værdi fundet ved MSE mellem 0.15 og 0.25 med 6 punkter
# beta = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
#        0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
#        0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.13, 0.13, 0.13, 0.13,
#        0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,
#        0.13, 0.13, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19,
#        0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19,
#        0.19, 0.19, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
#        0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.13,
#        0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

# ## beta værdi fundet ved MMSE mellem 0.1 og 0.25 med 6 punkter
# beta = np.array([0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
#        0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
#        0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.22, 0.22, 0.22, 0.22,
#        0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
#        0.22, 0.22, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
#        0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
#        0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
#        0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.25,
#        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

# ## Beta værdi fundet ved MMSE mellem 0.08 og 0.28 med 7 punkter
# beta = np.array([0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.11333333,
#        0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.11333333,
#        0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.11333333,
#        0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.11333333,
#        0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.11333333,
#        0.11333333, 0.11333333, 0.11333333, 0.11333333, 0.21333333,
#        0.21333333, 0.21333333, 0.21333333, 0.21333333, 0.21333333,
#        0.21333333, 0.21333333, 0.21333333, 0.21333333, 0.21333333,
#        0.21333333, 0.21333333, 0.21333333, 0.21333333, 0.21333333,
#        0.21333333, 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.08      , 0.08      , 0.08      ,
#        0.08      , 0.08      , 0.24666667, 0.24666667, 0.24666667,
#        0.24666667, 0.24666667, 0.24666667, 0.24666667, 0.24666667,
#        0.24666667, 0.24666667, 0.24666667])

# ## Beta værdi fundet ved MMSE mellem 0.08 og 0.28 med 5 punkter (Thresholded)
# beta = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
#        0.08, 0.08, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
#        0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
#        0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23])

#Beta fundet mellem 0.05 og 0.3 med 13 punkter (Tresholded)
# beta = np.array([0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
#        0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
#        0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
#        0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
#        0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.1125    ,
#        0.1125    , 0.1125    , 0.1125    , 0.1125    , 0.15416667,
#        0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
#        0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
#        0.15416667, 0.15416667, 0.15416667, 0.15416667, 0.15416667,
#        0.15416667, 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
#        0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
#        0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
#        0.2375    , 0.2375    , 0.2375    , 0.2375    , 0.2375    ,
#        0.2375    , 0.2375    , 0.2375    , 0.19583333, 0.19583333,
#        0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
#        0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
#        0.19583333, 0.19583333, 0.19583333, 0.19583333, 0.19583333,
#        0.19583333, 0.19583333, 0.09166667, 0.09166667, 0.09166667,
#        0.09166667, 0.09166667, 0.09166667, 0.09166667, 0.09166667,
#        0.09166667, 0.09166667, 0.09166667])

#%% Models

def SIR_SEG(z, t, beta = beta):
    rows, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((rows, l+1))
    I = np.zeros((rows, l+1))
    R = np.zeros((rows, l+1))
    P = np.zeros((rows, l))

    Sa = np.zeros((rows, rows))
    Ia = np.zeros((rows, rows))
    Ra = np.zeros((rows, rows))

    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    P[:, 0 ] = S[:, 0] + I[: , 0] + R[:, 0]

    L = g * MT

    for i in range(1, l):
        if k%2: #Being home step
            dSdt = -beta*S[:,i-1]*I[:,i-1] / population - 2*g*S[:,i-1]
            dIdt = beta *S[:,i-1]*I[:,i-1] / population - gammas*I[:,i-1] - 2*g*I[:,i-1]
            dRdt = gammas*I[:,i-1] - 2*g*R[:,i-1]

            Sa = (L*S[:,i-1]).T
            Ia = (L*I[:,i-1]).T
            Ra = (L*R[:,i-1]).T

        else: #Being on work step
            Ia_tot = np.sum(Ia,0)   #Folk på arbejde

            dSa = -beta * Sa *(I[:,i-1] + Ia_tot)/population
            dIa = beta * Sa *(I[:,i-1] + Ia_tot)/population - gammas * Ia
            dRa = gammas * Ia

            Sa += 1/2 *dSa; Ia += 1/2*dIa; Ra += 1/2*dRa

            dSdt = -beta* S[:,i-1] * (I[:,i-1] + Ia_tot) / population + 2*np.sum(Sa, 1)
            dIdt = beta *S[:,i-1] * (I[:,i-1] + Ia_tot) / population - gammas*I[:,i-1] + 2*np.sum(Ia, 1)
            dRdt = gammas * I[:,i-1] + 2*np.sum(Ra, 1)

        
        k += 1
        S[:, i] = S[:, i-1] + 1/2 * dSdt
        I[:, i] = I[:, i-1] + 1/2 * dIdt
        R[:, i] = R[:, i-1] + 1/2 * dRdt
        P[:, i] = S[:, i] + I[: , i] + R[:, i]

    return S, I, R, P


def SIQRS_SEG(z, t, beta = beta):
    rows, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((rows, l+1))
    I = np.zeros((rows, l+1))
    Q = np.zeros((rows, l+1))
    R = np.zeros((rows, l+1))
    P = np.zeros((rows, l))

    Sa = np.zeros((rows, rows))
    Ia = np.zeros((rows, rows))
    Qa = np.zeros((rows, rows))
    Ra = np.zeros((rows, rows))

    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    Q[:, 0] = z[:, 2]
    R[:, 0] = z[:, 3]
    P[:, 0 ] = S[:, 0] + I[: , 0] + Q[:, 0] + R[:, 0]

    L = g * MT

    for i in range(1, l):
        if k%2: #Being home step
            dSdt = -beta*S[:,i-1]*I[:,i-1] / population + alpha*R[:, i-1] - 2*g*S[:,i-1]
            dIdt = (1-r)*beta *S[:,i-1]*I[:,i-1] / population - gammas*I[:,i-1] - 2*g*I[:,i-1]
            dQdt = r*beta*S[:,i-1]*I[:,i-1] / population - gammas*Q[:,i-1]
            dRdt = gammas*I[:,i-1] + gammas*Q[:,i-1] - alpha*R[:,i-1] - 2*g*R[:,i-1]

            Sa = (L*S[:,i-1]).T
            Ia = (L*I[:,i-1]).T
            Qa = Qa*0.0
            Ra = (L*R[:,i-1]).T

        else: #Being on work step
            Ia_tot = np.sum(Ia,0)   #Folk på arbejde

            dSa = -beta * Sa *(I[:,i-1] + Ia_tot)/population + alpha * Ra
            dIa = (1-r)*beta * Sa *(I[:,i-1] + Ia_tot)/population - gammas * Ia
            dQa = r*beta* Sa *(I[:,i-1] + Ia_tot)/population - gammas * Qa
            dRa = gammas * Ia + gammas * Qa - alpha * Ra

            Sa += 1/2 *dSa; Ia += 1/2*dIa; Qa += 1/2*dQa; Ra += 1/2*dRa

            dSdt = -beta* S[:,i-1] * (I[:,i-1] + Ia_tot) / population + alpha * R[:,i-1] + 2*np.sum(Sa, 1)
            dIdt = (1-r)*beta *S[:,i-1] * (I[:,i-1] + Ia_tot) / population - gammas*I[:,i-1] + 2*np.sum(Ia, 1)
            dQdt = r*beta*S[:,i-1]*(I[:,i-1] + Ia_tot) / population - gammas*Q[:,i-1] + 2*np.sum(Qa, 1)
            dRdt = gammas * I[:,i-1] + gammas*Q[:,i-1] - alpha*R[:,i-1] + 2*np.sum(Ra, 1)

        
        k += 1
        S[:, i] = S[:, i-1] + 1/2 * dSdt
        I[:, i] = I[:, i-1] + 1/2 * dIdt
        Q[:, i] = Q[:, i-1] + 1/2 * dQdt
        R[:, i] = R[:, i-1] + 1/2 * dRdt
        P[:, i] = S[:, i] + I[: , i] + Q[:, i] + R[:, i]

    return S, I, Q, R, P

        



#%% Starting conditions

I0 = np.zeros(num_regions)

#23 i Kbh, 1 i Odense og 25 i Aarhus
I0[0] = 23; I0[53] = 1; I0[78] = 25     #udtræk fra 09/03-2020

Q0 = np.zeros(num_regions)
R0 = np.zeros(num_regions)
S0 = population - I0 - Q0 - R0 #np.ones(num_regions) * (N - 1)
Z0 = np.transpose(np.array([S0, I0, Q0, R0]))
ts = t.size

S, I, Q, R, P = SIQRS_SEG(Z0, t)


#%% Plotting 3 first cities

names[0] = "København"; names[14] = "Lyngby-Taarbæk"


plt.figure()
#plt.plot(range(int(ts/2)), S[0, 0:ts:2], color = '#00BFFF', label='S')
plt.plot(range(int(ts/2)), I[0, 0:ts:2], color = '#228B22', label='I')
plt.plot(range(int(ts/2)), Q[0, 0:ts:2], color = '#FF8C00', label='Q')
#plt.plot(range(int(ts/2)), R[0, 0:ts:2], color = '#B22222', label='R')
plt.plot(range(int(ts/2)), I[0, 0:ts:2] + Q[0, 0:ts:2], label='I + Q')
plt.ylabel("Antal mennesker")
plt.xlabel("t [Dage]")
plt.legend(loc='right')
plt.title(r"Segmenteret SIQRS model med $\beta$={} i {}".format(beta,names[0]))
plt.grid()

plt.figure()
#plt.plot(range(int(ts/2)), S[14, 0:ts:2], color = '#00BFFF', label='S')
plt.plot(range(int(ts/2)), I[14, 0:ts:2], color = '#228B22', label='I')
plt.plot(range(int(ts/2)), Q[14, 0:ts:2], color = '#FF8C00', label='Q')
#plt.plot(range(int(ts/2)), R[14, 0:ts:2], color = '#B22222', label='R')
plt.plot(range(int(ts/2)), I[14, 0:ts:2] + Q[14, 0:ts:2], label='I + Q')
plt.ylabel("Antal mennesker")
plt.xlabel("t [Dage]")
plt.legend(loc='right')
plt.title(r"Segmenteret SQIRS model med $\beta$={} i {}".format(beta,names[14]))
plt.grid()


# for i in range(6):
#     plt.figure()
#     plt.plot(range(len(sick)), S[i, 0:ts:2], color = '#00BFFF', label='S')
#     plt.plot(range(len(sick)), I[i, 0:ts:2], color = '#228B22', label='I')
#     plt.plot(range(len(sick)), Q[i, 0:ts:2], color = '#FF8C00', label='Q')
#     plt.plot(range(len(sick)), R[i, 0:ts:2], color = '#B22222', label='R')
#     #plt.plot(t[0:ts:2], P[i, 0:ts:2], label='P')
#     plt.ylabel(names[i])
#     plt.legend(loc='best')
#     plt.grid()


# plt.show()


S_tot = np.sum(S,0)
I_tot = np.sum(I,0)
Q_tot = np.sum(Q,0)
R_tot = np.sum(R,0)
P_tot = np.sum(P,0)
If = I_tot + Q_tot


plt.figure()
plt.plot(range(len(sick)), sick, color = '#B22222', label='Data')
#plt.plot(range(int(ts/2)), S_tot[0:ts:2], color = '#00BFFF', label='S')
plt.plot(range(int(ts/2)), I_tot[0:ts:2], color = '#228B22', label='I')
plt.plot(range(int(ts/2)), Q_tot[0:ts:2], color = '#FF8C00', label='Q')
#plt.plot(range(int(ts/2)), R_tot[0:ts:2], color = '#B22222', label='R')
#plt.plot(range(int(ts/2)), P_tot[0:ts:2], label='Population')
plt.plot(range(int(ts/2)), If[0:ts:2], label = 'I + Q')
plt.ylabel("Antal mennesker")
plt.xlabel("t [Dage]")
plt.legend(loc='upper left')
plt.title("Segmenteret SIQRS model på dansk data med \n {}".format(r"$\beta = \{0.113, 0.092, 0.238, 0.154, 0.196 \}$"))
#plt.title(r"Segmenteret SIQRS model på dansk data med $\beta$ = {}".format(beta))
plt.grid()
plt.show()


     # %%
#Plot af hældning
plt.figure()
plt.plot(range(len(cases)), cases)
plt.grid()
plt.show()

# %%
