#%%
import matplotlib.pyplot as plt
from Models import*
from Kom_dat import *

# num_regions = 3
# population = np.array([N, N, N])
# gammas = np.array([1/7, 1/7, 1/7])
# travel_out = np.array([0.1, 0.3, 0.0])
# travel_in = np.array([[0, 0.8, 0.2],
#                       [0.3, 0, 0.7],
#                       [0.5, 0.5, 0]])

num_regions = len(travel_out)
population = np.array(population) * 2
gammas = np.ones(num_regions) / 6
travel_out = np.array(travel_out)
travel_in = np.array(travel_in)
#beta = 0.188 #Beta value for SIR and SIRS model
beta = np.array([0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ,
       0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ,
       0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
       0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
       0.15, 0.15, 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ,
       0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.15,
       0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])




#%% Models

def SIR_SEG(z, t):
    r, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((r, l+1))
    I = np.zeros((r, l+1))
    R = np.zeros((r, l+1))
    P = np.zeros((r,l))
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    P[:, 0 ] = S[:, 0] + I[: , 0] + R[:, 0]
    for i in range(1, t.size):
        #Standard form:
        # dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
        # dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
        # dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])

        #Homecoming form:
        if k%2:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
            dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
            dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])
        else:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population + travel_out * S[:, i - 2] - (travel_out * travel_in.T).dot(S[:, i - 2])
            dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 2] + travel_out * I[:, i - 1] - (travel_out * travel_in.T).dot(I[:, i - 2])
            dRdt = gammas * I[:, i - 1] + travel_out * R[:, i - 2] - (travel_out * travel_in.T).dot(R[:, i - 2])

        k += 1

        S[:, i] = S[:, i-1] + 1/2 * dSdt
        I[:, i] = I[:, i-1] + 1/2 * dIdt
        R[:, i] = R[:, i-1] + 1/2 * dRdt
        P[:, i] += S[:, i] + I[: , i] + R[:, i]

        

    return S, I, R, P

def SIRS_SEG(z, t):
    r, c = z.shape
    l = t.size
    k = 1
    S = np.zeros((r, l+1))
    I = np.zeros((r, l+1))
    R = np.zeros((r, l+1))
    P = np.zeros((r,l))
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    R[:, 0] = z[:, 2]
    P[:, 0 ] = S[:, 0] + I[: , 0] + R[:, 0]
    for i in range(1, t.size):
        #Standard form:
        # dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
        # dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
        # dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])

        #Homecoming form:
        if k%2:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1]) + alpha * R[:,i-1]
            dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
            dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1]) - alpha * R[:, i-1]
        else:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population + travel_out * S[:, i - 2] - (travel_out * travel_in.T).dot(S[:, i - 2]) + alpha *R[:,i-1]
            dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 2] + travel_out * I[:, i - 1] - (travel_out * travel_in.T).dot(I[:, i - 2])
            dRdt = gammas * I[:, i - 1] + travel_out * R[:, i - 2] - (travel_out * travel_in.T).dot(R[:, i - 2]) - alpha *R[:,i-1]

        k += 1

        S[:, i] = S[:, i-1] + 1/2 * dSdt
        I[:, i] = I[:, i-1] + 1/2 * dIdt
        R[:, i] = R[:, i-1] + 1/2 * dRdt
        P[:, i] += S[:, i] + I[: , i] + R[:, i]

        

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
    S[:, 0] = z[:, 0]
    I[:, 0] = z[:, 1]
    Q[:, 0] = z[:, 2]
    R[:, 0] = z[:, 3]
    P[:, 0 ] = S[:, 0] + I[: , 0] + Q[:, 0] + R[:, 0]
    for i in range(1, t.size):
        #Standard form:
        # dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population - travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1])
        # dIdt = beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
        # dRdt = gammas * I[:, i - 1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1])

        #Homecoming form:
        if k%2:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population -travel_out * S[:, i - 1] + (travel_out * travel_in.T).dot(S[:, i - 1]) + alpha * R[:,i-1]
            dIdt = (1 - r) * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 1] - travel_out * I[:, i - 1] + (travel_out * travel_in.T).dot(I[:, i - 1])
            dQdt = r * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * Q[:, i - 1]
            dRdt = gammas * I[:, i - 1] + gammas * Q[:, i-1] - travel_out * R[:, i - 1] + (travel_out * travel_in.T).dot(R[:, i - 1]) - alpha * R[:, i-1]
        else:
            dSdt = -beta * S[:, i - 1] * I[:, i - 1] / population + travel_out * S[:, i - 2] - (travel_out * travel_in.T).dot(S[:, i - 2]) + alpha *R[:,i-1]
            dIdt = (1-r) * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * I[:, i - 2] + travel_out * I[:, i - 1] - (travel_out * travel_in.T).dot(I[:, i - 2])
            dQdt = r * beta * S[:, i - 1] * I[:, i - 1] / population - gammas * Q[:, i - 1]
            dRdt = gammas * I[:, i - 1] + gammas * Q[:, i-1] + travel_out * R[:, i - 2] - (travel_out * travel_in.T).dot(R[:, i - 2]) - alpha *R[:,i-1]

        k += 1
        S[:, i] = S[:, i-1] + 1/2 * dSdt
        I[:, i] = I[:, i-1] + 1/2 * dIdt
        Q[:, i] = Q[:, i-1] + 1/2 * dQdt
        R[:, i] = R[:, i-1] + 1/2 * dRdt
        P[:, i] += S[:, i] + I[: , i] + Q[:, i] + R[:, i]

        

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

# fig1 = plt.figure(1)

# fig1.add_subplot(3, 1, 1)
# plt.plot(t, S[0, 0:ts], color = '#00BFFF', label='S')
# plt.plot(t, I[0, 0:ts], color = '#228B22', label='I')
# plt.plot(t, R[0, 0:ts], color = '#B22222', label='R')
# plt.plot(t, P[0, 0:ts], label='P')
# plt.ylabel(names[0])
# #plt.legend(loc='best')
# plt.grid()

# fig1.add_subplot(3, 1, 2)
# plt.plot(t, S[1, 0:ts], color = '#00BFFF', label='S')
# plt.plot(t, I[1, 0:ts], color = '#228B22', label='I')
# plt.plot(t, R[1, 0:ts], color = '#B22222', label='R')
# plt.plot(t, P[1, 0:ts], label='P')
# plt.ylabel(names[1])
# #plt.legend(loc='best')
# plt.grid()

# fig1.add_subplot(3, 1, 3)
# plt.plot(t, S[2, 0:ts], color = '#00BFFF', label='S')
# plt.plot(t, I[2, 0:ts], color = '#228B22', label='I')
# plt.plot(t, R[2, 0:ts], color = '#B22222', label='R')
# plt.plot(t, P[2, 0:ts], label='P')
# plt.ylabel(names[2])
# #plt.legend(loc='best')
# plt.grid()

# plt.tight_layout()

# for i in range(num_regions):
#     plt.figure()
#     plt.plot(t[0:ts:2], S[i, 0:ts:2], color = '#00BFFF', label='S')
#     plt.plot(t, I[i, 0:ts], color = '#228B22', label='I')
#     plt.plot(t, R[i, 0:ts], color = '#B22222', label='R')
#     plt.plot(t[0:ts:2], P[i, 0:ts:2], label='P')
#     plt.ylabel(names[i])
#     plt.legend(loc='best')
#     plt.grid()


# plt.show()

#%% Total plot:
S_tot = np.sum(S,0)
I_tot = np.sum(I,0)
Q_tot = np.sum(Q,0)
R_tot = np.sum(R,0)
P_tot = np.sum(P,0)
If = I_tot + Q_tot


plt.figure()
#plt.plot(t[0:ts:2], S_tot[0:ts:2], color = '#00BFFF', label='S')
#plt.plot(range(len(sick)), I_tot[0:ts:2], color = '#228B22', label='I')
#plt.plot(range(len(sick)), Q_tot[0:ts:2], color = '#FF8C00', label='Q')
#plt.plot(range(len(sick)), R_tot[0:ts], color = '#B22222', label='R')
#plt.plot(t[0:ts:2], P_tot[0:ts:2], label='Population')
plt.plot(range(len(sick)), If[0:ts:2], label = 'Inficerede')
plt.plot(range(len(sick)), sick, label='Positivt testede')
plt.ylabel("Antal Mennesker")
plt.xlabel("Dage")
plt.legend(loc='best')
plt.title(r"Segmenteret SIQRS model") #med $\beta$={}".format(beta))
plt.grid()

plt.show()

 # %%

# %%
