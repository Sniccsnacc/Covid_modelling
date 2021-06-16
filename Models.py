from Parameters import*


# region SIR model
def SIR(z, t, beta = beta):
    dSdt = -beta * z[0] * z[1] / N
    dIdt = beta * z[0] * z[1] / N - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt
# endregion

# region SIR model birth and death rate
def SIRBD(z, t):
    dSdt = tau - psi * z[0] / N - beta * z[0] * z[1] / N
    dIdt = beta * z[0] * z[1] / N - gamma * z[1]
    dRdt = gamma * z[1] - psi * z[2] / N
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt
# endregion



# region SIRS model
def SIS(z, t, alpha=alpha, beta=beta):
    dSdt = -beta * z[0] * z[1] / N + alpha * z[2]
    dIdt = beta*z[0]*z[1] / N - gamma * z[1]
    dRdt = gamma * z[1] - alpha * z[2]
    return [dSdt, dIdt, dRdt]
# endregion

# region SIQRS model
def SIQRS(z, t, alpha=alpha, mu=mu, beta=beta):
    dSdt = -beta * z[0] * z[1] / N + alpha * z[3]
    dIdt = beta * z[0] * z[1] / N - (gamma + mu) * z[1]
    dQdt = mu * z[1] - gamma * z[2]
    dRdt = gamma * (z[1] + z[2]) - alpha * z[3]
    return [dSdt, dIdt, dQdt, dRdt]
# endregion

# region SIQRSV model
def SIQRSV(z, t):
    dSdt = -beta * z[0] * z[1] - zeta * z[0] + alpha * z[3]
    dIdt = beta * z[0] * z[1] - (gamma + mu) * z[1]
    dQdt = mu * z[1] - gamma * z[2]
    dRdt = gamma * (z[1] + z[2]) - alpha * z[3] - zeta * z[3]
    dVdt = zeta * (z[0] + z[3])
    return [dSdt, dIdt, dQdt, dRdt, dVdt]
# endregion

# region Quar, SIRS with quarantine after a given number of infected
def Quar(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[2]
    dIdt = beta * z[0] * z[1] - gamma * z[1]
    dQdt = 0
    dRdt = gamma * z[1] - alpha * z[2]

    if z[1] >= quar_thresshold:
        dIdt = beta * z[0] * z[1] - (gamma + mu) * z[1]
        dQdt = mu * z[1] - gamma * z[2]
        dRdt = gamma * (z[1] + z[2]) - alpha * z[3]
    elif z[2] > 0:
        dQdt = - gamma * z[2]
    return [dSdt, dIdt, dQdt, dRdt]
# endregion


# region Mads function
def mads(z, t):
    dSdt = -beta * z[0] * z[1] / N + alpha * z[3]
    dIdt = beta * z[0] * z[1] / N - (1-r) * gamma * z[1] - r * gamma * z[1]
    dQdt = r * gamma * z[1] - gamma * z[2]
    dRdt = (1 - r) * gamma * z[1] - alpha * z[3] + gamma * z[2]
    return [dSdt, dIdt, dQdt, dRdt]


def co(z, t, alpha=alpha, r = r, beta = beta):
    dSdt = - (beta * z[0] * z[2] / N) + alpha * z[3]
    dIqdt = beta * z[0] * z[2] * r / N - gamma * z[1]
    dIidt = beta * z[0] * z[2] * (1-r) / N - gamma * z[2]
    dRdt = gamma * z[1] + gamma * z[2] - alpha * z[3]
    return [dSdt, dIqdt, dIidt, dRdt]

# region Extra functions

# region Explicit Euler
def ExplicitEuler_co(z0, t, alpha=alpha, r=r):
    l = t.size
    dt = t[1]-t[0]
    S = np.zeros(l)
    Iq = np.zeros(l)
    Ii = np.zeros(l)
    R = np.zeros(l)
    S[0] = z0[0]
    Iq[0] = z0[1]
    Ii[0] = z0[2]
    R[0] = z0[3]
    for i in range(1, t.size):
        dSdt = - (beta * S[i-1] * Ii[i-1] / N) + alpha * R[i-1]
        dIqdt = beta * S[i-1] * Ii[i-1] * r / N - gamma * Iq[i-1]
        dIidt = beta * S[i-1] * Ii[i-1] * (1 - r) / N - gamma * Ii[i-1]
        dRdt = gamma * S[i-1] + gamma * Ii[i-1] - alpha * R[i-1]

        S[i] = S[i - 1] + dSdt
        Iq[i] = Iq[i - 1] + dIqdt * dt
        Ii[i] = Iq[i - 1] + dIidt * dt
        R[i] = R[i - 1] + dRdt

    return S, Iq, Ii, R

# endregion

# Trapazoid
def trapezoid(fx, dt):
    sum = 0
    for i in range(len(fx) - 1):
        sum += 1/2 * dt * (fx[i] + fx[i+1])
    return sum
# endregion



