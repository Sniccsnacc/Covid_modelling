from Parameters import*

# region SIR model
def SIR(z, t):
    dSdt = -beta *z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt
# endregion

# region SIQRS model
def SIQRS(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * (z[2] + z[3])
    dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[2]
    dQdt = mu * z[1] - gamma * z[2]
    dRdt = gamma * z[1] - alpha * z[3]
    return [dSdt, dIdt, dQdt, dRdt]
# endregion

# region SIQRSV model
def SIQRSV(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[3]
    dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[1]
    dQdt = mu * z[1] - gamma * z[2] - zeta * z[2]
    dRdt = gamma * (z[1] + z[2]) - alpha * z[3] - zeta * z[3]
    dVdt = zeta * (z[2] + z[3])
    return [dSdt, dIdt, dQdt, dRdt, dVdt]
# endregion

# region Quar, SIRS with quarantine after a given number of infected
def Quar(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[3]
    dIdt = beta * z[0] * z[1] - gamma * z[1]
    dQdt = 0
    dRdt = gamma * z[1] - alpha * z[3]

    if z[1] >= 1600:
        dIdt = beta * z[0] * z[1] - gamma * z[1] - mu * z[2]
        dQdt = mu * z[1] - gamma * z[2]
        dRdt = gamma * z[1] - alpha * z[3]
    elif z[2] > 0:
        dQdt = - gamma * z[2]
    return [dSdt, dIdt, dQdt, dRdt]
# endregion




# region Explixit euler just for fun

def ExplicitEuler(fun, x0, tspan):
    nx = len(x0)
    X = np.zeros((nx, tspan.size), dtype='float')
    T = np.zeros(tspan.size, dtype='float')

    X[:, 0] = x0
    for k in range(tspan.size - 1):
        f = np.array(fun(X[:, k], tspan), dtype=float)
        dt = tspan[k+1] - tspan[k]
        X[:, k+1] = X[:, k] + f * dt

    return X.T
# endregion

