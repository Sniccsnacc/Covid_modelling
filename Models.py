from Parameters import*


# region SIR model
def SIR(z, t):
    dSdt = -beta *z[0]*z[1]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1]
    dzdt = [dSdt, dIdt, dRdt]
    return dzdt
# endregion

# region SIS model
def SIS(z, t):
    dSdt = -beta * z[0] * z[1] + alpha * z[2]
    dIdt = beta*z[0]*z[1] - gamma * z[1]
    dRdt = gamma * z[1] - alpha * z[2]
    return [dSdt, dIdt, dRdt]
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

# region Explicit Euler
def ExplicitEuler(fun, xa, Tspan):
    def feval(funcName, *args):
        return eval(funcName)(*args)

    nx = len(xa)
    N = len(Tspan)
    X = np.zeros((N, nx))
    T = np.zeros(N)

    #Explicit euler method
    T[0] = Tspan[0]
    X[0, :] = xa

    for k in range(N-1):
        f = np.array([feval(fun, X[k, :], T[k])])
        T[k+1] = Tspan[k+1]
        dt = T[k+1] - T[k]
        X[k+1, :] = X[k, :] + f*dt

    return X
# endregion

