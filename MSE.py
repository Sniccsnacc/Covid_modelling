#%% 

from Segmentation import *

betas = np.linspace(0.1, 0.3, 101)
t = np.linspace(0, 2*498, 2*498)
Z0 = np.transpose(np.array([S0, I0, Q0, R0]))

err = []
new_beta = []
m = max(sick)
sind = np.where(sick == m)[0][0]

for i in betas:
    S, I, Q, R, P = SIQRS_SEG(Z0, t, beta = i)
    I_tot = np.sum(I,0)
    Q_tot = np.sum(Q,0)
    If = I_tot + Q_tot

    mIf = max(If)
    mI = max(I_tot)

    if mIf >= m and abs(np.where(If == mIf)[0][0] - 2*sind) <= 10:
        new_beta.append(i)
        err.append(sum((If[0:ts:2] - sick) ** 2) / len(sick))

err = np.array(err)

ind = np.where(err == min(err))[0][0]
print(r"Minimal error is at beta = ", end = ' ')
print(new_beta[ind])

#%% 
from Segmentation import *
import time

n_h = 29
n_s = 17
n_sd = 22
n_mj = 19
n_nj = 11

betas = np.linspace(0.05, 0.3, 13)
beta2 = np.ones(len(names))
new_beta = []
err = []
m = max(sick)

start = time.time()

for hoved in betas:
    start2 = time.time()
    for sjael in betas:
        for syddan in betas:
            for midtjyl in betas:
                for nordjyl in betas:
                    beta2[0:n_h] = hoved
                    beta2[n_h:n_h+n_s] = sjael
                    beta2[n_h + n_s:n_h + n_s + n_sd] = syddan
                    beta2[n_h + n_s + n_sd:n_h + n_s + n_sd + n_mj] = midtjyl
                    beta2[n_h + n_s + n_sd + n_mj:n_h + n_s + n_sd + n_mj + n_nj] = nordjyl
                    
                    S, I, Q, R, P = SIQRS_SEG(Z0, t, beta = beta2)
                    I_tot = np.sum(I,0)
                    Q_tot = np.sum(Q,0)
                    If = I_tot + Q_tot
                    
                    if max(If) >= m:
                        new_beta.append(beta2.copy())
                        err.append(sum((If[0:ts:2] - sick) ** 2) /len(sick) )
                    
    
    end2 = time.time()
    print(end2 - start2)

end = time.time()

print(end - start)

ind = np.where(err == min(err))[0][0]



# %%
#### Simple SIRQS #####

from Models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.linspace(0, 498, 498)

I0 = 49 #Dansk data udtræk
Q0 = 0
R0 = 0
S0 = N - I0 - Q0 - R0
z0 = [S0, Q0, I0, R0]
m = max(sick)
sind = np.where(sick == m)[0][0]

betas = np.linspace(0.1, 0.3, 101)
err = []
new_beta = []

for i in betas:
    Z = odeint(co, z0, t, args = (alpha, r, i))
    If = Z[:, 1] + Z[:, 2]
    mIf = max(If)

    if mIf >= m and abs(np.where(If == mIf)[0][0] - sind) <= 10:
        new_beta.append(i)
        err.append( sum((If - sick) ** 2) / len(sick))


err = np.array(err)

ind = np.where(err == min(err))[0][0]
print(r"Minimal error is at beta = ", end = ' ')
print(new_beta[ind])

z = odeint(co, z0, t, args = (alpha, r, new_beta[ind]))


plt.figure()
plt.plot(t, z[:, 2], color = '#228B22', label='$I$')
plt.plot(t, z[:, 1], color = '#FF8C00', label='$Q$')
plt.plot(t, z[:, 1] + z[:, 2], label = 'I + Q')
plt.plot(t, sick, color = '#B22222', label='Data')
plt.ylabel('Antal mennesker')
plt.xlabel('t [dage]')
plt.legend(loc='best')
plt.title(r'SIQRS model på dansk data med $\beta$ = 0.224')
plt.grid()
plt.show()