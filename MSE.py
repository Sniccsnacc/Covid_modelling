#%% 

from Segmentation import *

betas = np.linspace(0.1, 0.3, 100)

err = []
new_beta = []

for i in betas:
    S, I, Q, R, P = SIQRS_SEG(Z0, t, beta = i)
    I_tot = np.sum(I,0)
    Q_tot = np.sum(Q,0)
    If = I_tot + Q_tot

    if abs(max(If) - max(sick)) <= 3000:
        new_beta.append(i)
        err.append(sum((If[0:ts:2] - sick) ** 2))

err = np.array(err)

ind = np.where(err == min(err))
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

betas = np.linspace(0.10, 0.25, 16)
beta2 = np.ones(len(names))
new_beta = []
err = []

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

                    if abs(max(If) - max(sick)) <= 3000:
                        new_beta.append(beta2.copy())
                        err.append(sum((If[0:ts:2] - sick) ** 2))
                    
    
    end2 = time.time()
    print(end2 - start2)

end = time.time()

print(end - start)

ind = np.where(err == min(err))[0][0]


