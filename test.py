#%%
# import multiprocessing

# def worker(num):
#     """Returns the string of interest"""
#     num2 = num ** 2
#     return "worker {},{}".format(num, num2)

# def main():
#     tot = []
#     for i in range(1,5):
#         pool = multiprocessing.Pool(4)
#         results = pool.map(worker, range(i))

#         pool.close()
#         pool.join()

#         for result in results:
#             # prints the result string in the main process
#             tot.append(result)
            
#     print(*tot)

# if __name__ == '__main__':
#     # Better protect your main function when you use multiprocessing
#     main()

#%%  

from Segmentation import *
import time
import multiprocessing as mp

n_h = 29
n_s = 17
n_sd = 22
n_mj = 19
n_nj = 11

ho = 0; sj = 0; sd = 0; mi = 0

betas = np.linspace(0.05, 0.3, 13)
beta2 = np.ones(len(names))
new_beta = []
err = []
m = max(sick)
Z0 = np.transpose(np.array([S0, I0, R0]))

def err_calc(c1):
    beta2[0:n_h] = c1
    temp_beta = []
    err_temp = []
    for c2 in betas:
        for c3 in betas:
            for c4 in betas:
                for c5 in betas:
                    beta2[n_h:n_h+n_s] = c2
                    beta2[n_h + n_s:n_h + n_s + n_sd] = c3
                    beta2[n_h + n_s + n_sd:n_h + n_s + n_sd + n_mj] = c4
                    beta2[n_h + n_s + n_sd + n_mj:n_h + n_s + n_sd + n_mj + n_nj] = c5
    
                    S, I, R, P = SIR_SEG(Z0, t, beta = beta2)
                    I_tot = np.sum(I,0)
                    #Q_tot = np.sum(Q,0)
                    #If = I_tot + Q_tot
                    
                    if max(I_tot) >= m:
                        temp_beta.append(beta2.copy())
                        err_temp.append(sum((I_tot[0:ts:2] - sick) ** 2) /len(sick))
    
    return [temp_beta, err_temp]

def main():

    pool = mp.Pool(7)
    results = pool.map(err_calc, betas)

    pool.close()
    pool.join()

    for res in results:
        for i in range(len(res[0])):
            new_beta.append(res[0][i])
            err.append(res[1][i])
        


    ind = np.where(err == min(err))[0][0]
    return ind

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    start = time.time()
    ind = main()
    end = time.time()
    print("Time elapsed: {} [s]".format(end-start))
 