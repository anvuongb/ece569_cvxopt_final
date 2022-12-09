import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *

# set params
# QoS
rho_min_qos_sigma2 = 1 #rho_min_i * sigma_i^2 = 1 for all i

# Max-Min
sigma_i = 1 

# power
P = 1
N_list = [4, 4, 8, 8]
M_list = [8, 16, 16, 32]

# perform calculations
for N, M in zip(N_list, M_list):
    print("Solving N/M = {}/{}".format(N, M))

    best_t = []
    best_t_sdr = []
    best_t_avg = []
    no_bfm = []

    w_nobmf = (1/np.sqrt(N))*np.ones(N).reshape((N,-1))

    # Monte Carlo 1000 runs
    for k in tqdm.tqdm(range(1000)):
        # regenerate channel, no path loss
        h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin = generate_channels(None, None, None, N, M, rho_min_qos_sigma2=rho_min_qos_sigma2, sigma_i=sigma_i)

        ### MAXMIN SDR ###
        W = cp.Variable((N, N), hermitian=True)
        t = cp.Variable(1)

        constraints = [W >> 0]
        constraints += [cp.real(cp.trace(W)) == P] # power constraint
        constraints += [cp.real(cp.trace(cp.matmul(W, H_array_norm_MaxMin[i]))) >= t for i in range(M)]

        prob = cp.Problem(cp.Maximize(t),
                     constraints)
        opt = prob.solve(solver="MOSEK", verbose=False)
        W_opt = W.value

        best_t.append(t.value[0])
        
        max_minSNR = -np.inf
        for i in range(30*N*M):
            # recover randomization
            w_randA = recover_w_randA(W_opt)
            w_randB = recover_w_randB(W_opt)
            w_randC = recover_w_randC(W_opt)
            
            # scale to norm P
            w_randA /= np.linalg.norm(w_randA)/np.sqrt(P)
            w_randB /= np.linalg.norm(w_randB)/np.sqrt(P)
            w_randC /= np.linalg.norm(w_randC)/np.sqrt(P)

            max_minSNR_tmp = np.max([np.min(abs(np.matmul(np.conjugate(w_randA).T, h_array_norm_MaxMin)).ravel()),
                                     np.min(abs(np.matmul(np.conjugate(w_randB).T, h_array_norm_MaxMin)).ravel()),
                                     np.min(abs(np.matmul(np.conjugate(w_randC).T, h_array_norm_MaxMin)).ravel())
                                   ])
            max_minSNR = np.max([max_minSNR, max_minSNR_tmp])
        best_t_sdr.append(max_minSNR**2)
        ### END MAXMIN SDR ###

        ### MAX AVERAGE SDR ###
        W = cp.Variable((N, N), hermitian=True)

        constraints = [W >> 0]
        constraints += [cp.real(cp.trace(W)) == 1] # power constraint
        snr_list = [cp.real(cp.trace(cp.matmul(W, H_array_norm_MaxMin[i]))) for i in range(M)]

        prob = cp.Problem(cp.Maximize(cp.sum(snr_list)),
                     constraints)
        opt = prob.solve(solver="MOSEK", verbose=False)
        W_opt = W.value
        
        best_t_avg.append(np.min([np.real(np.trace(np.matmul(W_opt, H_array_norm_MaxMin[i]))) for i in range(M)]))
        ### END AVERAGE SDR ###

        ### NO BEAMFORMING ###
        no_bfm.append(np.min(abs(np.matmul(np.conjugate(w_nobmf).T, h_array_norm_MaxMin)).ravel())**2)
        
    print("upper bound =", np.mean(best_t))
    print("sdr =", np.mean(best_t_sdr))
    print("avg =", np.mean(best_t_avg))
    print("no bmf =", np.mean(no_bfm))

    # format for latex copy paste
    print("{:.4f} & {:.4f} & {:.4f} & {:.4f}".format(np.mean(best_t), np.mean(best_t_sdr), np.mean(best_t_avg), np.mean(no_bfm)))
    print("\n\n")
    