import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *

## Set params
# QoS
rho_min_qos_sigma2 = 1 #rho_min_i * sigma_i^2 = 1 for all i
# Max-Min
sigma_i = 1 

N_list = [4, 4, 8, 8]
M_list = [8, 16, 16, 32]

for N, M in zip(N_list, M_list):
    # Generate LoS channels
    h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin = generate_channels(None, None, None, N, M, rho_min_qos_sigma2=rho_min_qos_sigma2, sigma_i=sigma_i)

    # PERFORM QOS OPTIMIZATION #
    W = cp.Variable((N, N), hermitian=True)

    # need to use cp.real since cp.trace will give imaginary part = 0j, which will break the code
    constraints = [W >> 0]
    constraints += [cp.real(cp.trace(cp.matmul(W, H_array_norm_QoS[i]))) >= 1 for i in range(M)]

    prob = cp.Problem(cp.Minimize( cp.real(cp.trace(W))),
                constraints)
    opt = prob.solve(solver="MOSEK", verbose=False)
    W_opt = W.value

    print("optimal objective value = ", opt)
    # print("optimal beamforming vector = ", W_opt)

    ubpb = []

    for i in range(1000):
        # recover randomization
        w_randA = recover_w_randA(W_opt)
        w_randB = recover_w_randB(W_opt)
        w_randC = recover_w_randC(W_opt)

        scale_factor_randA = get_min_scale_factor(w_randA, h_array_norm_QoS)
        scale_factor_randB = get_min_scale_factor(w_randB, h_array_norm_QoS)
        scale_factor_randC = get_min_scale_factor(w_randC, h_array_norm_QoS)

        w_randA *= scale_factor_randA
        w_randB *= scale_factor_randB
        w_randC *= scale_factor_randC
        
        w_l = [w_randA, w_randB, w_randC]
        norm_l = [calc_norm2(w_randA), calc_norm2(w_randB), calc_norm2(w_randC)]

        w_min = w_l[np.argmin(norm_l)]
        ubpb.append(calc_norm2(w_min)/opt)
        
    print("MC 1000 upper bound power boost (dB) = ", np.min(ubpb), "mean = ", np.mean(ubpb), "std = ", np.std(ubpb))

    w_best = None
    ubpb_best = np.inf

    for i in range(30*N*M):
    # for i in range(10000):
        # recover randomization
        w_randA = recover_w_randA(W_opt)
        w_randB = recover_w_randB(W_opt)
        w_randC = recover_w_randC(W_opt)

        scale_factor_randA = get_min_scale_factor(w_randA, h_array_norm_QoS)
        scale_factor_randB = get_min_scale_factor(w_randB, h_array_norm_QoS)
        scale_factor_randC = get_min_scale_factor(w_randC, h_array_norm_QoS)

        w_randA *= scale_factor_randA
        w_randB *= scale_factor_randB
        w_randC *= scale_factor_randC
        
        w_l = [w_randA, w_randB, w_randC]
        norm_l = [calc_norm2(w_randA), calc_norm2(w_randB), calc_norm2(w_randC)]

        w_min = w_l[np.argmin(norm_l)]
        ubpb.append(calc_norm2(w_min)/opt)
        
        if ubpb[-1] < ubpb_best:
            w_best = w_min
            ubpb_best = ubpb[-1]
    w_best_qos = w_best
    print("MC 30*N*M=", 30*N*M, "upper bound power (dB) boost = ", np.min(ubpb), "mean = ", np.mean(ubpb), "std = ", np.std(ubpb))
    # END QOS OPTIMIZATION #