import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *

## set params
# QoS
rho_min_qos_sigma2 = 1 #rho_min_i * sigma_i^2 = 1 for all i
# Max-Min
sigma_i = 1 
# power
P = 1

N=4
M=16

# perform calculations
print("Solving N/M = {}/{}".format(N, M))

best_t = []
best_t_maxmin = []
best_t_qos = []

# Monte Carlo 1000 runs
for k in tqdm.tqdm(range(300)):
    # regenerate channel, no path loss
    h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin = generate_channels_comparison_MaxMin_QoS(None, None, lambda_, N, M, rho_min_qos_sigma2=rho_min_qos_sigma2, sigma_i=sigma_i)

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
    best_t_maxmin.append(max_minSNR**2)
    ### END MAXMIN SDR ###

    ### QoS SDR ###
    W = cp.Variable((N, N), hermitian=True)

    # need to use cp.real since cp.trace will give imaginary part = 0j, which will break the code
    constraints = [W >> 0]
    constraints += [cp.real(cp.trace(cp.matmul(W, H_array_norm_QoS[i]))) >= 1 for i in range(M)]

    prob = cp.Problem(cp.Minimize( cp.real(cp.trace(W))),
                 constraints)
    opt = prob.solve(solver="MOSEK", verbose=False)
    W_opt = W.value

    w_best = None
    ubpb_best = np.inf
    ubpb = []
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
        norm_l = [np.linalg.norm(w_randA)**2, np.linalg.norm(w_randB)**2, np.linalg.norm(w_randC)**2]

        w_min = w_l[np.argmin(norm_l)]
        ubpb.append(np.linalg.norm(w_min)**2/opt)

        if ubpb[-1] < ubpb_best:
            w_best = w_min
            ubpb_best = ubpb[-1]
    w_best_qos = w_best/np.linalg.norm(w_best)/np.sqrt(P) # scale to norm P
    max_minSNR_tmp = np.min(abs(np.matmul(np.conjugate(w_best_qos).T, h_array_norm_QoS)).ravel())
    best_t_qos.append(max_minSNR_tmp**2)
    ### END QoS SDR ###


print("upper bound =", np.mean(best_t))
print("maxmin =", np.mean(best_t_maxmin))
print("qos =", np.mean(best_t_qos))
    
fig = plt.figure(figsize=(10,10))
plt.plot(100*((np.array(best_t) - np.array(best_t_maxmin))/np.array(best_t)), linestyle="dashed", label="percent MaxMin SDR")
plt.plot(100*((np.array(best_t) - np.array(best_t_qos))/np.array(best_t)), linestyle="dotted", label="percent QoS SDR")
# plt.hline(np.mean(100*(np.array(best_t)-np.array(best_t_qos)), label="mean MaxMin SDR"))
plt.hlines(100*np.mean((np.array(best_t) - np.array(best_t_maxmin))/np.array(best_t)), 0, 300, color="black", linestyle="dashed", label="mean percent MaxMin SDR")
plt.hlines(100*np.mean((np.array(best_t) - np.array(best_t_qos))/np.array(best_t)), 0, 300, color="r", linestyle="dotted", label="mean percent QoS SDR")
plt.xlabel("# run")
plt.ylabel("percent gap (gap/relaxation bound) %")
plt.title("Gap over Relaxation bound h=randn(4,16)+j*randn(4,16)")
plt.xlim(200,250)
plt.legend()
plt.tight_layout()
# plt.show()
fig.savefig("fig1c.png")