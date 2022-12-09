import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *

## Set params
N = 8 # number of transmit antennas
M = 16 # number of mobile users
# P = 1 # transmit power
lambda_ = 0.375 # carrier wavelength
antenna_sep = 1/2 # antenanna separation half wavelength
# QoS
rho_min_qos_sigma2 = 1 #rho_min_i * sigma_i^2 = 1 for all i
# Max-Min
sigma_i = 1 

angles = rand_angles(M)
distances = rand_distances(M)

# Generate LoS channels
h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin = generate_channels(angles, distances, lambda_, N, M, rho_min_qos_sigma2=rho_min_qos_sigma2, sigma_i=sigma_i)

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

# recover svd
w_SVD   = recover_w_SVD(W_opt)
scale_factor_SVD = get_min_scale_factor(w_SVD, H_array_norm_QoS)
w_SVD *= scale_factor_SVD
ubpb_SVD = np.linalg.norm(w_SVD)**2/opt
print("SVD upper bound power boost (dB) = ", ubpb_SVD)
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
    
    w_l = [w_SVD, w_randA, w_randB, w_randC]
    norm_l = [calc_norm2(w_SVD), calc_norm2(w_randA), calc_norm2(w_randB), calc_norm2(w_randC)]

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
    
    w_l = [w_SVD, w_randA, w_randB, w_randC]
    norm_l = [calc_norm2(w_SVD), calc_norm2(w_randA), calc_norm2(w_randB), calc_norm2(w_randC)]

    w_min = w_l[np.argmin(norm_l)]
    ubpb.append(calc_norm2(w_min)/opt)
    
    if ubpb[-1] < ubpb_best:
        w_best = w_min
        ubpb_best = ubpb[-1]
w_best_qos = w_best
print("MC 30*N*M=", 30*N*M, "upper bound power (dB) boost = ", np.min(ubpb), "mean = ", np.mean(ubpb), "std = ", np.std(ubpb))
# END QOS OPTIMIZATION #

# PERFORM MAXMIN OPTIMIZATION #
W = cp.Variable((N, N), hermitian=True)
t = cp.Variable(1)

constraints = [W >> 0]
constraints += [cp.real(cp.trace(W)) == 1] # power constraint
constraints += [cp.real(cp.trace(cp.matmul(W, H_array_norm_MaxMin[i]))) >= t for i in range(M)]

prob = cp.Problem(cp.Maximize(t),
             constraints)
opt = prob.solve(solver="MOSEK", verbose=False)
W_opt = W.value

print("optimal objective value = ", opt)
# print("optimal beamforming vector = ", W_opt)

w_best_maxmin = None
max_minSNR = -np.inf
for i in tqdm.tqdm(range(30*N*M)):
# for i in tqdm.tqdm(range(5)):
    # recover randomization
    w_randA = recover_w_randA(W_opt)
    w_randB = recover_w_randB(W_opt)
    w_randC = recover_w_randC(W_opt)

    # scale to norm P
    w_randA /= np.linalg.norm(w_randA)
    w_randB /= np.linalg.norm(w_randB)
    w_randC /= np.linalg.norm(w_randC)

    w_l = [w_randA, w_randB, w_randC]
    minSNR = [np.min(abs(np.matmul(np.conjugate(w_randA).T, h_array_norm_MaxMin)).ravel()), 
              np.min(abs(np.matmul(np.conjugate(w_randB).T, h_array_norm_MaxMin)).ravel()), 
              np.min(abs(np.matmul(np.conjugate(w_randC).T, h_array_norm_MaxMin)).ravel())]

    w_best_tmp = w_l[np.argmax(minSNR)]
    max_minSNR_tmp = np.max(minSNR)
    
    if max_minSNR_tmp > max_minSNR:
        w_best_maxmin = w_best_tmp
        max_minSNR = max_minSNR_tmp
    
print("min SNR = ", t.value[0])
print("max_minSNR = ", max_minSNR**2)
# END MAXMIN OPTIMIZATION #

# PLOT FIG 1.A #
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='polar', label="meters")

# max min
w = w_best_maxmin.ravel()
phi, g = gain(w)
DdBi = get_directive_gain(g)
ax.plot(phi, g*np.max(distances)/np.max(g), label="MaxMin beam pattern")

# QoS
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='polar', label="meters")

# max min
w = w_best_maxmin.ravel()
phi, g = gain(w)
DdBi = get_directive_gain(g)
ax.plot(phi, g*np.max(distances)/np.max(g), label="MaxMin beam pattern")

# QoS
w = w_best_qos.ravel()
phi, g = gain(w)
DdBi = get_directive_gain(g)
ax.plot(phi, g*np.max(distances)/np.max(g), label="QoS beam pattern")

ax.scatter(angles, distances, color="red", label="mobile user")
# ax.set_rticks([0, 50, 100, 200, 250])
ax.set_rlabel_position(45)
plt.title("Beamforming pattern Rayleigh channels (gain is normalized)")
plt.legend()
ax.legend(loc=(0.8,0.92))
plt.tight_layout()
plt.show()
fig.savefig("fig1b.png")
