import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *

## set params
N = 8 # number of transmit antennas
M = 2 # number of mobile users
# P = 1 # transmit power

lambda_carrier = 0.375 # carrier wavelength
antenna_sep = 1/2 # antenanna separation half wavelength

P = 1

angle_Eve = 0.52359878
angle_Bob = 3.07177948

distance_Eve = 2057
distance_Bob = 1202

angles = [angle_Eve, angle_Bob]
distances = [distance_Eve, distance_Bob]

sigma_Eve = 1
sigma_Bob = 1

d_Bob_Eve_init = np.sqrt(distance_Eve**2 + distance_Bob**2 -2*distance_Eve*distance_Bob*np.cos(angle_Bob-angle_Eve))
a_Bob_Eve_init = np.arccos(-(distance_Eve**2 - distance_Bob**2 - d_Bob_Eve_init**2)/(2*distance_Bob*d_Bob_Eve_init))

## perform calculations
capacity_list = []

for N in [2,4,8,16]:
    print("Solving N=",N)
    max_cap_randA = []
    max_cap_randB = []
    max_cap_randC = []
    max_cap = []
    for i in tqdm.tqdm(range(300)): # 300 MC runs
        h_Eve, h_Bob, H_Eve, H_Bob = generate_channels_secrecy(angle_Eve, angle_Bob, distance_Eve, distance_Bob, sigma_Eve, sigma_Bob, lambda_carrier, N, los=True)

        opt_vals = []
        opt_Ws = []
        for s in np.arange(1, 2, 1e-2):
            try:
                # optimization problem for secrecy
                W = cp.Variable((N, N), hermitian=True)

                # need to use cp.real since cp.trace will give imaginary part = 0j, which will break the code
                constraints = [W >> 0]
                constraints += [cp.real(cp.trace(W)) == P]
                constraints += [1 + cp.real(cp.trace(W @ H_Eve)) == s]

                prob = cp.Problem(cp.Maximize( cp.log(1 + cp.real(cp.trace(W @ H_Bob))) - cp.log(s)),
                                constraints)
                _ = prob.solve(solver="MOSEK", verbose=False)
                opt_vals.append(prob.value)
                opt_Ws.append(W.value)
            except Exception as e:
                print(e)

        W_opt = opt_Ws[np.argmax(opt_vals)]
        opt = np.max(opt_vals)
        max_cap.append(opt)

        max_capacity_A = -np.inf
        max_capacity_B = -np.inf
        max_capacity_C = -np.inf
        for i in range(30*N*2):
            # recover randomization
            w_randA = recover_w_randA(W_opt)
            w_randB = recover_w_randB(W_opt)
            w_randC = recover_w_randC(W_opt)

            # scale to norm P
            w_randA /= np.linalg.norm(w_randA)
            w_randB /= np.linalg.norm(w_randB)
            w_randC /= np.linalg.norm(w_randC)

            capacityA, capacityB, capacityC = [(np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randA))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randA))**2)).ravel()[0],
                                                (np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randB))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randB))**2)).ravel()[0],
                                                (np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randC))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randC))**2)).ravel()[0]]

            max_capacity_A = np.max([max_capacity_A, capacityA])
            max_capacity_B = np.max([max_capacity_B, capacityB])
            max_capacity_C = np.max([max_capacity_C, capacityC])

        max_cap_randA.append(max_capacity_A)
        max_cap_randB.append(max_capacity_B)
        max_cap_randC.append(max_capacity_C)
    
    print("{:.4}f & {:.4}f & {:.4}f & {:.4}f".format(np.mean(max_cap), np.mean(max_cap_randA), np.mean(max_cap_randB), np.mean(max_cap_randC)))

