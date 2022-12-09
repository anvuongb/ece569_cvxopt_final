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

distance_list_list = []
capacity_list_list = []

for N in tqdm.tqdm([2,4,8,16]):
    distance_list = []
    capacity_list = []

    for d in tqdm.tqdm(range(100)):
        # Eve moves closer to Bob by 100m
        d_Bob_Eve_new = d_Bob_Eve_init-d*100

        if d_Bob_Eve_new < 100:
            break

        distance_list.append(d_Bob_Eve_new)

        distance_Eve_new = np.sqrt(d_Bob_Eve_new**2 + distance_Bob**2 - 2*d_Bob_Eve_new*distance_Bob*np.cos(a_Bob_Eve_init))
        angle_Eve_new = angle_Bob - np.arccos(-(d_Bob_Eve_new**2 - distance_Eve_new**2 - distance_Bob**2)/(2*distance_Bob*distance_Eve_new))

        h_Eve, h_Bob, H_Eve, H_Bob = generate_channels_secrecy(angle_Eve_new, angle_Bob, distance_Eve_new, distance_Bob, sigma_Eve, sigma_Bob, lambda_carrier, N, los=True)

        opt_vals = []
        opt_Ws = []
        for s in np.arange(1, 2, 1e-2):
            try:
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

        w_best = None
        max_capacity = -np.inf
        for i in range(30*N*2):
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
            capacity = [(np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randA))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randA))**2)).ravel()[0],
                        (np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randB))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randB))**2)).ravel()[0],
                        (np.log(1 + np.abs(np.dot(np.conjugate(h_Bob).T, w_randC))**2) - np.log(1 + np.abs(np.dot(np.conjugate(h_Eve).T, w_randC))**2)).ravel()[0]
                       ]

            w_best_tmp = w_l[np.argmax(capacity)]
            max_capacity_tmp = np.max(capacity)

            if max_capacity_tmp > max_capacity:
                w_best = w_best_tmp
                max_capacity = max_capacity_tmp

        capacity_list.append(max_capacity)
    
    distance_list_list.append(distance_list)
    capacity_list_list.append(capacity_list)

fig = plt.figure(figsize=(7,5))
for i, N in tqdm.tqdm(zip(range(5), [4,8,16,32,64])):
    plt.plot(distance_list_list[i], capacity_list_list[i], marker="x", markersize=5, label="{} antennas".format(N))
plt.title("secrecy capacity versus distance between Bob and Eve")
plt.ylabel("secrecy capacity (bps/Hz)")
plt.xlabel("distance between Bob and Eve (meters)")
# plt.xlim(1,2)
plt.show()
fig.savefig("fig2c.png")