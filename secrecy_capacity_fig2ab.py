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

w_best = None
max_capacity = -np.inf
for i in range(5000):
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
print("max secrecy capacity = ", max_capacity)

print("plotting beam pattern")
w = w_best.ravel()
# Calculate gain and directive gain; plot on a polar chart.
phi, g = gain(w)
DdBi = get_directive_gain(g)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='polar', label="meters")
ax.plot(phi, g*np.max(distances)/np.max(g), label="beam pattern")
# ax.plot(phi, np.log(g)*np.max(distances)/np.log(np.max(g)), label="beam pattern")
ax.scatter(angle_Eve, distance_Eve, color="red", label="Eve")
ax.scatter(angle_Bob, distance_Bob, color="blue", label="Bob")
# ax.set_rticks([0, 50, 100, 200, 250])
ax.set_rlabel_position(45)
plt.title("Beamforming pattern (gain is normalized)\n best capacity = {:.4f}".format(max_capacity))
plt.legend()
ax.legend(loc=(0.8,0.92))
plt.tight_layout()
# plt.show()
fig.savefig("fig2a.png")
    
print("plotting C_s vs s")
fig = plt.figure(figsize=(7,5))

plt.plot(np.arange(1, 2, 1e-2), opt_vals)
plt.title("capacity versus s")
plt.ylabel("secrecy capacity (bps/Hz)")
plt.xlabel("s")
plt.xlim(1,2)
plt.show()
fig.savefig("fig2b.png")