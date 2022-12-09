import numpy as np

# contains helper functions for calculations

# set random seed for debugging purpose, need to turn off when perform monte carlo
# np.random.seed(42)

# function for plots
def gain(w):
    """Return the power as a function of azimuthal angle, phi."""
    phi = np.linspace(0, 2*np.pi, 1000)
    psi = 2*np.pi * 1 / 2 * np.cos(phi)
    j = np.arange(len(w))
    A = np.sum(w[j] * np.exp(j * 1j * psi[:, None]), axis=1)
    g = np.abs(A)**2
    return phi, g

# function to generate channels
def get_directive_gain(g, minDdBi=-20):
    """Return the "directive gain" of the antenna array producing gain g."""
    DdBi = 10 * np.log10(g / np.max(g))
    return np.clip(DdBi, minDdBi, None)

def rand_angles(M):
    return np.random.randint(0, 180, M)/180*np.pi # 0->180deg

def rand_distances(M):
    return np.random.randint(1000, 2500, M) # 50-250m

def generate_channels_LOS(angles, distances, lambda_, N, M, rho_min_qos_sigma2=1, sigma_i=1):
    # generate line-of-sight channels
    r = np.arange(N)
    h_array_LOS = [np.exp(-2j*np.pi*distances[i]/lambda_)*np.exp(-1j*2*(np.pi*r/2)*np.cos(angles[i])).reshape((N, 1)) for i in range(M)]
    h_array_LOS_norm_QoS = []
    h_array_LOS_norm_MaxMin = []
    
    for i in range(M):
        # normalization QoS
        h_i_norm_QoS = h_array_LOS[i]/np.sqrt(rho_min_qos_sigma2)
        h_array_LOS_norm_QoS.append(h_i_norm_QoS)

        # normalization MaxMin
        h_i_norm_MaxMin = h_array_LOS[i]/sigma_i
        h_array_LOS_norm_MaxMin.append(h_i_norm_MaxMin)
    
    # generate H_i matrix
    H_array_LOS = []
    H_array_LOS_norm_QoS = []
    H_array_LOS_norm_MaxMin = []
    for i in range(M):
        H_array_LOS.append(np.matmul(h_array_LOS[i], np.conjugate(h_array_LOS[i]).T))
        H_array_LOS_norm_QoS.append(np.matmul(h_array_LOS_norm_QoS[i], np.conjugate(h_array_LOS_norm_QoS[i]).T))
        H_array_LOS_norm_MaxMin.append(np.matmul(h_array_LOS_norm_MaxMin[i], np.conjugate(h_array_LOS_norm_MaxMin[i]).T))
        
    return h_array_LOS, h_array_LOS_norm_QoS, h_array_LOS_norm_MaxMin, H_array_LOS, H_array_LOS_norm_QoS, H_array_LOS_norm_MaxMin
        
def generate_channels(angles, distances, lambda_, N, M, rho_min_qos_sigma2=1, sigma_i=1):
    # generate multipath channel
    h_array = []
    h_array_norm_QoS = []
    h_array_norm_MaxMin = []
    for i in range(M):
        # complex normal with zero-mean, unit-variance
        # h_i = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N, 2)).view(np.complex128)
        # h_i = np.exp(-2j*np.pi*distances[i]/lambda_)*(np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)))
        h_i = np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain
        # h_i = (np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain and no scale var
        h_array.append(h_i)

        # normalization QoS
        h_i_norm_QoS = h_i/np.sqrt(rho_min_qos_sigma2)
        h_array_norm_QoS.append(h_i_norm_QoS)

        # normalization MaxMin
        h_i_norm_MaxMin = h_i/sigma_i
        h_array_norm_MaxMin.append(h_i_norm_MaxMin)

    # generate H_i matrix
    H_array = []
    H_array_norm_QoS = []
    H_array_norm_MaxMin = []
    for i in range(M):
        H_array.append(np.matmul(h_array[i], np.conjugate(h_array[i]).T))
        H_array_norm_QoS.append(np.matmul(h_array_norm_QoS[i], np.conjugate(h_array_norm_QoS[i]).T))
        H_array_norm_MaxMin.append(np.matmul(h_array_norm_MaxMin[i], np.conjugate(h_array_norm_MaxMin[i]).T))
        
    return h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin

def generate_channels_comparison_MaxMin_QoS(angles, distances, lambda_, N, M, rho_min_qos_sigma2=1, sigma_i=1):
    # ggenerate multipath channel used in fig 1.c
    h_array = []
    h_array_norm_QoS = []
    h_array_norm_MaxMin = []
    for i in range(M):
        # complex normal with zero-mean, unit-variance
        # h_i = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N, 2)).view(np.complex128)
        # h_i = np.exp(-2j*np.pi*distances[i]/lambda_)*(np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)))
        # h_i = np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain
        # h_i = (np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain and no scale var
        h_i = (16*np.random.randn(N)+4 + 1j*(16*np.random.randn(N)+4)).reshape((N, 1)) # follows exp from paper h = randn(4,16) + j*randn(4,16)
        h_array.append(h_i)

        # normalization QoS
        h_i_norm_QoS = h_i/np.sqrt(rho_min_qos_sigma2)
        h_array_norm_QoS.append(h_i_norm_QoS)

        # normalization MaxMin
        h_i_norm_MaxMin = h_i/sigma_i
        h_array_norm_MaxMin.append(h_i_norm_MaxMin)

    # generate H_i matrix
    H_array = []
    H_array_norm_QoS = []
    H_array_norm_MaxMin = []
    for i in range(M):
        H_array.append(np.matmul(h_array[i], np.conjugate(h_array[i]).T))
        H_array_norm_QoS.append(np.matmul(h_array_norm_QoS[i], np.conjugate(h_array_norm_QoS[i]).T))
        H_array_norm_MaxMin.append(np.matmul(h_array_norm_MaxMin[i], np.conjugate(h_array_norm_MaxMin[i]).T))
        
    return h_array, h_array_norm_QoS, h_array_norm_MaxMin, H_array, H_array_norm_QoS, H_array_norm_MaxMin
    
def generate_channels_secrecy(angle_Eve, angle_Bob, distance_Eve, distance_Bob, sigma_Eve, sigma_Bob, lambda_carrier, N, los=True):
    # generate channels for serecy capacity simulation
    # generate LoS channel
    r = np.arange(N)
    if los:
        h_Eve = np.exp(-2j*np.pi*distance_Eve/lambda_carrier)*np.exp(-1j*2*(np.pi*r/2)*np.cos(angle_Eve)).reshape((N, 1))
        h_Eve = h_Eve/sigma_Eve
        H_Eve = np.matmul(h_Eve, np.conjugate(h_Eve).T)
        
        h_Bob = np.exp(-2j*np.pi*distance_Bob/lambda_carrier)*np.exp(-1j*2*(np.pi*r/2)*np.cos(angle_Bob)).reshape((N, 1))
        h_Bob = h_Bob/sigma_Bob
        H_Bob = np.matmul(h_Bob, np.conjugate(h_Bob).T)
        
        return h_Eve, h_Bob, H_Eve, H_Bob
    
    # generate rayleigh channel
    h_Eve = np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain
    h_Eve = h_Eve/sigma_Eve
    H_Eve = np.matmul(h_Eve, np.conjugate(h_Eve).T)

    h_Bob = np.sqrt(2)/2*(np.random.randn(N) + 1j*np.random.randn(N)).reshape((N, 1)) # no baseband gain
    h_Bob = h_Bob/sigma_Bob
    H_Bob = np.matmul(h_Bob, np.conjugate(h_Bob).T)
    return h_Eve, h_Bob, H_Eve, H_Bob

# Randomization techniques to recover w
def recover_w_SVD(W_opt):
    v, d = np.linalg.eig(W_opt)
    w_opt = d[:,0]
    return w_opt.reshape((W_opt.shape[0], 1))

def recover_w_randA(W_opt):
    v, d = np.linalg.eig(W_opt)
    # rand e
    e_js = np.random.uniform(0, 2*np.pi, W_opt.shape[0]).reshape((W_opt.shape[0], 1))
    e = np.cos(e_js) + 1j*np.sin(e_js)
    
    Sigma_sqrt = np.sqrt(np.diag(v))
    w_opt = np.matmul(np.matmul(d, Sigma_sqrt), e)
    return w_opt.reshape((W_opt.shape[0], 1))

def recover_w_randB(W_opt):
    e_js = np.random.uniform(0, 2*np.pi, W_opt.shape[0]).reshape((W_opt.shape[0], 1))
    e = np.cos(e_js) + 1j*np.sin(e_js)
    w_opt = np.matmul(np.sqrt(np.diag(np.diag(W_opt))), e)
    return w_opt.reshape((W_opt.shape[0], 1))
    
def recover_w_randC(W_opt):
    v, d = np.linalg.eig(W_opt)
    # rand e
    e = np.sqrt(2)/2*(np.random.randn(W_opt.shape[0]) + 1j*np.random.randn(W_opt.shape[0])).reshape((W_opt.shape[0], 1))
    
    Sigma_sqrt = np.sqrt(np.diag(v))
    w_opt = np.matmul(np.matmul(d, Sigma_sqrt), e)
    return w_opt.reshape((W_opt.shape[0], 1))

# some other helpers
def check_violation(w, h):
    scale_factor = 1
    constraint = np.abs(np.dot(np.conjugate(h).T, w))[0][0]**2
    # print(constraint)
    violate = constraint < 1
    if violate:
        return np.sqrt(1.0/constraint), constraint
    return scale_factor, constraint

def get_min_scale_factor(w, h_array):
    s = np.min(np.abs(np.matmul(np.conjugate(w).T, h_array)).ravel())
    if s < 1:
        return 1/s
    return 1

def calc_norm2(w):
    # return np.sqrt(np.abs(np.dot(np.conjugate(w).T, w))[0][0])
    return np.linalg.norm(w)

def get_min_snr(w, h_array):
    min_snr = np.inf
    for h in h_array:
        snr = np.abs(np.dot(np.conjugate(h).T, w))**2
        # print(snr)
        min_snr = np.min([min_snr, snr])
    return min_snr
