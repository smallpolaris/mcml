import time
import numpy as np
import numpy.random as nrd
# import matplotlib.pyplot as plt
import scipy.stats as ss
# import scipy
import tool
import update
# from scipy.optimize import fsolve, root
# from scipy import integrate
# from numpy import linalg as la
# from sympy import *
# from scipy.interpolate import BSpline


Rep = 1
Iter = 4000
LN = 500
N = 500
N = LN
NY = 3
NXI = 1
MT = 6      # maximum of time observations
NX = 2
NG = 5   # number of grids in survival model
all_lam = np.zeros([Rep, Iter, NY, NXI])
all_psi = np.zeros([Rep, Iter, NY])
all_lambd = np.zeros([Rep, Iter, NG])
all_alpha = np.zeros([Rep, Iter, NX + 2])
all_alpha1 = np.zeros([Rep, Iter, NX + 2])
all_alpha2 = np.zeros([Rep, Iter, NX + 2])
all_sigma = np.zeros([Rep, Iter, 2, 2])
all_gamma = np.zeros([Rep, Iter, NX + 2])
all_beta = np.zeros([Rep, Iter, NX + 2])
# all_gamma_beta = np.zeros([Rep, Iter, 2 * (NX + 2)])
all_b = np.zeros([Rep, N, 2])
# ##================save data======================##
all_ET = np.zeros([Rep, N])
all_CT = np.zeros([Rep, N])
all_OT = np.zeros([Rep, N])
all_X = np.zeros([Rep, N, NX+2])
all_I = np.zeros([Rep, N])
all_S = np.zeros([Rep, N])
all_W = np.zeros([Rep, N, MT])
all_y = np.zeros([Rep, N, MT, NY])
all_obsT = np.zeros([Rep, N, MT])
all_NT = np.zeros([Rep, N])
all_tb = np.zeros([Rep, N, 2])
for r in range(Rep):
    nrd.seed(66 + r)
    NT = nrd.randint(3, MT, LN)  # observation number for each person
    # NT = np.ones(LN) * MT # observation number for each person
    NT = NT.astype(int)
    # TT = np.zeros([LN, MT])
    inc_t = np.concatenate((np.zeros([LN, 1]), nrd.rand(LN, MT-1)), 1) # increment of t
    obs_T = np.cumsum(inc_t, 1)  # LN * MT
    for i in range(LN):
        obs_T[i, :NT[i]] = (obs_T[i, :NT[i]] - np.mean(obs_T[i, :NT[i]])) / np.std(obs_T[i, :NT[i]])
        obs_T[i, NT[i]:] = 0
    # rep_MT = np.repeat(np.arange(MT)[np.newaxis], LN, 0)
    # rep_NT = np.repeat(NT[:, np.newaxis], MT, 1)
    # obs = np.where(rep_MT < rep_NT)
    # obs_ind = (rep_MT < rep_NT)
    X = np.zeros([LN, NX + 2])
    X[:, 0] = 1
    X[:, 1] = nrd.binomial(1, 0.4, LN)   # treatment （Z）
    X[:, 2] = nrd.normal(0, 1, LN)
    X[:, 3] = nrd.uniform(-1, 1, LN)
    rx = np.zeros([LN, MT, NX + 2])
    for tt in range(MT):
        rx[:, tt] = X
    t_gamma = np.array([0.5, 0.5, 0.5, 0.5])
    t_beta = np.array([0.5, 0.5, 0.5, 0.5])
    t_sigma = np.array([[0.5, 0.3], [0.3, 0.5]])  # covariance matrix for intercept and slope
    # t_sigma = np.eye(2)
    t_b = ss.multivariate_normal.rvs(mean=np.zeros(2), cov=t_sigma, size=LN)
    t_Im = np.sum(t_gamma * X, 1)
    t_I = t_Im + t_b[:, 0]
    t_Sm = np.sum(t_beta * X, 1)
    t_S = t_Sm + t_b[:, 1]     # N
    t_W = t_I[:, np.newaxis] + t_S[:, np.newaxis] * obs_T  # N * NT
    ##================factor analysis model================##
    t_lam = np.array([1, 0.6, 0.6])
    # t_lam = np.array([1]) # 3
    y_m = t_lam * t_W[:, :, np.newaxis]  # N * NT * 3
    t_psi = np.array([0.36, 0.36, 0.36])
    y = np.zeros_like(y_m)  #  N * NT * 3
    for k in range(NY):
        y[:, :, k] = nrd.normal(y_m[:, :, k], np.sqrt(t_psi[k]))
    for i in range(N):
        y[i, NT[i]:] = 0
#     ##===============cox model==========================##
    t_alpha = np.array([1, 1, 1, 1])  # the last parameter is for the mediator
   #####================Another method to get solution of Cox model=================+########
    U = -np.log(nrd.uniform(0, 1, LN))
    fixed = 0.5 * np.exp(np.sum(t_alpha[:-1] * X[:, 1:], 1) + t_alpha[-1] * t_I) / (t_alpha[-1] * t_S)
    ET = np.log(1 + U / fixed) / (t_alpha[-1] * t_S)
    large_number = 100000
    loc1 = np.setdiff1d(np.arange(LN), np.where(ET < large_number)[0])
    print("survival process is over")   # sometimes error is large, how to deal with???
  ####================end of survival process========================##########
    c_max = 5
    CT = nrd.uniform(0, c_max, N)
    OT = np.minimum(CT, ET)
    OT[loc1] = CT[loc1]
    delta = (CT >= ET) + 0  # observed failure (dead) is 1
    censor_rate = 1 - np.sum(delta) / N
    print("the censor rate for rep %d is %3f" % (r, censor_rate))
    all_ET[r] = ET
    all_OT[r] = OT
    all_CT[r] = CT
    all_X[r] = X
    all_I[r] = t_I
    all_S[r] = t_S
    all_W[r] = t_W
    all_obsT[r] = obs_T
    all_NT[r] = NT
    all_y[r] = y
    all_tb[r] = t_b
np.save("ET.npy", all_ET)
np.save("OT.npy", all_OT)
np.save("CT.npy", all_CT)
np.save("X.npy", all_X)
# np.save("t_I.npy", all_I)
# np.save("t_S.npy", all_S)
# np.save("t_W.npy", all_W)
np.save("obs_T.npy", all_obsT)
np.save("NT.npy", all_NT)
np.save("y.npy", all_y)
np.save("t_b.npy", all_tb)