import time
import numpy as np
import numpy.random as nrd
# import matplotlib.pyplot as plt
import scipy.stats as ss
# import scipy
import tool
import update
import multiprocessing
# from scipy.optimize import fsolve, root
# from scipy import integrate
# from numpy import linalg as la
# from sympy import *
# from scipy.interpolate import BSpline


Rep = 1
once = 1
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
# all_alpha1 = np.zeros([Rep, Iter, NX + 2])
# all_alpha2 = np.zeros([Rep, Iter, NX + 2])
all_sigma = np.zeros([Rep, Iter, 2, 2])
all_gamma = np.zeros([Rep, Iter, NX + 2])
all_beta = np.zeros([Rep, Iter, NX + 2])
all_b = np.zeros([Rep, Iter, N, 2])
# ##================save data======================##
# all_ET = np.zeros([Rep, N])
# all_CT = np.zeros([Rep, N])
# all_OT = np.zeros([Rep, N])
# all_X = np.zeros([Rep, N, NX+2])
# all_I = np.zeros([Rep, N])
# all_S = np.zeros([Rep, N])
# all_W = np.zeros([Rep, N, MT])
# all_y = np.zeros([Rep, N, MT, NY])
# all_obsT = np.zeros([Rep, N, MT])
# all_NT = np.zeros([Rep, N])
# all_tb = np.zeros([Rep, Iter, N, 2])

# ######-=========================Load data=========================================###
def do(o):
    Rep_s = o * once
    Rep_e = (o+1) * once
    for r in range(Rep_s, Rep_e):
        r1 = r - Rep_s
        nrd.seed(66 + r)
        ET = np.load("ET.npy")[r]
        OT = np.load("OT.npy")[r]
        CT = np.load("CT.npy")[r]
        delta = (CT >= ET) + 0
        X = np.load("X.npy")[r]
        rx = np.zeros([LN, MT, NX + 2])
        for tt in range(MT):
            rx[:, tt] = X
        # t_I = np.load("t_I.npy")[r]
        # t_S = np.load("t_S.npy")[r]
        # t_W = np.load("t_W.npy")[r]
        obs_T = np.load("obs_T.npy")[r]
        NT = np.load("NT.npy")[r].astype(int)
        y = np.load("y.npy")[r]
        #######=======================Initialization of parameters=========================#########
        rep_MT = np.repeat(np.arange(MT)[np.newaxis], N, 0)
        rep_NT = np.repeat(NT[:, np.newaxis], MT, 1)
        obs = np.where(rep_MT < rep_NT)
        obs_ind = (rep_MT < rep_NT)
        tool1 = tool.Tools(N, obs_T)
        lam = nrd.randn(NY, NXI)
        lam[0] = 1
        Ind = np.ones(shape=[NY, NXI])  # to show lambda is fixed or not fix is zero !!!! no intercept
        Ind[0] = 0  # keep the same dimension as the lam
        psi = nrd.rand(NY)
        # for i in range(N):
        #     y[i, NT[i]:] = 0
        #     # W[i, NT[i]:] = 0
        grid = np.zeros(shape=[NG + 1])
        grid[-1] = np.max(OT) + 0.5  # t_G > OT all
        # grid[-1] = c_max + 2  # t_G > OT all
        for g in range(1, NG):
            grid[g] = np.percentile(OT, g / NG * 100)
        ####============set delta  observed failure (dead) is 1
        nu = np.zeros(shape=[N, NG])  # indicate fail in which interval
        interval_nu = np.zeros(shape=[N, NG])
        interval = tool1.find_obs_num(OT, grid)
        nu[np.arange(N), interval - 1] = 1
        for i in range(N):
            interval_nu[i, :(interval[i] - 1)] = 1  # fill nu: before event is all 1 (
        # survival model
        alpha = nrd.randn(NX + 2)  #
        accept_alpha = 0
        lambd = nrd.uniform(0, 5, NG)
        c_alpha = 0.01
        gamma = nrd.rand(NX + 2)
        # c_gamma = 0.001
        accept_gamma = 0
        beta = nrd.rand(NX + 2)
        c_beta = 2
        accept_beta = 0
        rand_m = nrd.rand(2, 2)
        sigma = np.dot(rand_m, rand_m.T)
        # sigma = t_sigma
        b = ss.multivariate_normal.rvs(mean=np.zeros([2]), cov=sigma, size=N)
        # b[:, 1] = t_b[:, 1]
        c_b0 = 0.3
        c_b1 = 0.3
        accept_b0 = np.zeros(N)
        accept_b1 = np.zeros(N)
        #############============================  MCMC to update parameters =================#####
        data = update.MCMC(N, NG, obs, obs_ind, obs_T)
        for iter in range(Iter):
            t0 = time.time()
            gamma_beta, accept_beta = data.update_gamma_beta(gamma, beta, X, rx, b, y, lam, psi, OT, alpha, nu,
                                                             interval_nu, delta, lambd, grid, c_beta, accept_beta)
            gamma = gamma_beta[:NX + 2]
            beta = gamma_beta[NX + 2:]
            all_gamma[r1, iter] = gamma
            all_beta[r1, iter] = beta
            W = tool1.update_w(gamma, beta, X, b)
            lam, psi = data.update_lam_psi(psi, lam, Ind, W, y)
            all_lam[r1, iter] = lam
            all_psi[r1, iter] = psi
            lambd = data.update_lambd(lambd, nu, delta, alpha, X, gamma, beta, b, OT, grid)
            all_lambd[r1, iter] = lambd
            alpha, accept_alpha = data.update_alpha(alpha, X, gamma, beta, b, OT, nu, interval_nu, delta, lambd, grid,
                                                    c_alpha, accept_alpha)
            all_alpha[r1, iter] = alpha
            b, accept_b0 = data.update_b0(b, gamma, beta, X, y, lam, psi, OT, alpha, nu, interval_nu, delta,
                                          lambd, grid, sigma, c_b0, accept_b0)
            b, accept_b1 = data.update_b1(b, gamma, beta, X, y, lam, psi, OT, alpha, nu, interval_nu, delta,
                                          lambd, grid, sigma, c_b1, accept_b1)
            all_b[r1, iter] = b
            sigma = data.update_sigma(b)
            all_sigma[r1, iter] = sigma
            if iter > 99 and iter % 100 == 0:
                if iter % 1000 == 0:
                    if accept_beta / iter < 0.1:
                        c_beta = c_beta / 2
                    if accept_alpha / iter < 0.1:
                        c_alpha = c_alpha / 2
                process = (r * Iter + iter) / (Rep * Iter)
                one_iter_time = time.time() - t0
                print("%.3f seconds process time for one iter" % one_iter_time)
                print("%.3f seconds process time for one iter" % one_iter_time, flush=True, file=open("medation"+str(o)+".txt", "a"))
                rtime = Rep * Iter * one_iter_time * (1 - process) / 60
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime))
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r1, iter, process, rtime), flush=True,
                      file=open("medation"+str(o)+".txt", "a"))
        print("Rep: %d, beta :%d, gamma: %d, alpha: %d"%(r1, accept_beta, accept_gamma, accept_alpha), flush=True,
              file=open("medation"+str(o)+".txt", "a"))
        print("test is over")
    print("End of all replications")
    np.save("alpha" + str(o) + ".npy", all_alpha)
    np.save("beta" + str(o) + ".npy", all_beta)
    np.save("gamma" + str(o) + ".npy", all_gamma)
    np.save("sigma" + str(o) + ".npy", all_sigma)
    np.save("lambd" + str(o) + ".npy", all_lambd)
    np.save("lam" + str(o) + ".npy", all_lam)
    np.save("psi" + str(o) + ".npy", all_psi)
    np.save("b" + str(o) + ".npy", all_b)

if __name__ == '__main__':
    numList = []
    for o in range(0, 1):
        p = multiprocessing.Process(target=do, args=(o,))
        numList.append(p)
        p.start()
        p.join()

