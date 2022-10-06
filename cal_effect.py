import time
import numpy as np
# import numpy.random as nrd
# import matplotlib.pyplot as plt
# import scipy.stats as ss
# import scipy
import tool
import update
import multiprocessing


Iter = 4000
burnin = 2000
N = 500
NG = 5
K = 50
M = 100
nt = 10
total = Iter - burnin
Rep = 1
sur_prob = np.zeros([Rep, 2, 2, total, N, nt])
tsur_prob = np.zeros([Rep, 2, 2, N, nt])
msur_prob = np.zeros([Rep, 2, 2, N, nt])
# E = np.zeros([Rep, 2, 2, nt])
t_alpha = np.array([1, 1, 1, 1])
t_gamma = np.array([0.5, 0.5, 0.5, 0.5])
t_beta = np.array([0.5, 0.5, 0.5, 0.5])
t_sigma = np.array([[0.5, 0.3], [0.3, 0.5]])  # covariance matrix for intercept and slope
def do(o):
    name = multiprocessing.current_process().name
    print("name : %s staring, n:%d"% (name, o),  flush=True, file=open("cal" + str(o) + ".txt", "a"))
    Rep_s = o * Rep
    Rep_e = (o + 1) * Rep
    for r in range(Rep_s, Rep_e):
        X = np.load("X.npy")[r]
        OT = np.load("OT.npy")[r]
        t_b = np.load("t_b.npy")[r]
        r1 = r - Rep_s
        all_alpha = np.load("alpha" + str(o) +".npy")[r1]
        all_beta = np.load("beta" + str(o) + ".npy")[r1]
        all_gamma = np.load("gamma" + str(o) + ".npy")[r1]
        all_lambd = np.load("lambd" + str(o) + ".npy")[r1]
        # all_sigma = np.load("sigma" + str(o) + ".npy")[r1]
        all_b = np.load("b" + str(o) + ".npy")[r1]
        grid = np.zeros(shape=[NG + 1])
        grid[-1] = np.max(OT) + 0.5  # t_G > OT all
        # grid[-1] = c_max + 2  # t_G > OT all
        t_lambd = 0.5 * np.ones(NG)
        for g in range(1, NG):
            grid[g] = np.percentile(OT, g / NG * 100)
        t = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
        tp = np.linspace(0, t, K+1).T  # nt * (K+1)
        td = tp[:, 1:] - tp[:, :-1]  # nt * K
        t_nu = np.zeros(shape=[t.shape[0], K, NG])  # nt * K * NG
        tool1 = tool.Tools(NG, K)
        for nt in range(t.shape[0]):
            interval_nu = np.zeros(shape=[K, NG])
            interval = tool1.find_obs_num(tp[nt, 1:], grid)
            t_nu[nt, np.arange(K), interval - 1] = 1
        data = update.MCMC(N, NG, NG, NG, NG)
    ####============== using mean of parameters to obtain results===============##########
        e_alpha = np.mean(all_alpha[burnin:], 0)
        e_beta = np.mean(all_beta[burnin:], 0)
        e_gamma = np.mean(all_gamma[burnin:], 0)
        # e_sigma = np.mean(all_sigma[burnin:], 0)
        e_lambd = np.mean(all_lambd[burnin:], 0)
        e_b = np.mean(all_b[burnin:], 0)
        # r1 = r - Rep_s
        for z in range(2):
            for z_s in range(2):
                tsur_prob[r1, z, z_s] = data.surv_prob1(tp[:, 1:], td, t_nu, z, z_s, t_alpha, t_beta, t_gamma, t_b, X, t_lambd)
                msur_prob[r1, z, z_s] = data.surv_prob1(tp[:, 1:], td, t_nu, z, z_s, e_alpha, e_beta, e_gamma, e_b, X, e_lambd)
        for iter in range(burnin, Iter):
            alpha = all_alpha[iter]
            beta = all_beta[iter]
            gamma = all_gamma[iter]
            lambd = all_lambd[iter]
            # sigma = all_sigma[iter]
            b = all_b[iter]
            temp = iter - burnin
            t0 = time.time()
            for z in range(2):
                for z_s in range(2):
                    sur_prob[r1, z, z_s, temp] = data.surv_prob1(tp[:,1:], td, t_nu, z, z_s, alpha, beta, gamma, b, X, lambd)
            if iter % 100 == 0 and iter > 99:
                process = (r1 * total + temp) / (Rep * total)
                one_iter_time = time.time() - t0
                # print("%.3f seconds process time for one iter" % one_iter_time)
                print("%.3f seconds process time for one iter" % one_iter_time, flush=True, file=open("cal" + str(o) + ".txt", "a"))
                rtime = Rep * Iter * one_iter_time * (1 - process) / 60
                # print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime))
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r1, iter, process, rtime),
                      flush=True, file=open("cal" + str(o) + ".txt", "a"))
        print("One replication done")
    E = np.mean(sur_prob, 4)  # Rep * 2 * 2 * Iter * nt
    # ete = E[:, 1, 1] - E[:, 0, 0]  # Rep * Iter * nt
    # ede = E[:, 1, 0] - E[:, 0, 0]  # Rep * Iter * nt
    # eie = E[:, 1, 1] - E[:, 1, 0]  # Rep * Iter * nt
    tE = np.mean(tsur_prob, 3)     # Rep * 2 * 2 * nt
    # tte = tE[:, 1, 1] - tE[:, 0, 0]  # Rep * nt
    # tde = tE[:, 1, 0] - tE[:, 0, 0]
    # tie = tE[:, 1, 1] - tE[:, 1, 0]
    mE = np.mean(msur_prob, 3)
    # mte = mE[:, 1, 1] - mE[:, 0, 0]  # Rep * nt
    # mde = mE[:, 1, 0] - mE[:, 0, 0]
    # mie = mE[:, 1, 1] - mE[:, 1, 0]
    np.save("E" + str(o) + ".npy", E)
    np.save("mE" + str(o) + ".npy", mE)
    np.save("tE" + str(o) + ".npy", tE)

if __name__ == '__main__':
    numList = []
    for o in range(0, 1):
        p = multiprocessing.Process(target=do, args=(o,))
        numList.append(p)
        p.start()
        p.join()
