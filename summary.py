import time
import numpy as np
import numpy.random as nrd
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy
import tool
import update



Rep = 1
Iter = 4000
burnin = 2000
total = Iter - burnin
N = 500
NG = 5
K = 50
M = 1000
nt = 10
NY = 3
NXI = 1
NX = 2
once = 1
O = 1
t_lam = np.array([[1, 0.6, 0.6]]).T
t_psi = np.array([0.36, 0.36, 0.36])
t_alpha = np.array([1, 1, 1, 1])
t_gamma = np.array([0.5, 0.5, 0.5, 0.5])
t_beta = np.array([0.5, 0.5, 0.5, 0.5])
t_sigma = np.array([[0.5, 0.3], [0.3, 0.5]])
all_lam = np.zeros([Rep, Iter, NY, NXI])
all_psi = np.zeros([Rep, Iter, NY])
all_lambd = np.zeros([Rep, Iter, NG])
all_alpha = np.zeros([Rep, Iter, NX + 2])
# all_alpha1 = np.zeros([Rep, Iter, NX + 2])
# all_alpha2 = np.zeros([Rep, Iter, NX + 2])
all_sigma = np.zeros([Rep, Iter, 2, 2])
all_gamma = np.zeros([Rep, Iter, NX + 2])
all_beta = np.zeros([Rep, Iter, NX + 2])
# all_gamma_beta = np.zeros([Rep, Iter, 2 * (NX + 2)])
#===============Analyze the estimates of parameters===================#######
for o in range(O):
    all_lam[o*once:(o+1)*once] = np.load("lam" + str(o) + ".npy")
    all_psi[o*once:(o+1)*once] = np.load("psi" + str(o) + ".npy")
    all_lambd[o * once:(o + 1) * once] = np.load("lambd" + str(o) + ".npy")
    all_alpha[o * once:(o + 1) * once] = np.load("alpha" + str(o) + ".npy")
    all_gamma[o * once:(o + 1) * once] = np.load("gamma" + str(o) + ".npy")
    all_beta[o * once:(o + 1) * once] = np.load("beta" + str(o) + ".npy")
    all_sigma[o * once:(o + 1) * once] = np.load("sigma" + str(o) + ".npy")
m_lam = np.mean(all_lam[:, burnin:], 1)
m_psi = np.mean(all_psi[:, burnin:], 1)
# m_lambd = np.mean(all_lambd[:, burnin:], 1)
m_alpha = np.mean(all_alpha[:, burnin:], 1)
m_gamma = np.mean(all_gamma[:, burnin:], 1)
m_beta = np.mean(all_beta[:, burnin:], 1)
m_sigma = np.mean(all_sigma[:, burnin:], 1)
m_lambd = np.mean(all_lambd[:, burnin:], 1)
b_lam = np.mean(m_lam, 0) - t_lam
b_psi = np.mean(m_psi, 0) - t_psi
b_alpha = np.mean(m_alpha, 0) - t_alpha
b_gamma = np.mean(m_gamma, 0) - t_gamma
b_beta = np.mean(m_beta, 0) - t_beta
b_sigma = np.mean(m_sigma, 0) - t_sigma
tool1 = tool.Tools(N, N)
sd_lam = tool1.rep_rmse(t_lam, all_lam, burnin)
sd_psi = tool1.rep_rmse(t_psi, all_psi, burnin)
# sd_lambd = tool1.rep_rmse(all_lambd, t_lambd, burnin)
sd_alpha = tool1.rep_rmse(t_alpha, all_alpha, burnin)
sd_gamma = tool1.rep_rmse(t_gamma, all_gamma, burnin)
sd_beta = tool1.rep_rmse(t_beta, all_beta, burnin)
sd_sigma = tool1.rep_rmse(t_sigma, all_sigma, burnin)
msd_lam = np.mean(sd_lam, 0)
msd_psi = np.mean(sd_psi, 0)
msd_alpha = np.mean(sd_alpha, 0)
msd_beta = np.mean(sd_beta, 0)
msd_gamma = np.mean(sd_gamma, 0)
msd_sigma = np.mean(sd_sigma, 0)
print("End of parameters")
E = np.zeros([Rep, 2, 2, total, nt])
tE = np.zeros([Rep, 2, 2, nt])
mE = np.zeros([Rep, 2, 2, nt])
for o in range(O):
    start = once * o
    end = once * (o+1)
    E[start:end] = np.load("E" + str(o) + ".npy")
    tE[start:end] = np.load("tE" + str(o) + ".npy")
    mE[start:end] = np.load("mE" + str(o) + ".npy")
##=========True =========
tte = tE[:, 1, 1] - tE[:, 0, 0]  # Rep * nt
tde = tE[:, 1, 0] - tE[:, 0, 0]
tie = tE[:, 1, 1] - tE[:, 1, 0]
##=========Mean =========
mte = mE[:, 1, 1] - mE[:, 0, 0]  # Rep * nt
mde = mE[:, 1, 0] - mE[:, 0, 0]
mie = mE[:, 1, 1] - mE[:, 1, 0]
##=========Estimates (Rep)=========
ete = E[:, 1, 1] - E[:, 0, 0]  # Rep * Iter * nt
ede = E[:, 1, 0] - E[:, 0, 0]  # Rep * Iter * nt
eie = E[:, 1, 1] - E[:, 1, 0]
tool1 = tool.Tools(N, N)
#################========Rep=========############################
#----------te--------------
bias_te = tool1.bias(tte, ete, 0) # Rep * nt
mbias_te = np.mean(bias_te, 0)
rmse_te = tool1.rmse(tte, ete, 0)
mrmse_te= np.mean(rmse_te, 0)
#-----------de---------
bias_de = tool1.bias(tde, ede, 0) # Rep * nt
mbias_de = np.mean(bias_de, 0)
rmse_de = tool1.rmse(tde, ede, 0)
mrmse_de= np.mean(rmse_de, 0)
#------------ie---------------
bias_ie = tool1.bias(tie, eie, 0) # Rep * nt
mbias_ie = np.mean(bias_ie, 0)
rmse_ie = tool1.rmse(tie, eie, 0)
mrmse_ie= np.mean(rmse_ie, 0)
print("summary is over")