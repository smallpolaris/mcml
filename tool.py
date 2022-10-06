from __future__ import division
import numpy as np
import math
import scipy
import scipy.stats as ss

class Tools(object):
    def __init__(self, N, obs_T):
        self.N = N
        self.obs_T = obs_T

        # l is an array, find  y is in which interval, eg, l=[0,1,2,3], y =[2.4, 3.6], return 3, 4
    def find_obs_num(self, y, l):
        rep_y = np.tile(y[:, np.newaxis], (1, l.shape[0]))
        ind = rep_y >= np.tile(l, (y.shape[0], 1))
        return np.sum(ind, axis=1)

    def gen_is(self, gamma, beta, X, phi, M, z):
        ##=============Args: return I_S(M * N*2)
        n_X = X.copy()
        n_X[:, 1] = z   #  for X, (intercept,  treatment, ...)
        mi = np.sum(gamma * n_X, 1)[:, np.newaxis]
        ms = np.sum(beta * n_X, 1)[:, np.newaxis]
        mm = np.concatenate((mi, ms), 1)  # N * 2
        # I_S = np.zeros([mm.shape[0], 2])
        # for i in range(I_S.shape[0]):
        #     I_S[i] = ss.multivariate_normal.rvs(mm[i], phi, 1)
        I_S = ss.multivariate_normal.rvs(np.zeros(mm.shape[-1]), phi, (M, mm.shape[0])) + mm  # cuz phi is the same for each person
        return I_S


    def gen_isb(self, gamma, beta, X, b, z):
        ##=============Args: return I_S(M * N*2)
        n_X = X.copy()
        n_X[:, 1] = z   #  for X, (intercept,  treatment, ...)
        mi = np.sum(gamma * n_X, 1)[:, np.newaxis]
        ms = np.sum(beta * n_X, 1)[:, np.newaxis]
        mm = np.concatenate((mi, ms), 1)  # N * 2
        # I_S = np.zeros([mm.shape[0], 2])
        # for i in range(I_S.shape[0]):
        #     I_S[i] = ss.multivariate_normal.rvs(mm[i], phi, 1)
        # I_S = ss.multivariate_normal.rvs(np.zeros(mm.shape[-1]), phi, (M, mm.shape[0])) + mm  # cuz phi is the same for each person
        I_S = mm + b
        return I_S

    def update_w(self, gamma, beta, X, b):
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        W = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        return W

    def gen_w(self, gamma, beta, X, phi):
        mi = np.sum(gamma * X, 1)[:, np.newaxis]
        ms = np.sum(beta * X, 1)[:, np.newaxis]
        mm = np.concatenate((mi, ms), 1)
        I_S = np.zeros([self.N, 2])
        for i in range(self.N):
            I_S[i] = ss.multivariate_normal.rvs(mm[i], phi, 1)
        I = I_S[:, 0]
        S = I_S[:, 1]
        W = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        return W

    def mean(self, estimated, burnin):
        estimated = estimated[:, burnin:, ]
        estimated_mean = np.sum(estimated, axis=1) / estimated.shape[1]
        return estimated_mean

    def bias(self, true, estimated, burnin):
        estimated = estimated[:, burnin:, ]
        estimated_mean = np.sum(estimated, axis=1) / estimated.shape[1]
        bias = estimated_mean - true
        return bias

    def sd(self, true, estimated, burnin):
        estimated = estimated[burnin:, ]
        sd = np.sqrt(np.sum(np.power(estimated - true, 2), axis=0) / estimated.shape[0])
        return sd

    def rep_rmse(self, true, estimated, burnin):  # rmse
        estimated = estimated[:, burnin:, ]
        rmse = np.sqrt(np.sum(np.power(estimated - true[np.newaxis, np.newaxis], 2), axis=1) / estimated.shape[1])
        return rmse

    def rmse(self, true, estimated, burnin):  # rmse
        estimated = estimated[:, burnin:, ]
        rmse = np.sqrt(np.sum(np.power(estimated - true[:, np.newaxis], 2), axis=1) / estimated.shape[1])
        return rmse


    def std(self, T, v):
        v_m = np.sum(v, axis=(0, 1)) / np.sum(T)
        total = 0
        for i in range(v.shape[0]):
            total += np.sum(np.power(v[i, :T[i]] - v_m, 2), axis=0)
        return np.sqrt(total / np.sum(T))