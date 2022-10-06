from __future__ import division
import numpy as np
import tool
from numpy.linalg import cholesky
from scipy.stats import invwishart, multivariate_normal
import scipy.stats as ss
# import scipy
# import time
# import warnings
# # import torch
import numpy.random as nrd



class MCMC(tool.Tools):

    def __init__(self, N, NG, obs, obs_ind, obs_T):
        self.N = N
        self.NG = NG
        self.obs = obs  # observation location
        self.obs_ind = obs_ind    # observation indictor
        self.obs_T = obs_T

    def update_lam_psi(self, psi, lam, Ind, omega, y):
        # Args: Omega(W) : N* NT
        alpha_s0 = 9  # prior
        beta_s0 = 4
        sample = self.obs
        omega = omega[:, :, np.newaxis]
        for k in range(y.shape[-1]):
            fix = np.where(Ind[k] == 0)
            free = np.where(Ind[k] == 1)
            y_star = y[sample[0], sample[1], k] - np.sum(
                lam[k, fix[0]] * omega[sample[0], sample[1]][:, fix[0]], axis=1)  # n
            U = np.transpose(omega[sample[0], sample[1], :][:, free[0]])  # free * N
            H_0 = np.eye(free[0].shape[0]) * 4  # prior for free lam[k,]  free * free
            lam_0 = np.zeros(shape=[free[0].shape[0], 1])  # prior for free lam[k,]    free * 1
            temp = np.linalg.inv(H_0) + np.dot(U, np.transpose(U))  # free * free
            A = np.linalg.inv(temp)
            temp1 = np.dot(np.linalg.inv(H_0), lam_0) + np.dot(U, y_star[:, np.newaxis])  # free * 1
            a = np.dot(A, temp1)  # free * 1
            # for psi
            alpha_s = len(sample[0]) / 2 + alpha_s0
            temp2 = np.sum(y_star * y_star)
            temp3 = np.dot(np.dot(np.transpose(a), temp), a)
            temp4 = np.dot(np.dot(np.transpose(lam_0), np.linalg.inv(H_0)), lam_0)
            beta_s = beta_s0 + 1 / 2 * (temp2 - temp3 + temp4)
            # psi[s, k] = 1 / np.gamma(alpha_s, beta_s[0, 0])
            psi[k] = 1 / nrd.gamma(alpha_s, 1 / beta_s[0, 0])
            if free[0].shape[0] == 1:
                lam[k, free[0]] = nrd.normal(a[0, 0], np.sqrt(psi[k] * A[0, 0]))
            elif free[0].shape[0] > 1:
                lam[k, free[0]] = nrd.multivariate_normal(np.squeeze(a), psi[k] * A)
            # print("psi0:%.3f"%psi[s, 0])
        return lam, psi

    def update_lambd(self, lambd, nu, delta, alpha, X, gamma, beta, b, OT, grid):
        # W = I + S * OT         # size: N
        # prior
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        alpha_w = alpha[-1]
        alpha_x = alpha[:-1]
        alpha_1 = 0.2
        alpha_2 = 0.4
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I)   # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid) / (alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        # vary = vary_grid[:, 1:] - vary_grid[:, :-1]  # integral over (grid[g], grid[g+1])
        for g in range(self.NG):
            v_1 = nu[:, g] * (np.exp(alpha_w * S * OT) - np.exp(alpha_w * S * grid[g]))/ (alpha_w * S)   # integral of last interval
            v_2 = 0
            if g != self.NG-1:
                for j in range(g + 1, self.NG):
                    v_2 += nu[:, j] * (vary_grid[:, g+1] - vary_grid[:, g])  # N
                    # v_2 += nu[:, j] * (t[j + 1] - t[j])
            alpha_2_p = alpha_2 + np.sum(fixed * (v_1 + v_2))
            alpha_1_p = np.sum(nu[:, g] * delta) + alpha_1
            lambd[g] = nrd.gamma(alpha_1_p, 1/alpha_2_p)
        return lambd

    def update_alpha(self, alpha, X, gamma, beta, b, OT, nu, interval_nu, delta, lambd, grid, c_alpha, accept_alpha):
        alpha_0 = np.zeros_like(alpha)   # prior
        sigma_0 = np.eye(alpha.shape[0])
        star = nrd.multivariate_normal(alpha, c_alpha * np.eye(alpha.shape[0]))
        # star[-1] = alpha[-1]
        # star = alpha + 1 / 100
        alpha_w = alpha[-1]
        alpha_x = alpha[:-1]
        star_w = star[-1]
        star_x = star[:-1]
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        ##----------------hazard part----------------------------########
        W = I + S * OT
        n_x = np.concatenate((X[:, 1:], W[:, np.newaxis]), 1)
        cond = np.sum(alpha * n_x, 1)
        cond_star = np.sum(star * n_x, 1)
        ratio_1 = delta * cond
        star_1 = delta * cond_star
        ##----------------survival prob part (involves integral)---------------------##########
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S) # sum N * NG (only on dim is not 0) to N
        fixed_star = np.exp(np.sum(star_x * X[:, 1:], 1) + star_w * I, dtype=np.float128)  # N
        vary_grid_star = np.exp(star_w * S[:, np.newaxis] * grid, dtype=np.float128) / (star_w * S[:, np.newaxis])  # N * (NG + 1)
        vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        vary_last_star = np.sum(nu * lambd * (np.exp(star_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(star_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (star_w * S) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        integral_star = cum_vary_star + vary_last_star
        # sci = np.zeros_like(S)
        # sciq = np.zeros_like(S)
        # sci_star = np.zeros_like(S)
        # sci_starq = np.zeros_like(S)
        # t = symbols("t")
        # for i in range(S.shape[0]):
        #     sci[i] = integrate(exp(alpha_w * S[i] * t), (t, 0, OT[i]))
        #     sciq[i] = scipy.integrate.quad(lambda t: exp(alpha_w * S[i] * t), 0, OT[i])[0]
        #     sci_star[i] = integrate(exp(star_w * S[i] * t), (t, 0, OT[i]))
        ratio_2 = - fixed * integral
        star_2 = - fixed_star * integral_star
        ratio_3 = - np.dot(np.dot((alpha - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (alpha - alpha_0)[:, np.newaxis]).squeeze()
        star_3 = - np.dot(np.dot((star - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (star - alpha_0)[:, np.newaxis]).squeeze()
        log_ratio = np.sum(star_1 + star_2 - ratio_1 - ratio_2) + (star_3 - ratio_3) / 2
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        # print(ratio)
        rand_ratio = nrd.rand(1)
        if ratio > rand_ratio:
            alpha_out = star
            accept_alpha += 1
        else:
            alpha_out = alpha
        return alpha_out, accept_alpha



    def update_alphax(self, alpha, X, gamma, beta, b, OT, nu, interval_nu, delta, lambd, grid, c_alpha, accept_alphax):
        alphax = alpha[:-1]
        alphaw = alpha[-1]
        alpha_out = alpha.copy()
        alpha_0 = np.zeros_like(alphax)   # prior
        sigma_0 = np.eye(alphax.shape[0])
        starx = nrd.multivariate_normal(alphax, c_alpha * np.eye(alphax.shape[0]))
        # starx = alphax + 1 / 100
        alpha_out[:-1] = starx
        # star = alpha + 1 / 100
        # alpha = np.concatenate((alphax, alphaw))
        # star = np.concatenate((starx, alphaw))
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        ##----------------hazard part----------------------------########
        # W = I + S * OT
        # n_x = np.concatenate((X[:, 1:], W[:, np.newaxis]), 1)
        cond = np.sum(alphax * X[:, 1:], 1)
        cond_star = np.sum(starx * X[:, 1:], 1)
        ratio_1 = delta * cond
        star_1 = delta * cond_star
        ##----------------survival prob part (involves integral)---------------------##########
        fixed = np.exp(np.sum(alphax * X[:, 1:], 1) + alphaw * I, dtype=np.float128)  # N
        vary_grid = np.exp(alphaw * S[:, np.newaxis] * grid, dtype=np.float128) / (alphaw * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alphaw * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alphaw * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alphaw * S) # sum N * NG (only on dim is not 0) to N
        fixed_star = np.exp(np.sum(starx * X[:, 1:], 1) + alphaw * I, dtype=np.float128)  # N
        integral = cum_vary + vary_last
        # sci = np.zeros_like(S)
        # sci_star = np.zeros_like(S)
        # t = symbols("t")
        # for i in range(S.shape[0]):
        #     sci[i] = integrate(exp(alphaw * S[i] * t), (t, 0, OT[i]))
        ratio_2 = - fixed * integral
        star_2 = - fixed_star * integral
        ratio_3 = - np.dot(np.dot((alphax - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (alphax - alpha_0)[:, np.newaxis]).squeeze()
        star_3 = - np.dot(np.dot((starx - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (starx - alpha_0)[:, np.newaxis]).squeeze()
        log_ratio = np.sum(star_1 + star_2 - ratio_1 - ratio_2) + (star_3 - ratio_3) / 2
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        # print(ratio)
        rand_ratio = nrd.rand(1)
        # print(rand_ratio)
        if ratio > rand_ratio:
            return alpha_out, accept_alphax+1
        else:
            return alpha, accept_alphax

    def update_alphaw(self, alpha, X, gamma, beta, b, OT, nu, interval_nu, delta, lambd, grid, c_alpha, accept_alpha):
        alpha_0 = np.zeros_like(alpha)   # prior
        sigma_0 = np.eye(alpha.shape[0])
        star = nrd.multivariate_normal(alpha, c_alpha * np.eye(alpha.shape[0]))
        star[:-1] = alpha[:-1]
        # star[-1] = alpha[-1]
        # star = alpha + 1 / 100
        alpha_w = alpha[-1]
        alpha_x = alpha[:-1]
        star_w = star[-1]
        star_x = star[:-1]
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        ##----------------hazard part----------------------------########
        W = I + S * OT
        n_x = np.concatenate((X[:, 1:], W[:, np.newaxis]), 1)
        cond = np.sum(alpha * n_x, 1)
        cond_star = np.sum(star * n_x, 1)
        ratio_1 = delta * cond
        star_1 = delta * cond_star
        ##----------------survival prob part (involves integral)---------------------##########
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S) # sum N * NG (only on dim is not 0) to N
        fixed_star = np.exp(np.sum(star_x * X[:, 1:], 1) + star_w * I, dtype=np.float128)  # N
        vary_grid_star = np.exp(star_w * S[:, np.newaxis] * grid, dtype=np.float128) / (star_w * S[:, np.newaxis])  # N * (NG + 1)
        vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        vary_last_star = np.sum(nu * lambd * (np.exp(star_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(star_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (star_w * S) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        integral_star = cum_vary_star + vary_last_star
        # sci = np.zeros_like(S)
        # sci_star = np.zeros_like(S)
        # t = symbols("t")
        # for i in range(S.shape[0]):
        #     sci[i] = integrate(exp(alpha_w * S[i] * t), (t, 0, OT[i]))
        #     sci_star[i] = integrate(exp(star_w * S[i] * t), (t, 0, OT[i]))
        ratio_2 = - fixed * integral
        star_2 = - fixed_star * integral_star
        ratio_3 = - np.dot(np.dot((alpha - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (alpha - alpha_0)[:, np.newaxis]).squeeze()
        star_3 = - np.dot(np.dot((star - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
                           (star - alpha_0)[:, np.newaxis]).squeeze()
        log_ratio = np.sum(star_1 + star_2 - ratio_1 - ratio_2) + (star_3 - ratio_3) / 2
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        # print(ratio)
        rand_ratio = nrd.rand(1)
        if ratio > rand_ratio:
            alpha_out = star
            accept_alpha += 1
        else:
            alpha_out = alpha
        return alpha_out, accept_alpha


    # def update_alpha(self, alpha, X, I, S, OT, nu, delta, lambd, grid, c_alpha, accept_alpha):
    #     alpha_0 = np.zeros_like(alpha)   # prior
    #     sigma_0 = np.eye(alpha.shape[0])
    #     W = I + S * OT         # size: N
    #     n_x = np.concatenate((X[:, 1:], W[:, np.newaxis]), 1)  # N * NX
    #     x_x = np.matmul(n_x[:, :, np.newaxis], n_x[:, np.newaxis, :])
    #     const = 0
    #     v_3 = 0
    #     for g in range(self.NG):
    #         v_3 += nu[:, g] * delta
    #         v_1 = lambd[g] * (OT - grid[g])
    #         v_2 = 0
    #         if g != 0:
    #             for j in range(g):
    #                 v_2 += lambd[j] * (grid[j + 1] - grid[j])
    #         const += nu[:, g] * (v_1 + v_2)
    #     alpha_sigma = np.linalg.inv(np.sum(const[:, np.newaxis, np.newaxis] * x_x, 0) + np.linalg.inv(sigma_0))
    #     alpha_star = nrd.multivariate_normal(alpha, c_alpha * alpha_sigma)
    #     # alpha_star = nrd.multivariate_normal(alpha, c_alpha * np.eye(alpha.shape[0]))
    #     cond = np.sum(n_x * alpha, 1)
    #     cond_star = np.sum(n_x * alpha_star, 1)
    #     ratio_1 = v_3 * (cond_star - cond)
    #     ratio_2 = const * (np.exp(cond) - np.exp(cond_star))
    #     prior_ratio = - np.dot(np.dot((alpha - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
    #                        (alpha - alpha_0)[:, np.newaxis]).squeeze()
    #     prior_ratio_star = - np.dot(np.dot((alpha_star - alpha_0)[np.newaxis, :], np.linalg.inv(sigma_0)),
    #                        (alpha_star - alpha_0)[:, np.newaxis]).squeeze()
    #     ratio_3 = 1/ 2 *(prior_ratio - prior_ratio_star)
    #     log_ratio = np.sum(ratio_1 + ratio_2) + ratio_3
    #     ratio = np.exp(log_ratio) if log_ratio < 0 else 1
    #     # print(ratio)
    #     rand_ratio = nrd.rand(1)
    #     if ratio > rand_ratio:
    #         alpha_out = alpha_star
    #         accept_alpha += 1
    #     else:
    #         alpha_out = alpha
    #     return alpha_out, accept_alpha

    def update_sigma(self, b):
        # Args: b: N * 2
        var = np.dot(b.T, b)
        sigma_0 = 5
        sigma_1 = 6 * np.eye(b.shape[-1])
        sigma = invwishart.rvs(b.shape[0] + sigma_0, var + sigma_1)
        return sigma

    def update_gamma(self, gamma, beta, X, sigma, y, lam, psi, OT, alpha, nu, interval_nu, delta, lambd, grid, c_gamma, accept_gamma, t_S):
        # t4 = time.time()
        I_S = self.gen_is(gamma, beta, X, sigma)
        I = I_S[:, 0]
        # S = I_S[:, 1]
        S = ss.norm.rvs(np.sum(beta * X, 1), np.sqrt(sigma[-1, -1]))
        # S = np.sum(beta * X, 1)
        # S = t_S
        gamma_star = nrd.multivariate_normal(gamma, c_gamma * np.eye(gamma.shape[0]))
        # beta_star = nrd.multivariate_normal(beta, c_beta * np.eye(beta.shape[0]))
        beta_star = beta     # ???
        I_Sstar = self.gen_is(gamma_star, beta_star, X, sigma)
        I_star = I_Sstar[:, 0]
        # S_star = I_Sstar[:, 1]   #
        S_star = S
        # t0 = time.time()
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I_star[:, np.newaxis] + S_star[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]   # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        star_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        for k in range(y.shape[-1]):
            ratio_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my[self.obs[0], self.obs[1], k], scale=np.sqrt(psi[k]))
            star_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my_star[self.obs[0], self.obs[1], k], scale=np.sqrt(psi[k]))
        ###==============Cox model=========================######
        # t1 = time.time()
        # alpha_x = alpha[:-1]
        # alpha_w = alpha[-1]
        # W_c = I + S * OT
        # W_cstar = I_star + S_star * OT
        # ratio_2 = alpha_w * W_c * delta # N
        # star_2 = alpha_w * W_cstar * delta
        # # t2 = time.time()
        # fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I,  dtype=np.float128) # N
        # fixed_star = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I_star,  dtype=np.float128) # N
        # vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        # vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        # cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        # vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S) # sum N * NG (only on dim is not 0) to N
        # # vary_grid_star = np.exp(star_w * S[:, np.newaxis] * grid, dtype=np.float128) / (star_w * S[:, np.newaxis])  # N * (NG + 1)
        # # vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        # # cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        # # vary_last_star = np.sum(nu * lambd * (np.exp(star_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(star_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (star_w * S) # sum N * NG (only on dim is not 0) to N
        # integral = cum_vary + vary_last
        # # t3 = time.time()
        # # integral_star = cum_vary_star + vary_last_star
        # ratio_3 = - fixed * integral  # N
        # star_3 = - fixed_star * integral
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        log_ratio = np.sum(star_1 - ratio_1)
        # log_ratio = np.sum(star_1 - ratio_1) + np.sum(star_2 + star_3 - ratio_2 - ratio_3)
        rand_ratio = nrd.rand(1)
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        if ratio > rand_ratio:
            return gamma_star, accept_gamma + 1
        else:
            return gamma, accept_gamma


    def update_gamma(self, gamma, beta, X, b, y, lam, psi,  OT, alpha, nu, interval_nu, delta, lambd, grid, c_gamma, accept_gamma):
        # t4 = time.time()
        Im = np.sum(gamma * X, 1)
        I = Im + b[:, 0] # N
        Sm = np.sum(beta * X, 1)
        S = Sm + b[:, 1] # N
        gamma_star = nrd.multivariate_normal(gamma, c_gamma * np.eye(gamma.shape[0]))
        Ims = np.sum(gamma_star * X, 1)
        I_star = Ims + b[:, 0]
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I_star[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]   # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        star_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        for k in range(y.shape[-1]):
            ratio_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my[self.obs[0], self.obs[1], k], scale=np.sqrt(psi[k]))
            star_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my_star[self.obs[0], self.obs[1], k], scale=np.sqrt(psi[k]))
        ##==============Cox model=========================######
        # t1 = time.time()
        alpha_x = alpha[:-1]
        alpha_w = alpha[-1]
        W_c = I + S * OT
        W_cstar = I_star + S * OT
        ratio_2 = alpha_w * W_c * delta # N
        star_2 = alpha_w * W_cstar * delta
        # t2 = time.time()
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I,  dtype=np.float128) # N
        fixed_star = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I_star,  dtype=np.float128) # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S) # sum N * NG (only on dim is not 0) to N
        # vary_grid_star = np.exp(star_w * S[:, np.newaxis] * grid, dtype=np.float128) / (star_w * S[:, np.newaxis])  # N * (NG + 1)
        # vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        # cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        # vary_last_star = np.sum(nu * lambd * (np.exp(star_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(star_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (star_w * S) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        # t3 = time.time()
        # integral_star = cum_vary_star + vary_last_star
        ratio_3 = - fixed * integral  # N
        star_3 = - fixed_star * integral
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        # star_2 = star_3 = ratio_2 = ratio_3 = 0
        log_ratio = np.sum(star_1 - ratio_1) + np.sum(star_2 + star_3 - ratio_2 - ratio_3)
        rand_ratio = nrd.rand(1)
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        if ratio > rand_ratio:
            return gamma_star, accept_gamma + 1
        else:
            return gamma, accept_gamma

    def update_beta(self, beta, gamma, X, b, y, lam, psi,  OT, alpha, nu, interval_nu, delta, lambd, grid, c_beta, accept_beta):
        I = np.sum(gamma * X, 1) + b[:,0]
        S = np.sum(beta * X, 1) + b[:, 1]
        beta_star = nrd.multivariate_normal(beta, c_beta * np.eye(beta.shape[0]))
        S_star = np.sum(beta_star * X, 1) + b[:, 1]
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I[:, np.newaxis] + S_star[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]  # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        star_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        for k in range(y.shape[-1]):
            ratio_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my[self.obs[0], self.obs[1], k],
                                           scale=np.sqrt(psi[k]))
            star_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my_star[self.obs[0], self.obs[1], k],
                                          scale=np.sqrt(psi[k]))
        ##==============Cox model=========================######
        # t1 = time.time()
        alpha_x = alpha[:-1]
        alpha_w = alpha[-1]
        W_c = I + S * OT
        W_cstar = I + S_star * OT
        ratio_2 = alpha_w * W_c * delta  # N
        star_2 = alpha_w * W_cstar * delta
        # t2 = time.time()
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (
                    alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(
            alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1) / (
                                alpha_w * S)  # sum N * NG (only on dim is not 0) to N
        vary_grid_star = np.exp(alpha_w * S_star[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S_star[:, np.newaxis])  # N * (NG + 1)
        vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        vary_last_star = np.sum(nu * lambd * (np.exp(alpha_w * S_star * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S_star[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S_star) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        # t3 = time.time()
        integral_star = cum_vary_star + vary_last_star
        ratio_3 = - fixed * integral  # N
        star_3 = - fixed * integral_star
        # star_2 = star_3 = ratio_2 = ratio_3 = 0
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        log_ratio = np.sum(star_1 - ratio_1) + np.sum(star_2 + star_3 - ratio_2 - ratio_3)
        rand_ratio = nrd.rand(1)
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        if ratio > rand_ratio:
            return beta_star, accept_beta + 1
        else:
            return beta, accept_beta

    def update_gamma_beta(self, gamma, beta, X, rx, b, y, lam, psi,  OT, alpha, nu, interval_nu, delta, lambd, grid, c_beta, accept_beta):
        xt = self.obs_T[:, :, np.newaxis] * X[:, np.newaxis]  # N * NT * x
        nx = np.concatenate((rx, xt), 2)  # N * NT * (2x）
        x_square = np.matmul(nx[self.obs[0], self.obs[1]].T, nx[self.obs[0], self.obs[1]])
        lam_psi = np.square(lam[:,0]) / psi  # NY
        psigma = np.sum(lam_psi * x_square[:, :, np.newaxis], 2)
        gamma_beta = np.hstack((gamma, beta))
        star = nrd.multivariate_normal(gamma_beta, c_beta * np.linalg.inv(psigma))
        I = np.sum(gamma * X, 1) + b[:, 0]
        S = np.sum(beta * X, 1) + b[:, 1]
        gamma_star = star[:X.shape[1]]
        beta_star = star[X.shape[1]:]
        S_star = np.sum(beta_star * X, 1) + b[:, 1]
        I_star = np.sum(gamma_star * X, 1) + b[:, 0]
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I_star[:, np.newaxis] + S_star[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]  # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        star_1 = np.zeros_like(y[self.obs[0], self.obs[1]])
        for k in range(y.shape[-1]):
            ratio_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my[self.obs[0], self.obs[1], k],
                                           scale=np.sqrt(psi[k]))
            star_1[:, k] = ss.norm.logpdf(y[self.obs[0], self.obs[1], k], loc=my_star[self.obs[0], self.obs[1], k],
                                          scale=np.sqrt(psi[k]))
        ##==============Cox model=========================######
        # t1 = time.time()
        alpha_x = alpha[:-1]
        alpha_w = alpha[-1]
        W_c = I + S * OT
        W_cstar = I_star + S_star * OT
        ratio_2 = alpha_w * W_c * delta  # N
        star_2 = alpha_w * W_cstar * delta
        # t2 = time.time()
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (
                    alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(
            alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1) / (
                                alpha_w * S)  # sum N * NG (only on dim is not 0) to N
        fixed_star = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I_star, dtype=np.float128)  # N
        vary_grid_star = np.exp(alpha_w * S_star[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S_star[:, np.newaxis])  # N * (NG + 1)
        vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        vary_last_star = np.sum(nu * lambd * (np.exp(alpha_w * S_star * OT, dtype=np.float128)[:, np.newaxis] - np.exp(alpha_w * S_star[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (alpha_w * S_star) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        # t3 = time.time()
        integral_star = cum_vary_star + vary_last_star
        ratio_3 = - fixed * integral  # N
        star_3 = - fixed_star * integral_star
        # star_2 = star_3 = ratio_2 = ratio_3 = 0
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        log_ratio = np.sum(star_1 - ratio_1) + np.sum(star_2 + star_3 - ratio_2 - ratio_3)
        rand_ratio = nrd.rand(1)
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        if ratio > rand_ratio:
            return star, accept_beta + 1
        else:
            return gamma_beta, accept_beta



    def update_b0(self, b, gamma, beta, X, y, lam, psi, OT, alpha, nu, interval_nu, delta, lambd, grid, sigma, c_b, accept_b):
        b_out = b.copy()
        b0 = b[:, 0] # affect I
        b_star = nrd.normal(b0, c_b)
        Im = np.sum(gamma * X, 1)
        I = Im + b0 # N
        I_star = Im + b_star
        Sm = np.sum(beta * X, 1)
        S = Sm + b[:, 1]  # N
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I_star[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]  # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y)
        star_1 = np.zeros_like(y)
        for k in range(y.shape[-1]):
            ratio_1[:, :, k] = ss.norm.logpdf(y[:, :, k], loc=my[:, :, k], scale=np.sqrt(psi[k]))
            star_1[:, :, k] = ss.norm.logpdf(y[:, :, k], loc=my_star[:, :, k], scale=np.sqrt(psi[k]))
        ##==============Cox model=========================######
        # t1 = time.time()
        alpha_x = alpha[:-1]
        alpha_w = alpha[-1]
        W_c = I + S * OT
        W_cstar = I_star + S * OT
        ratio_2 = alpha_w * W_c * delta  # N
        star_2 = alpha_w * W_cstar * delta
        # t2 = time.time()
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        fixed_star = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I_star, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (
                    alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(
            alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1) / (
                                alpha_w * S)  # sum N * NG (only on dim is not 0) to N
        # vary_grid_star = np.exp(star_w * S[:, np.newaxis] * grid, dtype=np.float128) / (star_w * S[:, np.newaxis])  # N * (NG + 1)
        # vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        # cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        # vary_last_star = np.sum(nu * lambd * (np.exp(star_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(star_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1)/ (star_w * S) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        # t3 = time.time()
        # integral_star = cum_vary_star + vary_last_star
        ratio_3 = - fixed * integral  # N
        star_3 = - fixed_star * integral
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        ratio_4 = ss.multivariate_normal.logpdf(b, mean=np.zeros([2]), cov=sigma)
        star_b = b.copy()
        star_b[:, 0] = b_star
        star_4 = ss.multivariate_normal.logpdf(star_b, mean=np.zeros([2]), cov=sigma)
        # star_2 = star_3 = ratio_2 = ratio_3 = 0
        log_ratio = np.sum((star_1 - ratio_1) * self.obs_ind[:, :, np.newaxis], (1, 2)) + star_2 + star_3 + star_4 \
                    - ratio_2 - ratio_3 - ratio_4
        rand_ratio = nrd.rand(self.N)
        loc = np.where(log_ratio < 0)[0]
        ratio = np.ones([self.N])
        ratio[loc] = np.exp(log_ratio[loc])
        accept_loc = np.where(ratio > rand_ratio)[0]
        b_out[accept_loc, 0] = b_star[accept_loc]
        accept_b[accept_loc] += 1
        return b_out, accept_b

    def update_b1(self, b, gamma, beta, X, y, lam, psi, OT, alpha, nu, interval_nu, delta, lambd, grid, sigma, c_b,
                  accept_b):
        b_out = b.copy()
        b1 = b[:, 1]  # affect I
        b_star = nrd.normal(b1, c_b)
        Im = np.sum(gamma * X, 1)
        I = Im + b[:, 0]  # N
        Sm = np.sum(beta * X, 1)
        S = Sm + b1  # N
        S_star = Sm + b_star
        ########=============Factor analysis model===============#############
        W_f = I[:, np.newaxis] + S[:, np.newaxis] * self.obs_T
        W_fstar = I[:, np.newaxis] + S_star[:, np.newaxis] * self.obs_T
        my = np.squeeze(lam) * W_f[:, :, np.newaxis]  # N * MT * NY
        my_star = np.squeeze(lam) * W_fstar[:, :, np.newaxis]
        ratio_1 = np.zeros_like(y)
        star_1 = np.zeros_like(y)
        for k in range(y.shape[-1]):
            ratio_1[:, :, k] = ss.norm.logpdf(y[:, :, k], loc=my[:, :, k], scale=np.sqrt(psi[k]))
            star_1[:, :, k] = ss.norm.logpdf(y[:, :, k], loc=my_star[:, :, k], scale=np.sqrt(psi[k]))
        ##==============Cox model=========================######
        # t1 = time.time()
        alpha_x = alpha[:-1]
        alpha_w = alpha[-1]
        W_c = I + S * OT
        W_cstar = I + S_star * OT
        ratio_2 = alpha_w * W_c * delta  # N
        star_2 = alpha_w * W_cstar * delta
        # t2 = time.time()
        fixed = np.exp(np.sum(alpha_x * X[:, 1:], 1) + alpha_w * I, dtype=np.float128)  # N
        vary_grid = np.exp(alpha_w * S[:, np.newaxis] * grid, dtype=np.float128) / (
                alpha_w * S[:, np.newaxis])  # N * (NG + 1)
        vary = (vary_grid[:, 1:] - vary_grid[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary = np.sum(vary * interval_nu, 1)  # cumulation until the grid before ET
        vary_last = np.sum(nu * lambd * (np.exp(alpha_w * S * OT, dtype=np.float128)[:, np.newaxis] - np.exp(
            alpha_w * S[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1) / (
                            alpha_w * S)  # sum N * NG (only on dim is not 0) to N
        vary_grid_star = np.exp(alpha_w * S_star[:, np.newaxis] * grid, dtype=np.float128) / (alpha_w * S_star[:, np.newaxis])  # N * (NG + 1)
        vary_star = (vary_grid_star[:, 1:] - vary_grid_star[:, :-1]) * lambd  # integral over (grid[g], grid[g+1])  # N * NG
        cum_vary_star = np.sum(vary_star * interval_nu, 1)  # cumulation until the grid before OT
        vary_last_star = np.sum(nu * lambd * (np.exp(alpha_w * S_star * OT, dtype=np.float128)[:, np.newaxis] \
            - np.exp(alpha_w * S_star[:, np.newaxis] * grid[:-1], dtype=np.float128)), 1) / (alpha_w * S_star) # sum N * NG (only on dim is not 0) to N
        integral = cum_vary + vary_last
        # t3 = time.time()
        integral_star = cum_vary_star + vary_last_star
        ratio_3 = - fixed * integral  # N
        star_3 = - fixed * integral_star
        # print("star is %.6f, Factor is %.6f, cox 1 is %.3f, cox 2 is %.6f"%(t0-t4, t1 - t0, t2- t1, t3 - t2))
        ratio_4 = ss.multivariate_normal.logpdf(b, mean=np.zeros([2]), cov=sigma)
        star_b = b.copy()
        star_b[:, 1] = b_star
        star_4 = ss.multivariate_normal.logpdf(star_b, mean=np.zeros([2]), cov=sigma)
        # star_2 = star_3 = ratio_2 = ratio_3 = 0
        log_ratio = np.sum((star_1 - ratio_1) * self.obs_ind[:, :, np.newaxis], (1, 2)) + star_2 + star_3 + star_4 \
            - ratio_2 - ratio_3 - ratio_4
        rand_ratio = nrd.rand(self.N)
        loc = np.where(log_ratio < 0)[0]
        ratio = np.ones([self.N])
        ratio[loc] = np.exp(log_ratio[loc])
        accept_loc = np.where(ratio > rand_ratio)[0]
        b_out[accept_loc, 1] = b_star[accept_loc]
        accept_b[accept_loc] += 1
        return b_out, accept_b

    def surv_prob(self, tp, td, t_nu, z, z_s, alpha, beta, gamma, X, sigma, lambd, M):
        ##Args:K is number of timepoints; M is sampling number of random effects; tp = Nt * K
        ##t_nu is the    (Nt * K * N_grid);  td is the time difference of tp (Nt * K)
        I_S = self.gen_is(gamma, beta, X, sigma, M, z_s)  # M * N * 2
        I = I_S[:, :, 0]  # M * N
        S = I_S[:, :, 1]  # # M * N
        hazard = np.sum(t_nu * lambd, 2) # Nt * K
        alpha_w = alpha[-1]
        alpha_z = alpha[0]  # coefficient of treatment
        alpha_x = alpha[1:-1]  # coefficient of confounder
        cum_hazard = np.sum(hazard * np.exp(alpha_w * S[:, :, np.newaxis, np.newaxis] * tp) * td, 3) # M * N * Nt
        fixed = np.exp(alpha_z * z + np.sum(alpha_x * X[:, 2:], 1) + alpha_w * I)  #M * N
        sur_prob = np.mean(np.exp(-fixed[:, :, np.newaxis] * cum_hazard), 0) # N * Nt
        return sur_prob

    def surv_prob1(self, tp, td, t_nu, z, z_s, alpha, beta, gamma, b, X, lambd):
        ##Args:K is number of timepoints; M is sampling number of random effects; tp = Nt * K
        ##t_nu is the    (Nt * K * N_grid);  td is the time difference of tp (Nt * K)
        I_S = self.gen_isb(gamma, beta, X, b, z_s)  #  N * 2
        I = I_S[:, 0]  # N
        S = I_S[:, 1]  # N
        hazard = np.sum(t_nu * lambd, 2) # Nt * K
        alpha_w = alpha[-1]
        alpha_z = alpha[0]  # coefficient of treatment
        alpha_x = alpha[1:-1]  # coefficient of confounder
        cum_hazard = np.sum(hazard * np.exp(alpha_w * S[:, np.newaxis, np.newaxis] * tp) * td, 2) #N * Nt
        fixed = np.exp(alpha_z * z + np.sum(alpha_x * X[:, 2:], 1) + alpha_w * I)  #N
        sur_prob = np.exp(-fixed[:, np.newaxis] * cum_hazard) # N * Nt
        return sur_prob









