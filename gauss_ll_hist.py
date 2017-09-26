import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
from theano.ifelse import ifelse
from theano import gradient
import theano
import sklearn.mixture as mixture, sklearn.utils.extmath as gmm_utils
from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode
import time

class GaussLLHistModel:


    # def __init__(self):
    #     pass


    def __init__(self, sample_num=10, gm_num = 2,bin_num = 100):
        self.bin_num = bin_num
        # self.binw = 200000.0/self.bin_num
        self.sample_num = sample_num
        # self.hstart = -100000
        self.gm_num = gm_num
        self.hmin = 0.0
        self.hmax = 1.0
        self.min_cov = 1e-6
        self.reg_coef = 1e-5
        self.initialize_calc_ll_gmm_hist_fun()
        

    def forward(self,X,Yp,Yn):
        X = X.astype(np.float64)
        Yp = Yp.astype(np.float64)
        Yn = Yn.astype(np.float64)
        self.X = X.copy()
        self.Yp = Yp.copy()
        self.Yn = Yn.copy()
        mean,cov,weights,score = self.build_gmm(X)
        self.mean = mean.copy().flatten()
        self.cov = cov.copy().flatten()
        self.weights = weights.copy().flatten()
        return self.gmmhist_f(mean.flatten(),cov.flatten(),weights.flatten(),Yp,Yn)[0]

    def backward(self,X,Yp,Yn):
        X = X.astype(np.float64)
        Yp = Yp.astype(np.float64)
        Yn = Yn.astype(np.float64)
        self.forward(X,Yp,Yn)
        dYp,dYn,dX = self.calc_gmm_probs_dif(self.X,self.Yp,self.Yn,self.mean,self.cov,self.weights)
        return dX.reshape(self.X.shape),dYp,dYn

    def calc_ll_gmm(self, Y, means, covars, weights):
        n_samples, n_dim = Y.shape
        lpr = (-0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)
                      + T.sum((means ** 2) / covars, 1)
                      - 2 * T.dot(Y, (means / covars).T)
                      + T.dot(Y ** 2, T.transpose(1.0 / covars))) + T.log(weights))
        lpr = T.transpose(lpr, (1,0))
        # Use the max to normalize, as with the log this is what accumulates
        # the less errors
        vmax = T.max(lpr,axis=0)
        out = T.log(T.sum(T.exp(lpr- vmax), axis=0))
        out += vmax
        # responsibilities = out
        responsibilities = T.exp(lpr - T.tile(out, (means.shape[0],1)))
        # logprob = T.log(T.sum(T.exp(lpr), axis=1))
        return out, responsibilities, T.transpose(lpr)


    def calc_ll_gmm_noth(self, Y, means, covars, weights):
        n_samples, n_dim = Y.shape
        lpr = (-0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                      + np.sum((means ** 2) / covars, 1)
                      - 2 * np.dot(Y, (means / covars).T)
                      + np.dot(Y ** 2, np.transpose(1.0 / covars))) + np.log(weights))

        lpr = np.transpose(lpr, (1,0))
        # Use the max to normalize, as with the log this is what accumulates
        # the less errors
        vmax = np.max(lpr,axis=0)
        out = np.log(np.sum(np.exp(lpr- vmax), axis=0))
        out += vmax

        responsibilities = np.exp(lpr - np.tile(out, (means.shape[0], 1)))

        # logprob = T.log(T.sum(T.exp(lpr), axis=1))
        return out, responsibilities, np.transpose(lpr)

    def calc_ll_gmm_single(self, Y, means, covars, weights):
        n_dim = Y.shape[0]
        lpr = (-0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)
                       + T.sum((means ** 2) / covars, 1)
                       - 2 * T.dot(Y, (means / covars).T)
                       + T.dot(Y ** 2, T.transpose(1.0 / covars))) + T.log(weights))
        logprob = T.log(T.sum(T.exp(lpr)))
        return logprob


            # def calc_ll_gmm_single(self, y, means, covars, weights):
    #     n_dim = y.shape[0]
    #     lpr = (-0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)
    #                    + T.sum((means ** 2) / covars, 1)
    #                    - 2 * T.dot(y, (means / covars).T)
    #                    + T.dot(y ** 2, T.transpose(1.0 / covars))) + T.log(weights))
    #     logprob = T.log(T.sum(T.exp(lpr), axis=1))
    #     return logprob



            # scan_result, scan_updates = theano.scan(fn = lambda ind, A, means, covars, weights, Y: A+weights[ind]*self.calc_gauss_fun_theano(Y, means[ind, :], covars[ind, :]),
        #             outputs_info=T.zeros_like(Y[:,0]),
        #             sequences=T.arange(self.gm_num),
        #             non_sequences=[means, covars, weights, Y])
        #
        # return T.log(scan_result[-1])

    def initialize_calc_ll_gmm_hist_fun(self):
        Yvec = T.dvector('Y')
        meansvec = T.dvector('means')
        covarsvec = T.dvector('covars')
        weights = T.dvector('weights')
        lam = T.dscalar('lambda')
        gm_num = weights.shape[0]
        ndim = meansvec.shape[0] / gm_num
        Y = T.reshape(Yvec, (Yvec.shape[0] / ndim, ndim))
        means = T.reshape(meansvec, (gm_num, meansvec.shape[0] / gm_num))
        covars = T.reshape(covarsvec, (gm_num, meansvec.shape[0] / gm_num))
        LL, resps, LL_m = self.calc_ll_gmm(Y, means,
                              covars,
                              weights)
        
        LL_lag = T.sum(LL) + lam * (T.sum(weights) - 1)
        LL_sum = T.sum(LL)
        self.gmm_f = function([Yvec, meansvec, covarsvec, weights, lam], LL_lag)
        
        LLg = gradient.jacobian(LL_lag, [Yvec, meansvec, covarsvec, weights, lam])
        
        LL_sum_g = gradient.jacobian(LL_sum, [Yvec, meansvec, covarsvec, weights])
        
        LL_g = gradient.jacobian(LL, [Yvec, meansvec, covarsvec, weights])
        
        self.gmm_df = function([Yvec, meansvec, covarsvec, weights], LL_g, allow_input_downcast=True)
        self.f = function([Yvec, meansvec, covarsvec, weights], LL, allow_input_downcast=True)
        
        
        llhm = gradient.jacobian(LLg[1], [Yvec, meansvec, covarsvec, weights])
        llhc = gradient.jacobian(LLg[2], [Yvec, meansvec, covarsvec, weights])
        llhw = gradient.jacobian(LLg[3], [Yvec, meansvec, covarsvec, weights, lam])
        
        self.gmm_hm = function([Yvec, meansvec, covarsvec, weights, lam], llhm, allow_input_downcast=True)
        self.gmm_hc = function([Yvec, meansvec, covarsvec, weights, lam], llhc, allow_input_downcast=True)
        self.gmm_hw = function([Yvec, meansvec, covarsvec, weights, lam], llhw, allow_input_downcast=True)

        Yvecp = T.dvector('Yp')
        Yvecn = T.dvector('Yn')
        Yp = T.reshape(Yvecp, (Yvecp.shape[0] / ndim, ndim))
        Yn = T.reshape(Yvecn, (Yvecn.shape[0] / ndim, ndim))
        Yp = T.dmatrix('Yp')
        Yn = T.dmatrix('Yn')
        p_p,r_p,p_p_m = self.calc_ll_gmm(Yp, means, covars, weights)
        p_n,r_n,p_n_m = self.calc_ll_gmm(Yn, means, covars, weights)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(p_n, p_p)
        dL = T.jacobian(L, [meansvec, covarsvec, weights, Yp, Yn])
        self.gmmhist_df = function([meansvec, covarsvec, weights, Yp, Yn], dL, allow_input_downcast=True)
        self.gmmhist_f = function([meansvec, covarsvec, weights, Yp, Yn], [L, hmax, hmin, hn, hp], allow_input_downcast=True)

        p_p_1 = T.dvector('p_p_1')
        p_n_1 = T.dvector('p_n_1')
        Lh = self.calc_hist_loss_from_probs(p_p_1, p_n_1, hmin, hmax)
        dLh = T.jacobian(Lh, [p_p_1, p_n_1])
        self.df_hist = function([p_p_1, p_n_1, hmin, hmax], dLh)
        
        hmax1, hmin1 = self.calc_min_max(p_p_1, p_n_1)
        hp1 = self.calc_hist_vals_triang(p_p_1, hmin1, hmax1)
        hn1 = self.calc_hist_vals_triang(p_n_1, hmin1, hmax1)
        self.hist_build = function([p_p_1, p_n_1], [hp1, hn1])


    def solve_lin_sys_for_gmm(self, Xvec, meansvec, covarsvec, weights):
        # backup_path = '/media/hpc-4_Raid/avakhitov/'
        s0 = time.time()

        gm_num = len(weights)
        n_dim = len(meansvec)/len(weights)
        n_samples = len(Xvec)/n_dim
        lam = n_samples
        hm = self.gmm_hm(Xvec, meansvec, covarsvec, weights, lam)
        hc = self.gmm_hc(Xvec, meansvec, covarsvec, weights, lam)
        hw = self.gmm_hw(Xvec, meansvec, covarsvec, weights, lam)
        f0 = time.time()

        mean_row = np.concatenate((hm[1], hm[2], hm[3], np.zeros((len(meansvec), 1))), axis=1)
        cov_row = np.concatenate((hc[1], hc[2], hc[3], np.zeros((len(meansvec), 1))), axis=1)
        weight_row = np.concatenate((hw[1], hw[2], hw[3], np.reshape(hw[4], (gm_num, 1))), axis=1)
        lambda_row = np.concatenate(
            (np.zeros((1, len(meansvec))),
             np.zeros((1, len(meansvec))),
             np.reshape(hw[4], (1, gm_num)),
             np.zeros((1, 1))), axis=1)

        M = np.concatenate((mean_row, cov_row, weight_row, lambda_row))
        N = np.concatenate((-hm[0], -hc[0], -hw[0], np.zeros((1, hw[0].shape[1]))))

        # M = M + self.reg_coef * np.eye(M.shape[0])

        par_dim = gm_num * n_dim
        a = np.diag(M[0:par_dim, 0:par_dim])
        A = np.diag(a)
        b = np.diag(M[0:par_dim, par_dim:2 * par_dim])
        B = np.diag(b)
        c = np.diag(M[par_dim:2 * par_dim, par_dim:2 * par_dim])
        C = np.diag(c)

        dX = []

        # np.save('/home/avakhitov/M.npy', M)
        # np.save('/home/avakhitov/N.npy', N)

        s1 = 0
        fs1 = 0

        if (np.linalg.norm(A - M[0:par_dim, 0:par_dim]) < 1e-15 and
            np.linalg.norm(B - M[0:par_dim, par_dim:2*par_dim]) < 1e-15 and
            np.linalg.norm(C - M[par_dim:2*par_dim, par_dim:2 * par_dim]) < 1e-15):

            D = M[2 * par_dim:, 2 * par_dim:]

            if (np.linalg.matrix_rank(A) < A.shape[0]):
                a = a + np.ones(a.shape[0])*self.reg_coef
            # if (np.linalg.matrix_rank(B) < B.shape[0]):
            #     b = b + np.ones(a.shape[0]) * self.reg_coef
            if (np.linalg.matrix_rank(C) < C.shape[0]):
                c = c + np.ones(a.shape[0]) * self.reg_coef
            if (np.linalg.matrix_rank(D) < D.shape[0]):
                D = D + np.eye(D.shape[0]) * self.reg_coef

            Di = np.linalg.inv(D)
            e = 1 / (a - b / c * b)
            f = -e * b / c
            # print np.linalg.norm(e * b + f * c)
            # print np.linalg.norm(e * a + f * b - np.ones(200))
            h = (np.ones(a.shape[0]) - f * b) / c
            # print np.linalg.norm(f * b + h * c - np.ones(200))
            # print np.linalg.norm(f * a + h * b)


            # Z1 = np.zeros((par_dim, M.shape[0] - 2 * par_dim))
            # R1 = np.concatenate([np.diag(e), np.diag(f), Z1], axis=1)
            # R2 = np.concatenate([np.diag(f), np.diag(h), Z1], axis=1)
            # R3 = np.concatenate([np.zeros((M.shape[0] - 2 * par_dim, 2 * par_dim)), np.linalg.inv(D)], axis=1)
            # Mi = np.concatenate([R1, R2, R3], axis=0)
            s1 = time.time()
            # dX = Mi.dot(N)

            dX = np.zeros(N.shape)
            n_samples = 10
            for i in range(0, n_samples):
                N1 = N[:, i * n_dim:(i + 1) * n_dim]
                for gi in range(0, gm_num):
                    n_mu_gi = np.diag(N1[gi * n_dim:(gi + 1) * n_dim, 0:n_dim])
                    e_gi = e[gi * n_dim:(gi + 1) * n_dim]
                    n_cov_gi = np.diag(
                        N1[n_dim * gm_num + gi * n_dim:n_dim * gm_num + (gi + 1) * n_dim, 0:n_dim])
                    f_gi = f[gi * n_dim:(gi + 1) * n_dim]
                    h_gi = h[gi * n_dim:(gi + 1) * n_dim]
                    dX[gi * n_dim: (gi + 1) * n_dim, i * n_dim:(i + 1) * n_dim] = np.diag(
                        e_gi * n_mu_gi + f_gi * n_cov_gi)
                    dX[n_dim * gm_num + gi * n_dim: n_dim * gm_num + (gi + 1) * n_dim,
                    i * n_dim: (i + 1) * n_dim] = np.diag(f_gi * n_mu_gi + h_gi * n_cov_gi)

            dX[n_dim * 2 * gm_num:, :] = Di.dot(N[n_dim * 2 * gm_num:, :])

            fs1 = time.time()
            # if (len(np.nonzero(np.isnan(dX))[0]) > 0 or len(np.nonzero(np.isinf(dX))[0]) > 0):
            #     np.save(backup_path +'Me.npy', M)
            #     np.save(backup_path +'Ne.npy', N)
            #     np.save(backup_path +'dX.npy', dX)

        else:

            # np.save('/media/hpc-4_Raid/avakhitov/M.npy', M)
            # np.save('/media/hpc-4_Raid/avakhitov/N.npy', N)

            M = M + self.reg_coef * np.eye(M.shape[0])



            dX = np.linalg.solve(M, N)

            # f1 = time.time()

        f1 = time.time()
        # linsys_times_fout = open('/home/avakhitov/timelog_solver.txt', 'a')
        # linsys_times_fout.write('make ' + str(f0-s0)+' '+str(f1-f0) + ' ' + str(fs1-s1)+'\n')
        # linsys_times_fout.flush()

        return dX


    def calc_gmm_probs_dif(self, X, Yp, Yn, meansvec, covarsvec, weights):
        Xvec = np.reshape(X, np.prod(X.shape))
        # meansvec = np.reshape(means, np.prod(means.shape))
        # covarsvec = np.reshape(covars, np.prod(covars.shape))

        dX = self.solve_lin_sys_for_gmm(Xvec, meansvec, covarsvec, weights)

        df = self.gmmhist_df(meansvec, covarsvec, weights, Yp, Yn)

        df_vec = np.concatenate((df[0], df[1], df[2]))
        dXf = (df_vec).dot(dX[0:dX.shape[0]-1, :])
        # dXp = np.reshape(dXfp, X.shape)
        # dXn = np.reshape(dXfn, X.shape)

        return df[3], df[4], dXf




    def initialize_calc_ll_gmm_fun(self):
        Yvec = T.dvector('Y')
        meansvec = T.dvector('means')
        covarsvec = T.dvector('covars')
        weights = T.dvector('weights')
        lam = T.dscalar('lambda')
        ndim = meansvec.shape[0]/self.gm_num
        Y = T.reshape(Yvec, (Yvec.shape[0]/ndim, ndim))
        LL = self.calc_ll_gmm(Y, T.reshape(meansvec, (self.gm_num, meansvec.shape[0]/self.gm_num)),
                              T.reshape(covarsvec, (self.gm_num, meansvec.shape[0]/self.gm_num)),
                              weights)
        LL_lag = T.sum(LL)+lam*(T.sum(weights)-1)
        LL_sum = T.sum(LL)
        self.gmm_f = function([Yvec, meansvec, covarsvec, weights, lam], LL_lag)

        # ll0 = self.calc_ll_gmm_single(y, T.reshape(meansvec, (self.gm_num, meansvec.shape[0]/self.gm_num)),
        #                       T.reshape(covarsvec, (self.gm_num, meansvec.shape[0]/self.gm_num)),
        #                       weights)
        # self.gmm_f0 = function([y,  meansvec, covarsvec, weights], ll0)

        LLg = gradient.jacobian(LL_lag, [Yvec, meansvec, covarsvec, weights, lam])

        LL_sum_g = gradient.jacobian(LL_sum, [Yvec, meansvec, covarsvec, weights])

        self.gmm_df = function([Yvec, meansvec, covarsvec, weights], LL_sum_g)
        # llg = gradient.jacobian(LLg, [Yvec, meansvec, covarsvec, weights])

        # llhy = gradient.jacobian(LLg[0], [Yvec, meansvec, covarsvec, weights])
        llhm = gradient.jacobian(LLg[1], [Yvec, meansvec, covarsvec, weights])
        llhc = gradient.jacobian(LLg[2], [Yvec, meansvec, covarsvec, weights])
        llhw = gradient.jacobian(LLg[3], [Yvec, meansvec, covarsvec, weights, lam])

        self.gmm_df = function([Yvec, meansvec, covarsvec, weights], LL_sum_g)
        self.gmm_hm = function([Yvec, meansvec, covarsvec, weights, lam], llhm)
        self.gmm_hc = function([Yvec, meansvec, covarsvec, weights, lam], llhc)
        self.gmm_hw = function([Yvec, meansvec, covarsvec, weights, lam], llhw)

    def gmm_dif_by_X(self, X, Y, means, covars, weights):
        Xvec = np.reshape(X, np.prod(X.shape))
        meansvec = np.reshape(means, np.prod(means.shape))
        covarsvec = np.reshape(covars, np.prod(covars.shape))

        hm = self.gmm_hm(Xvec, meansvec, covarsvec, weights, 0)
        hc = self.gmm_hc(Xvec, meansvec, covarsvec, weights, 0)
        hw = self.gmm_hw(Xvec, meansvec, covarsvec, weights, 0)

        Yvec = np.reshape(Y, np.prod(Y.shape))
        df = self.gmm_df(Yvec, meansvec, covarsvec, weights)

        mean_row = np.concatenate((hm[1], hm[2], hm[3], np.zeros((len(meansvec), 1))), axis=1)
        cov_row = np.concatenate((hc[1], hc[2], hc[3], np.zeros((len(meansvec), 1))), axis=1)
        weight_row = np.concatenate((hw[1], hw[2], hw[3], np.reshape(hw[4], (self.gm_num, 1))), axis=1)
        lambda_row = np.concatenate(
            (np.zeros((1, len(meansvec))), np.zeros((1, len(meansvec))), np.reshape(hw[4], (1, self.gm_num)), np.zeros((1, 1))), axis=1)

        M = np.concatenate((mean_row, cov_row, weight_row, lambda_row))
        N = np.concatenate((-hm[0], -hc[0], -hw[0], np.zeros((1, hw[0].shape[1]))))
        dX = np.linalg.solve(M, N)
#correction for lagrangian
        df[3] = df[3] -np.linalg.norm(df[3])*np.ones_like(df[3])

        dl = np.ones((1))
        dl[0] = -dl[0]+np.sum(weights)

        df_vec = np.concatenate((df[1], df[2], df[3], dl))
        dXf = np.transpose(df_vec).dot(dX)

        dXf = np.reshape(dXf, X.shape)
        return dXf, df[0]



    def build_gmm(self, X, n_it = 1000, min_cov = 0.000001):
        gmm = mixture.GMM(covariance_type='diag', init_params='wmc', min_covar=min_cov,
                    n_components=self.gm_num, n_init=1, n_iter=n_it, params='wmc',
                    random_state=None)
        gmm.fit(X)


        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_), gmm.score(X)

    def build_adagmm(self, X, min_cov=0.000001):
        bics = []
        scores = []
        for cnum in range(1, 5):
            gmm = mixture.GMM(covariance_type='diag', init_params='wmc', min_covar=min_cov,
                        n_components=cnum, n_init=1, n_iter=1000, params='wmc',
                        random_state=None)
            #print X.shape
            gmm.fit(X)
            bics.append(gmm.bic(X))
            scores.append(np.sum(gmm.score(X)))

            # bics.append(gmm.aic(X))

        # f_log = open('/home/avakhitov/ada_log.txt', 'a')
        # score_log = open('/home/avakhitov/ada_score_log.txt', 'a')
        # best_score_log = open('/home/avakhitov/ada_best_score_log.txt', 'a')
        # for i in range(0, len(bics)):
        #     f_log.write(str(bics[i]) + ' ')
        #     score_log.write(str(scores[i]) + ' ')
        # f_log.write('\n')
        # score_log.write('\n')


        best_cnum = np.argmin(bics)+1
        # best_cnum = 1
        # for i in range(1, len(bics)):
        #     if (bics[i] < bics[best_cnum-1] - 0.05*np.abs(bics[best_cnum-1])):
        #         best_cnum = i+1
        #
        # best_score_log.write(str(best_cnum)+'\n')


        gmm = mixture.GMM(covariance_type='diag', init_params='wmc', min_covar=min_cov,
                          n_components=best_cnum, n_init=1, n_iter=1000, params='wmc',
                          random_state=None)
        gmm.fit(X)

        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_), gmm.score(X)


    def refine_gmm(self, X, means, covs, weights, min_covar = 0.000001, n_iter = 1000):
        gm_num = weights.shape[0]
        gmm = mixture.GMM(covariance_type='diag', init_params='', min_covar=min_covar,
                    n_components=gm_num, n_init=1, n_iter=n_iter, params='wmc',
                    random_state=None)
        gmm.means_ = means
        gmm.covars_ = covs
        gmm.weights_ = weights
        gmm.fit(X)
        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_)








    def build_gauss_model(self, X):
        mean = np.mean(X, 0)
        Xc = X - np.tile(mean, (X.shape[0], 1))
        covs = np.sum(Xc ** 2, 0) / Xc.shape[0]
        return mean, covs

    def build_gauss_model_theano(self, X):
        mean = T.mean(X, 0)
        Xc = X - T.tile(mean, (X.shape[0], 1))
        covs = T.sum(Xc ** 2, 0) / Xc.shape[0] + self.min_cov


        # scan_result, scan_updates = theano.scan(fn=lambda ii, minval, covs : T.max([covs[ii], minval]),
        #                                         outputs_info=None,
        #                                         sequences=T.arange(covs.shape[0]),
        #                                         non_sequences=[self.min_cov, covs])
        #
        # covs = scan_result

        return mean, covs

    def build_gauss_model_theano_batched(self, X):
        mean_all = T.mean(X, axis = 1).reshape((X.shape[0], X.shape[2]))
        var_all = T.var(X, axis = 1).reshape((X.shape[0], X.shape[2])) + self.min_cov
        return mean_all, var_all

    def calc_gauss_fun(self, Y, mean, covs):
        Yc = Y - np.tile(mean, (Y.shape[0], 1))
        exp_val = -np.sum(Yc**2/np.tile(covs, (Y.shape[0], 1)), 1)
        norm_scal = 1.0/np.sqrt(2*np.pi)/np.prod(np.sqrt(covs))
        return np.exp(exp_val)*norm_scal

    def calc_gauss_fun_theano(self, Y, mean, covs):
        Yc = T.tile(Y, (mean.shape[0],1,1)) - T.tile(mean, (Y.shape[0], 1, 1))
        exp_val = -T.sum(Yc**2/T.tile(covs, (Y.shape[0], 1, 1)),1)
        norm_scal = 1.0/T.sqrt(2*np.pi)/T.prod(T.sqrt(covs))
        return T.exp(exp_val)*norm_scal

    def calc_gauss_fun_theano_batched(self, X, mean_all, var_all):
        X_tiled = T.tile(X, (mean_all.shape[0], 1, 1))
        M_tiled = T.tile(mean_all.reshape((mean_all.shape[0], 1, mean_all.shape[1])), (1, X.shape[0], 1))
        C_tiled = T.tile(var_all.reshape((mean_all.shape[0], 1, mean_all.shape[1])), (1, X.shape[0], 1))
        Xc = X_tiled - M_tiled
        exp_val = -T.sum(Xc**2/C_tiled, 2)
        norm_scal = 1.0/T.sqrt(2*np.pi) * (T.prod(T.sqrt(var_all), axis = 1)) ** (-1)
        return T.exp(exp_val)*T.tile(norm_scal.reshape((-1,1)), (1, exp_val.shape[1]))

    def calc_log_gauss_fun_theano_batched(self, X, mean_all, var_all):
        X_tiled = T.tile(X, (mean_all.shape[0], 1, 1))
        M_tiled = T.tile(mean_all.reshape((mean_all.shape[0], 1, mean_all.shape[1])), (1, X.shape[0], 1))
        C_tiled = T.tile(var_all.reshape((mean_all.shape[0], 1, mean_all.shape[1])), (1, X.shape[0], 1))
        Xc = X_tiled - M_tiled
        exp_val = -0.5*T.sum(Xc**2/C_tiled, 2)
        norm_scal = -0.5 * T.log(2 * np.pi) * X.shape[1] - 0.5 * T.sum(T.log(var_all), 1)
        return exp_val + T.tile(norm_scal.reshape((-1, 1)), (1, exp_val.shape[1]))

    def calc_softmax_for_gauss(self, gauss_vals):
        return T.transpose(T.nnet.softmax(T.transpose(gauss_vals)))



    def init_batched_gauss_fun(self):
        X_train = T.dtensor3('X_train')
        M,C = self.build_gauss_model_theano_batched(X_train)
        X_test = T.dmatrix('X_test')
        G = self.calc_log_gauss_fun_theano_batched(X_test, M, C)
        S = self.calc_softmax_for_gauss(G)
        S = S + 1e-4
        Y = T.dmatrix('Y')
        loss = -T.sum(Y * T.log(S))
        self.f = function([X_train, X_test, Y], loss, allow_input_downcast=True)
        df = T.grad(loss, [X_train, X_test])
        self.df = function([X_train, X_test, Y], df, allow_input_downcast=True)

    def init_batched_cosine_fun(self):
        X_train = T.dtensor3('X_train')
        X_test = T.dmatrix('X_test')
        G = X_train.dot(X_test.transpose())
        G = G.reshape((-1, X_test.shape[0]))
        S = self.calc_softmax_for_gauss(G + 1e-4)
        S = S + 1e-4
        Y_train = T.dmatrix('Y_train')
        Y_test = T.dmatrix('Y_test')
        Y_pred_vec = Y_train.dot(S)
        loss = -T.sum(Y_test * T.log(Y_pred_vec + 1e-4))
        self.f = function([X_train, X_test, Y_train, Y_test], loss, allow_input_downcast=True)
        df = T.grad(loss, [X_train, X_test])
        self.df = function([X_train, X_test, Y_train, Y_test], df, allow_input_downcast=True)


    def calc_log_gauss_fun_theano(self, Y, mean, covs):
        n_samples, n_dim = Y.shape
        Yc = Y - T.tile(mean, (Y.shape[0], 1))
        exp_val = -0.5*T.sum(Yc**2/T.tile(covs, (Y.shape[0], 1)),1)
        norm_scal = -0.5*T.log(2*np.pi)*n_dim-0.5*T.sum(T.log(covs))
        #temp
        # return mean
        return exp_val+norm_scal


    def calc_log_gauss_fun(self, Y, mean, covs):
        n_samples, n_dim = Y.shape
        Yc = Y - np.tile(mean, (Y.shape[0], 1))
        exp_val = -0.5*np.sum(Yc**2/np.tile(covs, (Y.shape[0], 1)),1)
        norm_scal = -0.5*np.log(2*np.pi)*n_dim-0.5*np.sum(np.log(covs))
        return exp_val+norm_scal


    def calc_ll(self, Y, mean, covs):
        Yc = Y - np.tile(mean, (Y.shape[0], 1))
        NYc = np.divide(Yc ** 2, 2*np.tile(covs, (Y.shape[0], 1)))
        exp_val = -np.sum(NYc, 1)
        norm_coef = -0.5*np.log(2*np.pi) - 0.5*np.sum(np.log(covs))
        return exp_val + norm_coef

    def calc_binning(self, Y, mean, covs):
        p = self.calc_gauss_fun(Y, mean, covs)
        binw = 1.0/self.bin_num
        bin_vals = np.zeros(p.shape[0])
        for i in range(0, self.sample_num):
            bin_ind = int(np.floor(p[i] / binw))
            bin_vals[i] = bin_ind

        return bin_vals

    def calc_one_hist_val(self, ind, p,hmin, hmax):
        w = (hmax-hmin)/self.bin_num
        z1 = T.le(hmin+ind*w, p)
        z2 = T.le(p, hmin+(ind+1)*w)
        hvs = T.abs_((z1*z2).dot(p)/(hmin+(ind+0.5)*w))
        return hvs

    def calc_one_hist_val_test(self, ind, p,hmin, hmax):
        w = (hmax-hmin)/self.bin_num
        lb = hmin+ind*w
        rb = hmin+(ind+1.0)*w
        T.printing.Print(lb)
        T.printing.Print(rb)

        scan_result, scan_updates = theano.scan(fn=lambda ii, x,lb,rb : T.abs_(x[ii])*T.le(lb, x[ii])*T.le(x[ii],rb),
                                                outputs_info=None,
                                                sequences=T.arange(p.shape[0]),
                                                non_sequences=[p, lb, rb])

        return T.sum(scan_result) *2.0/ (T.abs_(lb+rb)+0.01)


    def calc_one_hist_val_triang(self, ind, p,hmin, hmax):
        w = (hmax-hmin)/self.bin_num+self.min_cov
        w_triang = 4 * w+self.min_cov
        lb = hmin+ind*w
        rb = hmin+(ind+1.0)*w
        cp = 0.5 * (lb + rb)
        fn = lambda ii, x, lb, rb, cp : ((-2.0/w_triang**2)*T.abs_(cp - x[ii]) + 1.0/w_triang)*T.le(x[ii]-w_triang/2, cp)*T.le(cp, x[ii]+w_triang/2)
        scan_result, scan_updates = theano.scan(fn=fn,
                                                outputs_info=None,
                                                sequences=T.arange(p.shape[0]),
                                                non_sequences=[p, lb, rb, cp])

        return T.sum(scan_result)

    def calc_one_hist_val_test_noth(self, ind, p, hmin, hmax):
        w = (hmax - hmin) / self.bin_num
        lb = hmin + ind * w
        rb = hmin + (ind + 1.0) * w

        vals = []
        for i in range(0, p.shape[0]):
            t1 = 0.0
            if (p[i] >= lb):
                t1 = 1.0
            t2 = 0.0
            if (p[i] <= rb):
                t2 = 1.0
            v = t1*t2*np.abs(p[i])
            vals.append(v)
        return np.sum(vals)*2.0/ (np.abs(lb+rb)+0.01)

    def calc_one_hist_val_triang_noth(self, ind, p, hmin, hmax):
        w = (hmax - hmin) / self.bin_num
        w_triang = 4 * w
        lb = hmin + ind * w
        rb = hmin + (ind + 1.0) * w
        cp = 0.5*(lb+rb)
        vals = []
        for i in range(0, p.shape[0]):
            t1 = 0.0
            # if (abs(p[i] >= lb):
            #     t1 = 1.0
            # t2 = 0.0
            # if (p[i] <= rb):
            #     t2 = 1.0
            t1 = 1.0
            t2 = 0.0
            if (np.abs(p[i] - cp) <= w_triang/2):
                t2 = 1.0
#triang function with center x_0 and width w: -2/w^2*abs(x-x_0)+1/w
            v = t1 * t2 * w_triang*( (-2.0/w_triang**2)*np.abs(cp - p[i]) + 1.0/w_triang)
            vals.append(v)
        return np.sum(vals)

    def calc_one_hist_val_noth(self, ind, p,hmin, hmax):
        w = (hmax-hmin)/self.bin_num
        z1 = np.less_equal(hmin+ind*w, p)
        z2 = np.less_equal(p, hmin+(ind+1)*w)
        hvs = (z1*z2).dot(p)/(hmin+(ind+0.5)*w)
        return hvs

    def calc_hist_vals_test_noth(self, p, hmin, hmax):
        hvs = []
        for i in range(0, self.bin_num):
            hvs.append(self.calc_one_hist_val_test_noth(i, p, hmin, hmax))

        return hvs


    def calc_hist_vals_triang_noth(self, p, hmin, hmax):
        hvs = []
        for i in range(0, self.bin_num):
            hvs.append(self.calc_one_hist_val_triang_noth(i, p, hmin, hmax))

        return hvs

    def calc_hist_vals_triang(self, p, hmin, hmax):

        scan_result, scan_updates = theano.scan(fn = self.calc_one_hist_val_triang,
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=[p, hmin, hmax])

        return scan_result

    def calc_one_hist_val_unrolled_epan(self, i, p, hmin, hmax):
        bin_id = T.mod(i, self.bin_num)
        sample_id = T.floor_div(i-bin_id, self.bin_num)
        w = (hmax-hmin)/self.bin_num+self.min_cov
        w_triang = 4 * w+self.min_cov
        lb = hmin+bin_id*w
        rb = hmin+(bin_id+1.0)*w
        cp = 0.5 * (lb + rb)
        return 3.0/4.0*(1.0 - ((cp - p[sample_id])*2.0/w_triang)**2)*T.le(p[sample_id]-w_triang/2, cp)*T.le(cp, p[sample_id]+w_triang/2)

    def calc_one_hist_val_unrolled_epan_noth(self, i, p, hmin, hmax):
        bin_id = np.mod(i, self.bin_num)
        sample_id = np.floor((i-bin_id)/self.bin_num)
        w = (hmax-hmin)/self.bin_num+self.min_cov
        w_triang = 4 * w+self.min_cov
        lb = hmin+bin_id*w
        rb = hmin+(bin_id+1.0)*w
        cp = 0.5 * (lb + rb)
        val = 0.0
        if (np.abs(p[sample_id]-cp) <= w_triang/2):
            val = 3.0/4.0*(1.0 - ((cp - p[sample_id])*2.0/w_triang)**2)
        return val


    def calc_one_hist_val_unrolled(self, i, p, hmin, hmax):
        bin_id = T.mod(i, self.bin_num)
        sample_id = T.floor_div(i-bin_id, self.bin_num)
        w = (hmax-hmin)/self.bin_num+self.min_cov
        w_triang = 4 * w+self.min_cov
        lb = hmin+bin_id*w
        rb = hmin+(bin_id+1.0)*w
        cp = 0.5 * (lb + rb)
        return w_triang*((-2.0/w_triang**2)*T.abs_(cp - p[sample_id]) + 1.0/w_triang)*T.le(p[sample_id]-w_triang/2, cp)*T.le(cp, p[sample_id]+w_triang/2)

    def calc_one_hist_val_unrolled_noth(self, i, p, hmin, hmax):
        bin_id = np.mod(i, self.bin_num)
        sample_id = np.floor((i-bin_id)/self.bin_num)
        w = (hmax-hmin)/self.bin_num+self.min_cov
        w_triang = 4 * w+self.min_cov
        lb = hmin+bin_id*w
        rb = hmin+(bin_id+1.0)*w
        cp = 0.5 * (lb + rb)
        val = 0.0
        if (np.abs(p[sample_id]-cp) <= w_triang/2):
            val = w_triang * ((-2.0 / w_triang ** 2) * np.abs(cp - p[sample_id]) + 1.0 / w_triang)
        return val


    def calc_hist_vals_unrolled(self, p, hmin, hmax):

        scan_result, scan_updates = theano.scan(fn = self.calc_one_hist_val_unrolled,
                    outputs_info=None,
                    sequences=T.arange(self.bin_num*self.sample_num),
                    non_sequences=[p, hmin, hmax])

        h_table = T.reshape(scan_result, (self.sample_num, self.bin_num))

        return T.sum(h_table, 0)

    def calc_hist_vals_unrolled_epan(self, p, hmin, hmax):

        scan_result, scan_updates = theano.scan(fn = self.calc_one_hist_val_unrolled_epan,
                    outputs_info=None,
                    sequences=T.arange(self.bin_num*self.sample_num),
                    non_sequences=[p, hmin, hmax])

        h_table = T.reshape(scan_result, (self.sample_num, self.bin_num))

        return T.sum(h_table, 0)


    def calc_hist_vals_unrolled_epan_noth(self, p, hmin, hmax):

        hvs = np.zeros(self.sample_num*self.bin_num)
        for i in range(0, self.sample_num*self.bin_num):

            hv = self.calc_one_hist_val_unrolled_epan_noth(i, p, hmin, hmax)
            hvs[i] = hv


        h_table = np.reshape(hvs, (self.sample_num, self.bin_num))

        return np.sum(h_table, 0)

    def calc_hist_vals_unrolled_noth(self, p, hmin, hmax):

        hvs = np.zeros(self.sample_num*self.bin_num)
        for i in range(0, self.sample_num*self.bin_num):

            hv = self.calc_one_hist_val_unrolled_noth(i, p, hmin, hmax)
            hvs[i] = hv


        h_table = np.reshape(hvs, (self.sample_num, self.bin_num))

        return np.sum(h_table, 0)

    def calc_hist_vals_vector_th(self, p, hmn, hmax):
        sample_num = p.shape[0]
        p_mat = T.tile(p.reshape((sample_num, 1)), (1, self.bin_num))
        w = (hmax - hmn) / self.bin_num + self.min_cov
        grid_vals = T.arange(0, self.bin_num)*(hmax-hmn)/self.bin_num+hmn+w/2.0
        grid = T.tile(grid_vals, (sample_num, 1))
        w_triang = 4 * w + self.min_cov
        D = T._tensor_py_operators.__abs__(grid-p_mat)
        mask = (D<=w_triang/2)
        D_fin = w_triang * (D*(-2.0 / w_triang ** 2) + 1.0 / w_triang)*mask
        hist_corr = T.sum(D_fin, 0)
        return hist_corr



    def calc_hist_vals_vector(self, p, hmn, hmax):
        p_mat = np.tile(p.reshape((self.sample_num, 1)), (1, self.bin_num))
        w = (hmax - hmn) / self.bin_num + self.min_cov
        grid_vals = np.arange(0, self.bin_num)*(hmax-hmn)/self.bin_num+hmn+w/2.0
        grid = np.tile(grid_vals, (self.sample_num, 1))
        w_triang = 4 * w + self.min_cov
        D = np.abs(grid-p_mat)
        mask = (D<=w_triang/2)
        D_fin = w_triang * (D*(-2.0 / w_triang ** 2) + 1.0 / w_triang)*mask
        hist_corr = np.sum(D_fin, 0)
        return hist_corr


    def set_hist_val(self, i, h, hi, hv):
        h = T.set_subtensor(h[hi[i]], hv[i])
        return h
        # h[hi[i]] = hv[i]


    def calc_hist_vals_fast(self, p, hmin, hmax):
        bw = (hmax-hmin)/self.bin_num

        w_triang = bw*4
        hinds, scan_updates = theano.scan(fn = lambda i, p : T.cast(T.floor_div(p[i]-hmin, bw), 'int32'),
                    outputs_info=None,
                    sequences=T.arange(p.shape[0]),
                    non_sequences=[p])


        hvals_inds, scan_updates = theano.scan(fn=lambda i, p, w_triang: ((-2.0/w_triang**2)*T.abs_((hinds[i]+0.5)*bw - p[i]) + 1.0/w_triang),
                                          outputs_info=None,
                                          sequences=T.arange(p.shape[0]),
                                          non_sequences=[p,w_triang])


        sr, scan_updates = theano.scan(
            fn=self.set_hist_val,
            outputs_info=T.zeros((self.bin_num)),
            sequences=T.arange(p.shape[0]),
            non_sequences=[hinds, hvals_inds])


        return sr[-1]






    def calc_hist_vals_test(self, p, hmin, hmax):

        scan_result, scan_updates = theano.scan(fn = self.calc_one_hist_val_test,
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=[p,hmin, hmax])

        return scan_result


    def calc_hist_vals(self, p, hmin, hmax):

        scan_result, scan_updates = theano.scan(fn = self.calc_one_hist_val,
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=[p,hmin, hmax])

        return scan_result

    def calc_hist_vals_noth(self, p, hmin, hmax):
        hvs = []
        for i in range(0, self.bin_num):
            hvs.append(self.calc_one_hist_val_noth(i, p, hmin, hmax))

        return hvs


    def calc_hist(self, Y, X):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Y, mean, covs)
        hp = self.calc_hist_vals(p_p)
        return hp

    def initialize_theano_hist_fun(self):
        X = T.dmatrix('X')
        Yp = T.dmatrix('Yp')
        hp = self.calc_hist(Yp, X)
        return function([X, Yp], hp)

    def initialize_theano_probs_fun(self):
        X = T.dmatrix('X')
        Yp = T.dmatrix('Yp')
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        return function([X, Yp], p_p)

    def initialize_BG_fun(self):
        X = T.dmatrix('X')
        mean, covs = self.build_gauss_model_theano(X)
        self.build_gauss_mean_fun = function([X], mean)
        mean_grad = gradient.jacobian(mean, X)
        self.build_gauss_mean_grad = function([X], mean_grad)
        self.build_gauss_cov_fun = function([X], covs)
        cov_grad = gradient.jacobian(covs, X)
        self.build_gauss_cov_grad = function([X], cov_grad)

    def initialize_gauss_ll_fun(self):
        m = T.dvector('m')
        c = T.dvector('c')
        Y = T.dmatrix('Y')
        p = self.calc_log_gauss_fun_theano(Y, m, c)
        self.ll_fun = function([Y,m,c], p)

    def initialize_gauss_ll_hist_fun(self):
        m = T.dvector('m')
        c = T.dvector('c')
        Y = T.dmatrix('Y')
        hmin = T.dscalar('hmin')
        hmax = T.dscalar('hmax')
        p = self.calc_log_gauss_fun_theano(Y, m, c)

        h = self.calc_hist_vals(p, hmin, hmax)
        self.ll_hist_fun = function([Y, m, c, hmin, hmax], h)
        dh = gradient.jacobian(h, [Y, m, c])
        self.ll_hist_grad = function([Y, m, c, hmin, hmax], dh)

    def initialize_hist_loss_fun(self):
        hp = T.dvector('hp')
        hn = T.dvector('hn')
        L = self.hist_loss(hn, hp)
        self.hist_loss_fun = function([hn, hp], L)
        dL = T.grad(L, [hn, hp])
        self.hist_loss_grad = function([hn, hp], dL)


    def calc_hist_loss_noth(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model(X)
        p_p = self.calc_log_gauss_fun(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun(Yn, mean, covs)

        hmax, hmin = self.calc_min_max_noth(p_n, p_p)

        hp = self.calc_hist_vals_noth(p_p, hmin, hmax)
        hn = self.calc_hist_vals_noth(p_n, hmin, hmax)
        L = self.hist_loss_noth(hn, hp)# L = ifelse(T.lt(0.01, hp[0]-hn[0]), 1.0, 0.0)

        return L, hp, hn

    def initialize_calc_min_max(self):
        pp = T.dvector('pp')
        pn = T.dvector('pn')
        [hmax, hmin] = self.calc_min_max(pn, pp)
        self.calc_mm_fun = function([pn, pp], [hmax, hmin])

    def make_selective_g_ll(self, means, covs, c, Y):
        fn = lambda ind, means, covs, c, Y : self.calc_log_gauss_fun_theano_single(Y[ind, :], means[c[ind], :], covs[c[ind], :])
        scan_result, scan_updates = theano.scan(fn = fn,
                    outputs_info=None,
                    sequences=T.arange(Y.shape[0]),
                    non_sequences=[ means, covs, c, Y])
        return scan_result

    def calc_hist_loss_gauss_selective(self, Yp, Yn, inds_p, inds_n, means, covs):

        p_p = self.make_selective_g_ll(means, covs, inds_p, Yp)
        p_n = self.make_selective_g_ll(means, covs, inds_n, Yn)

        L, hmax, hmin, hn, hp = self.calc_hist_loss(p_n, p_p)

        return L, hp, hn, hmax, hmin

    def calc_hist_loss_gauss_mean_cov(self, Yp, Yn):
        p_p = self.calc_log_gauss_fun_theano_1(Yp)
        p_n = self.calc_log_gauss_fun_theano_1(Yn)

        L, hmax, hmin, hn, hp = self.calc_hist_loss(p_n, p_p)

        return L, hp, hn, hmax, hmin

    def calc_hist_loss_gauss_fast(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_fast(p_n, p_p)

        #regularize covariances
        scan_result, scan_updates = theano.scan(fn=lambda ii, covs : 1000*T.max([0, 0.01-covs[ii]]),
                                                outputs_info=None,
                                                sequences=T.arange(covs.shape[0]),
                                                non_sequences=[covs])
        L += T.sum(scan_result)

        return L, hp, hn, hmax, hmin

    def calc_hist_loss_gauss_mm(self, X, Yp, Yn, hmin, hmax):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_mm(p_n, p_p, hmin, hmax)

        #regularize covariances
        # scan_result, scan_updates = theano.scan(fn=lambda ii, covs : 1000*T.max([0, 0.01-covs[ii]]),
        #                                         outputs_info=None,
        #                                         sequences=T.arange(covs.shape[0]),
        #                                         non_sequences=[covs])
        # L += T.sum(scan_result)

        return L, hp, hn, hmax, hmin


    def calc_hist_loss_gauss(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        L, hmax, hmin, hn, hp = self.calc_hist_loss(p_n, p_p)

        #regularize covariances
        # scan_result, scan_updates = theano.scan(fn=lambda ii, covs : 1000*T.max([0, 0.01-covs[ii]]),
        #                                         outputs_info=None,
        #                                         sequences=T.arange(covs.shape[0]),
        #                                         non_sequences=[covs])
        # L += T.sum(scan_result)

        return L, hp, hn, hmax, hmin

    def calc_hist_loss_gauss_vector(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(p_n, p_p)

        #regularize covariances
        # scan_result, scan_updates = theano.scan(fn=lambda ii, covs : 1000*T.max([0, 0.01-covs[ii]]),
        #                                         outputs_info=None,
        #                                         sequences=T.arange(covs.shape[0]),
        #                                         non_sequences=[covs])
        # L += T.sum(scan_result)

        return L, hp, hn, hmax, hmin


    def calc_hist_loss_gauss_epan(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_epan(p_n, p_p)
        return L, hp, hn, hmax, hmin

    def calc_hist_loss_gauss_symm(self, X, Yp, Yn):
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)

        hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        # hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hp = self.calc_hist_vals_unrolled(p_p, hmin, hmax)
        # hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        hn = self.calc_hist_vals_unrolled(p_n, hmin, hmax)

        L = self.hist_loss_symm(hn, hp)

        return L, hp, hn, hmax, hmin

    def calc_hist_loss_mm(self, p_n, p_p, hmin, hmax):
        # hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        # hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hp = self.calc_hist_vals_unrolled(p_p, hmin, hmax)
        # hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        hn = self.calc_hist_vals_unrolled(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp


    def calc_hist_loss_epan(self, p_n, p_p):
        hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        # hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hp = self.calc_hist_vals_unrolled_epan(p_p, hmin, hmax)
        # hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        hn = self.calc_hist_vals_unrolled_epan(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp



    def calc_hist_loss(self, p_n, p_p):
        hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        # hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hp = self.calc_hist_vals_unrolled(p_p, hmin, hmax)
        # hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        hn = self.calc_hist_vals_unrolled(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp

    def calc_hist_loss_vector(self, p_n, p_p):
        hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        # hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hp = self.calc_hist_vals_vector_th(p_p, hmin, hmax)
        # hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        hn = self.calc_hist_vals_vector_th(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp

    def calc_hist_loss_fast(self, p_n, p_p):
        hmax, hmin = self.calc_min_max(p_n, p_p)
        hmin -= self.min_cov
        hmax += self.min_cov
        hp = self.calc_hist_vals_fast(p_p, hmin, hmax)
        hn = self.calc_hist_vals_fast(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp

    def calc_hist_loss_triang(self, p_n, p_p):
        hmax, hmin = self.calc_min_max(p_n, p_p)
        hp = self.calc_hist_vals_triang(p_p, hmin, hmax)
        hn = self.calc_hist_vals_triang(p_n, hmin, hmax)
        L = self.hist_loss(hn, hp)
        return L, hmax, hmin, hn, hp


    def select_by_inds(self, p_p_m, comp_inds_p):
        scan_result, scan_updates = theano.scan(fn=lambda ind, A, c : A[ind, c[ind]],
                                                outputs_info=None,
                                                sequences=T.arange(p_p_m.shape[0]),
                                                non_sequences=[p_p_m, comp_inds_p])
        return scan_result

    def select_by_inds_noth(self, p_p_m, comp_inds_p):
        seld = []
        for ind in range(0, p_p_m.shape[0]):
            seld.append(p_p_m[ind, comp_inds_p[ind]])

        return seld

    def calc_hist_loss_from_probs(self, p_p_v, p_n_v, hmin, hmax):
        # hmax, hmin = self.calc_min_max(p_n_v, p_p_v)
        hp = self.calc_hist_vals_test(p_p_v, hmin, hmax)
        hn = self.calc_hist_vals_test(p_n_v, hmin, hmax)
        return self.hist_loss(hn, hp)


    def calc_hist_by_comp_loss(self, p_n, p_p, r_n, r_p):
            comp_inds_p = T.argmax(r_p, axis=0)
            comp_inds_n = T.argmax(r_n, axis=0)

            p_p_v = self.select_by_inds(p_p, comp_inds_p)
            p_n_v = self.select_by_inds(p_n, comp_inds_n)
            # p_p_v = p_p[:, comp_inds_p]
            # p_n_v = p_n[:, comp_inds_n]
            hmax, hmin = self.calc_min_max(p_n_v, p_p_v)
            hp = self.calc_hist_vals_test(p_p_v, hmin, hmax)
            hn = self.calc_hist_vals_test(p_n_v, hmin, hmax)
            L = self.hist_loss(hn, hp)  # L = ifelse(T
            # L = T.sum(p_p)+T.sum(p_n)
            return L, hmax, hmin, hn, hp

    def calc_hist_by_comp_loss_noth(self, p_n, p_p, r_n, r_p):
        comp_inds_p = np.argmax(r_p, axis=1)
        comp_inds_n = np.argmax(r_n, axis=1)
        p_p_v = self.select_by_inds_noth(p_p, comp_inds_p)
        p_n_v = self.select_by_inds_noth(p_n, comp_inds_n)
        hmax, hmin = self.calc_min_max_noth(p_n_v, p_p_v)
        hp = self.calc_hist_vals_noth(p_p_v, hmin, hmax)
        hn = self.calc_hist_vals_noth(p_n_v, hmin, hmax)
        L = self.hist_loss_noth(hn, hp)  # L = ifelse(T
        # L = T.sum(p_p)+T.sum(p_n)
        return L, hmax, hmin, hn, hp



    def calc_min_max(self, p_n, p_p):
        hminn = T.min(p_n)
        hmaxn = T.max(p_n)
        hminp = T.min(p_p)
        hmaxp = T.max(p_p)
        hmin = ifelse(T.lt(hminp,hminn), hminp, hminn)
        hmax = ifelse(T.lt(hmaxp, hmaxn), hmaxn, hmaxp)
        return hmax, hmin

    def calc_min_max_noth(self, p_n, p_p):
        hminn = np.min(p_n)
        hmaxn = np.max(p_n)
        hminp = np.min(p_p)
        hmaxp = np.max(p_p)
        hmin = hminn
        if (hminp < hminn):
            hmin = hminp
        hmax = hmaxp
        if (hmaxp < hmaxn):
            hmax = hmaxn
        return hmax, hmin

    def calc_min_max_robust_noth(self, p_n, p_p):
        hminn = np.percentile(p_n, 10)
        hmaxn = np.percentile(p_n, 90)
        hminp = np.percentile(p_p, 10)
        hmaxp = np.percentile(p_p, 90)
        hmin = hminn
        if (hminp < hminn):
            hmin = hminp
        hmax = hmaxp
        if (hmaxp < hmaxn):
            hmax = hmaxn
        return hmax, hmin

    def hist_loss_symm(self, hn, hp):
        scan_result, scan_updates = theano.scan(fn = lambda ind, A: T.sum(A[0:ind]),
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=hp)
        agg_p = scan_result

        scan_result, scan_updates = theano.scan(fn = lambda ind, A: T.sum(A[0:ind]),
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=hn)
        agg_n = scan_result

        L1 = T.sum(T.dot(agg_p, hn))
        L2 = T.dot(agg_n, hp)
        L =  L1+L2

        return L


    def hist_loss(self, hn, hp):
        scan_result, scan_updates = theano.scan(fn = lambda ind, A: T.sum(A[0:ind+1]),
                    outputs_info=None,
                    sequences=T.arange(self.bin_num),
                    non_sequences=hp)

        # for i in range(1, self.bin_num):
        #     agg_p.append(agg_p[i - 1] + hp[i])
        agg_p = scan_result

        L = T.sum(T.dot(agg_p, hn))

        return L

    def hist_loss_noth(self, hn, hp):
        ap = np.zeros(self.bin_num)
        for i in range(0, self.bin_num):
            ap[i] = np.sum(hp[0:i+1])

        L = np.sum(np.dot(ap, hn))

        return L

    def initialize_theano_fun_mc(self):
        Yp = T.dmatrix('Yp')
        Yn = T.dmatrix('Yn')
        L,hp, hn, hmax, hmin  = self.calc_hist_loss_gauss_mean_cov(Yp, Yn)
        self.f = function([Yp, Yn], [L, hp, hn])
        gf = T.grad(L, [Yp, Yn])
        self.df = function([Yp, Yn], gf)

    def initialize_theano_unrolled_hist(self):
        p = T.fvector('p')
        hmin = T.min(p)
        hmax = T.max(p)
        h = self.calc_hist_vals_triang(p, hmin, hmax)

        self.fh = function([p], h)

        h2 = self.calc_hist_vals_unrolled(p, hmin, hmax)
        self.fh2 = function([p], h2)

        return

    def initialize_gloss_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        mean, covs = self.build_gauss_model_theano(X)
        p_p = self.calc_log_gauss_fun_theano(Yp, mean, covs)
        p_n = self.calc_log_gauss_fun_theano(Yn, mean, covs)
        L = T.sum(p_p)-T.sum(p_n)

        self.gauss_f = function([X, Yp, Yn], L)
        gf = T.grad(L, [X, Yp, Yn])
        self.dgauss_f = function([X, Yp, Yn], gf)

        return

    def initialize_theano_fun_epan(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        # L,hp, hn, hmax, hmin  = self.calc_hist_loss_gauss(X, Yp, Yn)
        hmin = T.fscalar('hmin')
        hmax = T.fscalar('hmax')
        L, hp, hn, hmax, hmin = self.calc_hist_loss_gauss_epan(X, Yp, Yn)
        self.f = function([X, Yp, Yn], [L, hp, hn, hmax, hmin], allow_input_downcast=True)#self.sample_size

        gf = T.grad(L, [X, Yp, Yn])
        self.df = function([X, Yp, Yn], gf,allow_input_downcast=True)#, profile=True

        return






    def initialize_theano_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        # L,hp, hn, hmax, hmin  = self.calc_hist_loss_gauss(X, Yp, Yn)
        hmin = T.fscalar('hmin')
        hmax = T.fscalar('hmax')
        # L, hp, hn, hmax, hmin = self.calc_hist_loss_gauss(X, Yp, Yn)
        L, hp, hn, hmax, hmin = self.calc_hist_loss_gauss_vector(X, Yp, Yn)
        self.f = function([X, Yp, Yn], [L, hp, hn, hmax, hmin], allow_input_downcast=True)#self.sample_size

        # Lv, hpv, hnv, hmaxv, hminv = self.calc_hist_loss_gauss_vector(X, Yp, Yn)
        # self.fvec = function([X, Yp, Yn], [Lv, hpv, hnv, hmaxv, hminv], allow_input_downcast=True)#self.sample_size


        Lsymm, hpsymm, hnsym, hmaxsymm, hminsymm = self.calc_hist_loss_gauss_symm(X, Yp, Yn)
        self.fsymm = function([X, Yp, Yn], [Lsymm, hpsymm, hnsym, hmaxsymm, hminsymm], allow_input_downcast=True)  # self.sample_size

        gf = T.grad(L, [X, Yp, Yn])
        self.df = function([X, Yp, Yn], gf,allow_input_downcast=True)#, profile=True

        # gfvec = T.grad(Lv, [X, Yp, Yn])
        # self.dfvec = function([X, Yp, Yn], gfvec, allow_input_downcast=True)  # , profile=True

        gf_symm = T.grad(Lsymm, [X, Yp, Yn])
        self.dfsymm = function([X, Yp, Yn], gf_symm, allow_input_downcast=True)  # , profile=True

        p = T.fvector('p')
        h = self.calc_hist_vals_unrolled(p, hmin, hmax)
        self.bh = function([p, hmin, hmax], h,allow_input_downcast=True)

        hn = T.fvector('hn')
        hp = T.fvector('hp')
        L1 = self.hist_loss(hn, hp)

        mean, covs = self.build_gauss_model_theano(X)
        self.hl = function([hn, hp], L1,allow_input_downcast=True)
        self.bgm = function([X], [mean, covs],allow_input_downcast=True)

        mean1 = T.fvector('mean')
        covs1 = T.fvector('covs')

        p_p1 = self.calc_log_gauss_fun_theano(Yp, mean1, covs1)
        self.bgp = function([Yp, mean1, covs1], p_p1,allow_input_downcast=True)

        return

        # h_p = T.fvector('h_p')
        # h_n = T.fvector('h_n')
        #
        # hl = self.hist_loss(h_n, h_p)
        # p = T.fvector('p')
        # hmin1 = T.fscalar('hmin1')
        # hmax1 = T.fscalar('hmax1')
        #
        # self.bh = self.calc_hist_vals(p, hmin1, hmax1)
        #
        # mean1 = T.fvector('mean1')
        # covs1 = T.fvector('covs1')
        # p_p = self.calc_log_gauss_fun_theano(Yp, mean1, covs1)
        #
        # self.bp = function([Yp, mean1, covs1], p_p)


    def initialize_theano_fun_fast(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        L,hp, hn, hmax, hmin  = self.calc_hist_loss_gauss_fast(X, Yp, Yn)
        self.ff = function([X, Yp, Yn], [L, hp, hn, hmax, hmin])
        gf = T.grad(L, [X, Yp, Yn])
        self.dff = function([X, Yp, Yn], gf)

        p = T.fvector('p')
        hmin1 = T.fscalar('hmin1')
        hmax1 = T.fscalar('hmax1')

        self.bhf = self.calc_hist_vals_fast(p,hmin1,hmax1)

    def compute_fun(self, X, Yp, Yn):
        # [mean, covs] = self.build_gauss_model(X)


        # self.bin_pos = shared(np.zeros(X.shape[0]))
        # self.bin_pos = self.calc_binning(Yp, mean, covs)
        # self.bin_neg = shared(np.zeros(X.shape[0]))
        # self.bin_neg = self.calc_binning(Yn, mean, covs)

        self.initialize_theano_fun()
        return self.f(X, Yp, Yn)


    def calc_nn_ll_normalized_descs(self, X, Y):
        all_dists = Y.dot(T.transpose(X))
        d = T.max(all_dists, 1)
        return d

    def calc_mean_ll_descs(self, X, Y):
        meanvec = T.mean(X, 0)
        D = -T.sqrt(T.sum((Y - T.tile(meanvec, (Y.shape[0], 1)))**2, 1))
        return D


    def calc_nn_hist_loss(self, X, Yp, Yn):
        pp = self.calc_nn_ll_normalized_descs(X, Yp)
        pn = self.calc_nn_ll_normalized_descs(X, Yn)
        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(pn, pp)
        return L, hmax, hmin, hn, hp

    def calc_mean_hist_loss(self, X, Yp, Yn):
        pp = self.calc_mean_ll_descs(X, Yp)
        pn = self.calc_mean_ll_descs(X, Yn)
        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(pn, pp)
        return L, hmax, hmin, hn, hp


    def initialize_nn_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        L, hmax, hmin, hn, hp = self.calc_nn_hist_loss(X, Yp, Yn)
        self.f_nn = function([X, Yp, Yn], L)

        dL = T.grad(L, [X, Yp, Yn])
        self.df_nn = function([X, Yp, Yn], dL)

    def initialize_mean_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        L, hmax, hmin, hn, hp = self.calc_mean_hist_loss(X, Yp, Yn)
        self.f_mean = function([X, Yp, Yn], L)

        dL = T.grad(L, [X, Yp, Yn])
        self.df_mean = function([X, Yp, Yn], dL)

    def easy_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def test_compute_cosine_batched(self, X_train, X_test, Y_train, Y_test):

        G = X_train.dot(X_test.transpose())
        G = G.reshape(-1, X_test.shape[0])
        S = np.zeros_like(G)

        for dind in range(0, G.shape[1]):
            S[:, dind] = self.easy_softmax(G[:, dind].reshape(-1))

        S = S + 1e-4

        Y_pred_vec = Y_train.dot(S)

        labels_pred = np.argmax(Y_pred_vec, axis=0).reshape(-1)
        labels_true = np.argmax(Y_test, axis=0).reshape(-1)
        err_num = len(np.nonzero(np.abs(labels_pred - labels_true))[0])
        acc = 1.0 - err_num / (Y_test.shape[1] + 0.0)

        return -np.sum(Y_test * np.log(Y_pred_vec)), acc











