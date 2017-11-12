import theano
from theano import gradient,function
import numpy as np
import theano.tensor as T
from sklearn import mixture
from theano.gof import Variable
from theano.gradient import grad
from theano.gradient import format_as

def jacobian(expression, wrt, consider_constant=None,
             disconnected_inputs='raise'):
    '''
    similar implementation as in theano.gradient, but ignore not empty updates 
    (because when you use it in lasagna there is should be some update and it is ok)
    '''
    from theano.tensor import arange
    # Check inputs have the right format
    assert isinstance(expression, Variable), \
        "tensor.jacobian expects a Variable as `expression`"
    assert expression.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expression`. If not use flatten to make it a vector")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return format_as(using_list, using_tuple,
                         grad(expression,
                              wrt,
                              consider_constant=consider_constant,
                              disconnected_inputs=disconnected_inputs))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = grad(expr[idx],
                        inp,
                        consider_constant=consider_constant,
                        disconnected_inputs=disconnected_inputs)
            rvals.append(rval)
        return rvals
    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = theano.scan(inner_function,
                                  sequences=arange(expression.shape[0]),
                                  non_sequences=[expression] + wrt)
#the only difference from theano implementation -- no assertion for updates
#     assert not updates, \
#         ("Scan has returned a list of updates. This should not "
#          "happen! Report this to theano-users (also include the "
#          "script that generated the error)")
    return format_as(using_list, using_tuple, jacobs)

def calc_log_prob_gmm_componetwise(Y,means,covars,weights = None):
    
    n_samples, n_dim = Y.shape
    lpr = -0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)[None,:]
                  + T.sum(T.square(means[None,:,:]-Y[:,None,:]) / covars[None,:,:], 2))
    if not (weights is None):
        lpr = lpr + T.log(weights)[None,:]
    return lpr


def calc_log_prob_gmm(Y,means,covars,weights = None):
    """
    calc probability of gmm/gauss vector
    Y: matrix n_samples x n_dim
    means: martix gm_num x n_dim
    covars: matrix gm_num x n_dim
    weights: vector gm_num
    out: vector n_samples
    """   
    lpr = calc_log_prob_gmm_componetwise(Y,means,covars,weights)
    lpr = T.transpose(lpr, (1,0))
    vmax = T.max(lpr,axis=0)
    out = T.log(T.sum(T.exp(lpr- vmax), axis=0))
    out += vmax
    return out

class GMM(mixture.GaussianMixture):
    '''
    similar as scipy.mixture.GaussianMixture but if fit calls with the same X as in previous call, it uses previous parameters
    '''
    def __init__(self,gm_num):
        super(GMM,self).__init__(covariance_type='diag',
                                           n_components=gm_num,
                                           max_iter=2000,
                                           warm_start=True)
        self.X = None
        
    def fit(self,X):
        if(not self.X is None):
            if(self.X.shape == X.shape):
                if(np.abs(self.X-X).mean()<1e-10):
                    return self;
        self.X = X.copy()
        super(GMM,self).fit(X)
        return self
        
        
   
class GMMOp(theano.Op):
    def __init__(self,gm_num,ndim,gmm=None,use_approx_grad=False):
        """
        fit gmm with diagonal covariances to input vectors
        input: vector[n_samples*n_dim] flatten
        output: vector means[gm_num*n_dim],covars[gm_num*n_dim],weights[gm_num], flatten
        """
        super(GMMOp, self).__init__()
        self.otypes = [T.fvector]
        self.itypes = [T.fvector]
        self.gm_num = gm_num
        self.ndim = ndim
        if(gmm is None):
            self.gmm =GMM(self.gm_num)
        else:
            self.gmm = gmm
        self.reg_coef = 1e-15
        self.use_approx_grad = use_approx_grad

    def perform(self, node, (X,), output_storage):       
        self.gmm.fit(X.reshape((-1,self.ndim)))
        means = self.gmm.means_.flatten()
        covars = self.gmm.covariances_.flatten()
        weights = self.gmm.weights_.flatten()
        output_storage[0][0] = np.concatenate((means,covars,weights)).astype(np.float32)
            
    def split_params(self,mcwl_vec):
        gm_num,n_dim = self.gm_num,self.ndim
        means = mcwl_vec[:n_dim*gm_num].reshape((gm_num,n_dim))
        covars = mcwl_vec[n_dim*gm_num:2*n_dim*gm_num].reshape((gm_num,n_dim))
        weights = mcwl_vec[2*n_dim*gm_num:2*n_dim*gm_num+gm_num].reshape((gm_num,))
        lam = mcwl_vec[-1]
        return means,covars,weights,lam     
    
    @staticmethod
    def build_lagrangian(Y,means,covars,weights,lam):
        log_prob = calc_log_prob_gmm(Y, means,covars,weights)        
        return T.sum(log_prob) + lam * (T.sum(weights) - np.float32(1.)) 
        
    def build_linear_system(self,Yvec,mcwl_vec):
        means,covars,weights,lam = self.split_params(mcwl_vec)
        lagrangian = self.build_lagrangian(Yvec.reshape((-1,self.ndim)),means,covars,weights,lam)
        d_mcwl = jacobian(lagrangian, [mcwl_vec],consider_constant=[Yvec,mcwl_vec])[0]
        dmcwl_dY = jacobian(d_mcwl, [Yvec],consider_constant=[Yvec,mcwl_vec])[0]
        d2mcwl = jacobian(d_mcwl, [mcwl_vec],consider_constant=[Yvec,mcwl_vec])[0]
        #d2mcwl grad(mcwl)^T = -dY_dmcwl
        return -dmcwl_dY,d2mcwl
    
    def solve_diag_linear(self,N,a,b,c,D):#MX=N
        '''
          |A B 0|
        M=|B C 0|
          |0 0 D|
        '''
        n_dim = self.ndim
        gm_num = self.gm_num
        n_samples = N.shape[1]//n_dim
        a = a + T.ones_like(a)*self.reg_coef
        c = c + T.ones_like(c) * self.reg_coef
        D = D + T.eye(D.shape[0]) * self.reg_coef
        e = 1. / (a - b / c * b)
        f = -e * b / c
        h = (T.ones_like(a) - f * b) / c
        
        e = e.reshape((gm_num,n_dim))
        h = h.reshape((gm_num,n_dim))
        f = f.reshape((gm_num,n_dim))
        
        eye = T.eye(n_dim)            
        mu = N[:gm_num*n_dim].reshape((gm_num,n_dim,n_samples,n_dim))*eye[None,:,None,:]
        cov = N[gm_num*n_dim:2*gm_num*n_dim].reshape((gm_num,n_dim,n_samples,n_dim))*eye[None,:,None,:]
        dX1 = e[:,None,None,:] * mu + f[:,None,None,:] * cov
        dX1 = dX1.reshape((-1,n_samples*n_dim))
        dX2 = f[:,None,None,:] * mu + h[:,None,None,:] * cov        
        dX2 = dX2.reshape((-1,n_samples*n_dim))        
        Di = T.nlinalg.matrix_inverse(D)
        dX3 = Di.dot(N[n_dim * 2 * gm_num:, :])
        
        dX = T.concatenate((dX1,dX2,dX3),axis=0)
        return dX
        
    def solve_general_linear(self,N,M):#MX=N
        M = M + self.reg_coef * T.eye(M.shape[0])
        Mi = T.nlinalg.matrix_inverse(M)        
        return Mi.dot(N)
    
    def solve_linear_system(self,N,M):
        '''
          |A B 0|
        M=|B C 0|
          |0 0 D|
        '''
        par_dim = self.ndim*self.gm_num
        def diag(M):            
            a = T.diag(M)
            A = T.diag(a)
            return a,abs(A - M).sum()
        a,na = diag(M[0:par_dim, 0:par_dim])
        b,nb = diag(M[0:par_dim, par_dim:2 * par_dim])
        c,nc = diag(M[par_dim:2 * par_dim, par_dim:2 * par_dim])
        D = M[2 * par_dim:, 2 * par_dim:]
        return theano.ifelse.ifelse(T.le(na+nb+nc,1e-10),\
                                 self.solve_diag_linear(N,a,b,c,D),\
                                 self.solve_general_linear(N,M))
    
    
    def approx_grad(self,Xvec,mcw):
        X = Xvec.reshape((-1,self.ndim))    
        means,covars,weights,_ = self.split_params(mcw)
        log_prob = calc_log_prob_gmm_componetwise(X,means,covars,weights)
        w = T.nnet.softmax(log_prob)
        s_w = T.sum(w,0)
        w_means = T.sum(w[:,:,None]*X[:,None,:],0)/(s_w[:,None]+0.0001)
        w_covars = T.sum(w[:,:,None]*((w_means[None,:,:]-X[:,None,:])**2),0)/(s_w[:,None]+0.0001)
        w_mcw = T.concatenate((w_means.flatten(),w_covars.flatten(),weights))
        return jacobian(w_mcw,[Xvec],consider_constant=[mcw,Xvec,w,s_w])[0]
    
    
    def grad(self, (Yvec,), output_grads):
        Yvec = gradient.disconnected_grad(Yvec)
        mcw_vec = GMMOp(self.gm_num,self.ndim,self.gmm)(Yvec)
        if(self.use_approx_grad):
            return  [output_grads[0].dot(self.approx_grad(Yvec,mcw_vec))]
        else:
            lam = Yvec.shape[0]//self.ndim
            mcwl_vec = T.concatenate((mcw_vec,lam.reshape((1,))))
            N,M = self.build_linear_system(Yvec,mcwl_vec)
            dX = self.solve_linear_system(N,M)
            return [output_grads[0].dot(gradient.disconnected_grad(dX[0:dX.shape[0]-1, :]))]

def get_gmm(X,gm_num,ndims,use_approx_grad=False):
    if(gm_num == 1):
        means = T.mean(X,0).reshape((1,-1))
        covars = (T.std(X,0)**2).reshape((1,-1))
        weights = T.ones(1)
    else:
        f = GMMOp(gm_num,ndims,use_approx_grad=use_approx_grad)(X.flatten())
        means = f[:gm_num*ndims].reshape((gm_num,ndims))
        covars = f[gm_num*ndims:2*gm_num*ndims].reshape((gm_num,ndims))
        weights = f[2*gm_num*ndims:]
    return means,covars,weights