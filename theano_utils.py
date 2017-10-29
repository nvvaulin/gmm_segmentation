import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

def calc_log_prob_gauss_vector(Y,means,covars,weights = None):
    """
    calc probability of gmm/gauss vector
    Y: matrix n_samples x n_dim
    means: martix gm_num x n_dim
    covars: matrix gm_num x n_dim
    weights: vector gm_num
    out: vector n_samples
    """
    n_samples, n_dim = Y.shape
    if(weights is None):
        lpr = (-0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)
                      + T.sum((means ** 2) / covars, 1)
                      - 2 * T.dot(Y, (means / covars).T)
                      + T.dot(Y ** 2, T.transpose(1.0 / covars))))
    else:
        lpr = (-0.5 * (n_dim * T.log(2 * np.pi) + T.sum(T.log(covars), 1)
                      + T.sum((means ** 2) / covars, 1)
                      - 2 * T.dot(Y, (means / covars).T)
                      + T.dot(Y ** 2, T.transpose(1.0 / covars))) + T.log(weights))
    lpr = T.transpose(lpr, (1,0))
    vmax = T.max(lpr,axis=0)
    out = T.log(T.sum(T.exp(lpr- vmax), axis=0))
    out += vmax
    return out

def histogram_loss(p_n, p_p,min_cov,bin_num): 
    '''
    p_n -- negative values (theano vector)
    p_p -- positive values (theano vector)
    min_cov -- python float
    bin_num -- python int
    '''
    def calc_hist_vals_vector_th(p, hmn, hmax):
        sample_num = p.shape[0]
        p_mat = T.tile(p.reshape((sample_num, 1)), (1, bin_num))
        w = (hmax - hmn) / bin_num + min_cov
        grid_vals = T.arange(0, bin_num)*(hmax-hmn)/bin_num+hmn+w/2.0
        grid = T.tile(grid_vals, (sample_num, 1))
        w_triang = 4 * w + min_cov
        D = T._tensor_py_operators.__abs__(grid-p_mat)
        mask = (D<=w_triang/2)
        D_fin = w_triang * (D*(-2.0 / w_triang ** 2) + 1.0 / w_triang)*mask
        hist_corr = T.sum(D_fin, 0)
        return hist_corr
    
    def hist_loss(hn, hp):
        scan_result, scan_updates = theano.scan(fn = lambda ind, A: T.sum(A[0:ind+1]),
                    outputs_info=None,
                    sequences=T.arange(bin_num),
                    non_sequences=hp)
        agg_p = scan_result
        L = T.sum(T.dot(agg_p, hn))
        return L
    
    def calc_min_max(p_n, p_p):
        hminn = T.min(p_n)
        hmaxn = T.max(p_n)
        hminp = T.min(p_p)
        hmaxp = T.max(p_p)
        hmin = ifelse(T.lt(hminp,hminn), hminp, hminn)
        hmax = ifelse(T.lt(hmaxp, hmaxn), hmaxn, hmaxp)
        return hmax, hmin
    
    hmax, hmin = calc_min_max(p_n, p_p)
    hmin -= min_cov
    hmax += min_cov
    hp = calc_hist_vals_vector_th(p_p, hmin, hmax)
    hn = calc_hist_vals_vector_th(p_n, hmin, hmax)
    L = hist_loss(hn, hp)
    return L, hmax, hmin, hn, hp


def split(X,Y,size):
    """
    Y: vector mask, 0 -- positive
    size: size of output
    outputs: X[Y==0][:size],X[Y==0][size:2*size],X[Y==1][:size]
    """
    tr = X[(Y<0.1).nonzero()][:size]
    p = X[(Y<0.1).nonzero()][size:2*size]
    n = X[(Y> 0.9).nonzero()][:size]
    return tr,p,n
    
    
def center_loss(_X,_Y):
    """
    Y: mask , 0 -- positive
    """
    _X = _X/(0.001+T.sqrt(T.sqr(_X).sum(1))[:,None])
    tX = _X[:64*24*24//2]
    tY = 1.-_Y[:64*24*24//2]
    Y = 1.- _Y[64*24*24//2:]
    X = _X[64*24*24//2:]
    center = (tX*tY[:,None]).sum(0)/(0.001+tY.sum())
    Y = 2.*Y-1.
    return (T.sqr(center[None,:]-X)*Y[:,None]).mean()

def center_accuracy(X,Y):
    """
    Y: mask , 0 -- positive
    """
    Tr = X[:len(X)//2][Y[:len(X)//2] < 0.1]
    Xp = X[len(X)//2:][Y[len(X)//2:] < 0.1]
    Xn = X[len(X)//2:][Y[len(X)//2:] > 0.9]
    c = Tr.mean(0)
    d = np.std(Tr,axis=0).sum()
    tp = Xp[np.square(Xp-c).sum(1) < d**2]
    tn = Xn[np.square(Xn-c).sum(1) > d**2]
    return (len(tp)+len(tn))/float(len(Xp)+len(Xn)),d