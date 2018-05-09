import time
import numpy as np
from loader import data_generator
from utils import get_network_str,save_weights,load_weights
import sys
import theano
import theano.tensor as T
from gmm_op import get_gmm,calc_log_prob_gmm
from theano_utils import split,histogram_loss,split_tr_p_n
from lasagne import layers as L
import lasagne
import os

def make_simple_gmm_classifier(X,label,non_learn_params):
    X = X.reshape((-1,X.shape[-1]))
    x_tr,x_p,x_n = split_tr_p_n(X,label.flatten())
    m,c,w = get_gmm(x_tr,
                    gm_num = non_learn_params['gm_num'],
                    ndims = non_learn_params['ndim'],
                    use_approx_grad=non_learn_params['use_approx_grad'])
    p_n = calc_log_prob_gmm(x_n,m,c,w)
    p_p = calc_log_prob_gmm(x_p,m,c,w)
    loss = histogram_loss(T.max(p_n).reshape((1,)),p_p,
                          non_learn_params['min_cov'],
                          non_learn_params['histogram_bins'],
                          non_learn_params['width'])[0]
    prediction = T.nnet.sigmoid(T.concatenate([p_p,p_n],axis=0))
    Y = T.concatenate([T.ones_like(p_p),T.zeros_like(p_n)],axis=0)
    return loss,X,Y,prediction,m,c,w,p_p,p_n

def make_train(net,data,label,non_learn_params,make_classifier):
    sym = L.get_output(net ,deterministic=False)
    s = int((L.get_output_shape(net)[1]-1)/2)
    sym = sym[:,s:s+1,s:s+1,:]
    loss,X,Y,prediction,m,c,w,p_p,p_n = make_classifier(sym,label,non_learn_params)
    params = L.get_all_params(net,trainable=True)
    grads = T.grad(loss,params)
    if( 'total_grad_constraint' in non_learn_params.keys()):
        grads = lasagne.updates.total_norm_constraint(grads,non_learn_params['total_grad_constraint'])
    updates = lasagne.updates.adam(grads,params,non_learn_params['lr'])
    return theano.function([data, label], [loss,X,Y,prediction,m,c,w,p_n,p_p],\
                               allow_input_downcast=True, updates=updates)

def iterate_batches(fn,data_generator,epoch,metrix = dict(),logger=None):
    loss=0
    acc=0
    labels,predicted = np.array([]),np.array([])
    for i,batch in enumerate(data_generator()):
        res = fn(*batch)
        mask = (res[2]>0.9) | (res[2]<0.1)
        loss+=res[0]
        labels = np.concatenate((labels,res[2][mask]))
        predicted = np.concatenate((predicted,res[3][mask]))
    s = ' '.join(['%s=%.3f'%(k,metrix[k](labels,predicted)) for k in metrix.keys() ])
    logger.log('epoch %i batch %i loss=%.2f l=%.2f %s\n'%(epoch,i,loss/float(i+1),res[0],s))