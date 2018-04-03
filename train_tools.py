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


def make_classifier(X,label,non_learn_params,gm_num,ndim):
    X = X.reshape((-1,X.shape[-1]))
    x_tr,x_p,x_n = split_tr_p_n(X,label.flatten())
    m,c,w = get_gmm(x_tr,gm_num,ndim,use_approx_grad=True)
    p_n = calc_log_prob_gmm(x_n,m,c,w)
    p_p = calc_log_prob_gmm(x_p,m,c,w)
    loss = histogram_loss(p_n,p_p,
                          non_learn_params['min_cov'],
                          100,
                          non_learn_params['width'])[0]
    prediction = T.nnet.sigmoid(T.concatenate([p_p,p_n],axis=0))
    Y = T.concatenate([T.ones_like(p_p),T.zeros_like(p_n)],axis=0)
    return loss,X,Y,prediction,m,c,w,p_p,p_n

def make_train(net,data,label,non_learn_params,gm_num,ndim):
    sym = L.get_output(net ,deterministic=False)
    s = int((L.get_output_shape(net)[1]-1)/2)
    sym = sym[:,s:s+1,s:s+1,:]
    loss,X,Y,prediction,m,c,w,p_p,p_n = make_classifier(sym,label,non_learn_params,gm_num,ndim)
    params = L.get_all_params(net,trainable=True)
    updates = lasagne.updates.adam(loss,params,non_learn_params['lr'])
    return theano.function([data, label], [loss,X,Y,prediction,m,c,w],\
                               allow_input_downcast=True, updates=updates)


def make_test(net,data,label,non_learn_params,gm_num,ndim):
    sym = L.get_output(net ,deterministic=True)
    s = int((L.get_output_shape(net)[1]-1)/2)
    sym = sym[:,s:s+1,s:s+1]
    loss,X,Y,prediction,m,c,w,p_p,p_n = make_classifier(sym,label,non_learn_params,gm_num,ndim)
    return theano.function([data, label], [loss,X,Y,prediction,m,c,w],\
                               allow_input_downcast=True)
        



def iterate_batches(fn,data_generator,epoch,metrix = dict(),ohem = None,logger=None):
    loss=0
    acc=0
    labels,predicted = np.array([]),np.array([])
    ohem_cur = 0
    for i,batch in enumerate(data_generator()):
        res = fn(*batch)
        mask = (res[2]>0.9) | (res[2]<0.1)
        loss+=res[0]
        labels = np.concatenate((labels,res[2][mask]))
        predicted = np.concatenate((predicted,res[3][mask]))
        if(not (ohem is None)):
            if(res[0] > 100):
                if(len(ohem) > 1000):
                        ohem[ohem_cur] = batch
                        ohem_cur = (ohem_cur+1)%len(ohem)
                else:
                    ohem.append(batch)
            if(len(ohem) > 0):
                for i in range(3):
                    batch = np.random.choice(ohem)
                    res = fn(*batch)
                    mask = (res[2]>0.9) | (res[2]<0.1)
                    loss+=res[0]
                    labels = np.concatenate((labels,res[2][mask]))
                    predicted = np.concatenate((predicted,res[3][mask]))
                    
    s = ' '.join(['%s=%.3f'%(k,metrix[k](labels,predicted)) for k in metrix.keys() ])
    print '\r epoch %i batch %i loss=%.2f l=%.2f %s     '%(epoch,i,loss/float(i+1),res[0],s)
    sys.stdout.flush()
    if (not (logger is  None)):
        logger.write('epoch %i batch %i loss=%.2f l=%.2f %s\n'%(epoch,i,loss/float(i+1),res[0],s))
        logger.flush()

def fit(name,
        net,
        train_fn,test_fn,
        train_loader,test_loader,
        non_learn_params,
        logger,
        epochs,train_esize,test_esize,
        metrix,update_params,use_ohem=False,
        start_epoch = 0):
    if(use_ohem):
        ohem = []
    else:
        ohem = None
    save_weights(net,'models/%s%03d'%(name,0))
    for j in range(start_epoch,epochs):
        update_params(j,non_learn_params)
        print('train')
        logger.write('train')
        logger.flush()
        sys.stdout.flush()
        np.random.seed(int(time.time()))
        iterate_batches(train_fn,\
                        lambda : data_generator(train_loader,epoch_size=train_esize,shuffle=True),
                       j,metrix,logger = logger,ohem=ohem)
        save_weights(net,'models/%s%03d'%(name,j))
        print('test')
        logger.write('test')
        logger.flush()
        sys.stdout.flush()
        np.random.seed(0)
        iterate_batches(test_fn,\
                        lambda : data_generator(test_loader,epoch_size=test_esize,shuffle=True),
                       j,metrix,logger = logger)
    print('final test')