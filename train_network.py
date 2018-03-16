import numpy as np
import cv2
from loader import TieLoader,data_generator
import theano
import theano.tensor as T
from utils import get_network_str,save_weights,load_weights
import lasagne
from lasagne import layers as L
from lasagne.nonlinearities import rectify,tanh
from lasagne.regularization import regularize_network_params,l2
from gmm_op import get_gmm,calc_log_prob_gmm,calc_log_prob_gmm_componetwise
from theano_utils import split,histogram_loss
from sklearn.metrics import average_precision_score
import sys

from easydict import EasyDict
cfg = EasyDict()
cfg.SEQ_LENGTH = 250
cfg.TILE_SIZE = 16
cfg.OUT_SIZE = 1
cfg.gm_num = 4

def make_conv(data,num_filters,name,nonl = rectify):
    return L.Conv2DLayer(data,filter_size=(3,3),num_filters=num_filters,
                        nonlinearity=nonl,pad='same',
                        name='conv'+str(name))

def make_deconv(data,num_filters,name):
    return  L.Deconv2DLayer(data,12,2,2,crop='valid',name='deconv'+str(name),nonlinearity=rectify)
    
def baseline(data,ndim,verbose=True,model_name=''):
    assert(ndim==3)
    datal = res = L.InputLayer((None,3,None,None)
                           ,data
                           ,name='data')
    if(model_name != ''):
        load_weights(res,model_name,)
    print get_network_str(res,incomings=True,outgoings=True)
    sym = lasagne.layers.get_output(res)
    sym = T.transpose(sym,(0,2,3,1))/256.
    params = lasagne.layers.get_all_params(res, trainable=True)
    l2_loss = 1e-4 * regularize_network_params(res, l2)
    return res,sym,params,l2_loss,dict()

def FCN(data,ndim,verbose=True,model_name=''):
    datal = res = L.InputLayer((None,3,None,None)
                           ,data
                           ,name='data')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=6,
                        nonlinearity=rectify,pad='same',
                        name='conv1')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=12,
                        nonlinearity=rectify,pad='same',
                        name='conv2')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=ndim,
                        nonlinearity=None,pad='same',
                        name='conv3')
    add_params = dict()
    if(model_name != ''):
        add_params =  load_weights(res,model_name)
        
    print get_network_str(res,incomings=True,outgoings=True)
    sym = lasagne.layers.get_output(res)
    sym = T.transpose(sym,(0,2,3,1))
    sym = sym/(T.sqrt(T.square(sym).sum(-1)+1e-8)[:,:,:,None])
    params = lasagne.layers.get_all_params(res, trainable=True)
    l2_loss = 1e-4 * regularize_network_params(res, l2)
    return res,sym,params,l2_loss,add_params

def FCN_color(data,ndim,verbose=True,model_name=''):
    datal = res = L.InputLayer((None,3,None,None)
                           ,data
                           ,name='data')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=6,
                        nonlinearity=rectify,pad='same',
                        name='conv1')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=12,
                        nonlinearity=rectify,pad='same',
                        name='conv2')
    res = L.Conv2DLayer(res,filter_size=(3,3),num_filters=ndim-3,
                        nonlinearity=None,pad='same',
                        name='conv3')
    if(model_name != ''):
        add_params =  load_weights(res,model_name)
    else:
        add_params = {'feature_matrix' : theano.shared(np.random.rand(ndim,ndim).astype(np.float32))}
        
    print get_network_str(res,incomings=True,outgoings=True)
    sym = lasagne.layers.get_output(res)
    sym = T.transpose(sym,(0,2,3,1))
    
    data = T.transpose(data,(0,2,3,1))/256.
    sym = sym/(T.sqrt(T.square(sym).sum(-1)+1e-8)[:,:,:,None])
    sym = T.concatenate([sym,data],-1)
    feature_matrix = add_params['feature_matrix']
    sym = T.dot(sym,feature_matrix/(T.sqrt((T.sum(T.square(feature_matrix),0)+1e-5))[None,:]))
    params = lasagne.layers.get_all_params(res, trainable=True)
    params= params+[i for i in add_params.values()]
    l2_loss = 1e-4 * regularize_network_params(res, l2)
    return res,sym,params,l2_loss,add_params
               
        
def soft_predict_sym(features,means,covars,weights):
    return 1.-T.nnet.sigmoid(calc_log_prob_gmm(features,means,covars,weights))

def get_output(X,t=cfg.TILE_SIZE,o=cfg.OUT_SIZE):
    return X[:,(t-o)//2:(t+o)//2,(t-o)//2:(t+o)//2,:]

def make_train_fn(net,X,data,label,params,l2_loss,ndim,gm_num,model_name=''):
    non_learn_params={'min_cov' : theano.shared(1e-3),
                      'p_p_weight' : theano.shared(0.),
                      'p_n_weight' : theano.shared(0.),
                      'lr' : theano.shared(np.array(1e-1, dtype=theano.config.floatX))}
    
    def split_tr_p_n(x,y):
        x_tr_p,x_n = split(x,y)
        x_p = x_tr_p[x_tr_p.shape[0]//2:]
        x_tr = x_tr_p[:x_tr_p.shape[0]//2]
        return x_tr,x_p,x_n
    
    X = get_output(X)
    X = X.reshape((-1,X.shape[-1]))
    x_tr,x_p,x_n = split_tr_p_n(X,label.flatten())    
    m,c,w = get_gmm(x_tr,gm_num,ndim,use_approx_grad=True)
    
    p_n = calc_log_prob_gmm(x_n,m,c,w)
    p_p = calc_log_prob_gmm(x_p,m,c,w)
    
    loss = histogram_loss(p_n,p_p,non_learn_params['min_cov'],70)[0]+\
            l2_loss-\
            non_learn_params['p_p_weight']*T.mean(T.clip(p_p,-100,40))+\
            non_learn_params['p_n_weight']*T.mean(T.clip(p_n,-40,100))
            
    prediction = T.nnet.sigmoid(T.concatenate([p_p,p_n],axis=0))
    
    Y = T.concatenate([T.ones_like(p_p),T.zeros_like(p_n)],axis=0)
    
    updates = lasagne.updates.adam(loss,params,non_learn_params['lr'])
    train_fn = theano.function([data, label], [loss,X,Y,prediction,m,c,w],\
                               allow_input_downcast=True, updates=updates)
    print 'train_fn compiled'
    test_fn = theano.function([data, label], [loss,X,Y,prediction,m,c,w,p_p,p_n],\
                               allow_input_downcast=True)
    print 'test_fn compiled'
    return train_fn,test_fn,non_learn_params,net

def get_pp_pn(l,pred):
    o = np.ones(len(l))
    pn = []
    pp = []
    R = pred[::max(len(pred)//100,1)].copy()
    R  = np.sort(R)
    for r in R:
        pn.append(o[(pred > r)&(l>0.9)].sum()/o[l>0.9].sum())
        pp.append(o[(pred < r)&(l<0.1)].sum()/o[l<0.1].sum())
    pp,pn = np.array(pp),np.array(pn)
    return (np.abs(pn[1:]-pn[:-1])*(pp[1:]+pp[:-1])/2.).sum()

def iterate_batches(fn,data_generator,epoch,metrix = dict()):
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
    if(i%1000 == 0 ):
        print '\r epoch %i batch %i loss=%.2f l=%.2f %s     '%\
        (epoch,i,loss/float(i+1),res[0],s)
        sys.stdout.flush()

        
    print '\r epoch %i batch %i loss=%.2f l=%.2f %s     '%\
    (epoch,i,loss/float(i+1),res[0],s)
    sys.stdout.flush()

metrix = { 'aps' : average_precision_score,
           'pp'  : lambda l,pred : np.ones(len(l))[(pred < 0.5)&(l<0.1)].sum()/np.ones(len(l))[l<0.1].sum(),
           'pn'  : lambda l,pred : np.ones(len(l))[(pred >= .5)&(l>0.9)].sum()/np.ones(len(l))[l>0.9].sum(),
           'int_pp_pn' : lambda l,pred : get_pp_pn(l,pred)}

def update_params(epoch,params):
    if(epoch == 1):
        params['min_cov'].set_value(1e-8)
        params['p_n_weight'].set_value(1e1)
        params['lr'].set_value(1e-2)
    if(epoch == 2):
        params['lr'].set_value(1e-3)
    
def fit(name,
        net,
        add_params,
        train_fn,test_fn,
        train_loader,test_loader,
        non_learn_params,
        epochs=6,train_esize=1500,test_esize=1500,
        metrix = metrix,update_params = update_params):
    for j in range(0,epochs):
        update_params(j,non_learn_params)
        print('train')
        sys.stdout.flush()
        iterate_batches(train_fn,\
                        lambda : data_generator(train_loader,epoch_size=train_esize,shuffle=True),
                       j,metrix)
        save_weights(net,'models/%s%03d'%(name,j),add_params)
        print('test')
        sys.stdout.flush()
        iterate_batches(test_fn,\
                        lambda : data_generator(test_loader,epoch_size=test_esize,shuffle=True),
                       j,metrix)
        
        
        
def run_train(name,net_generator,ndim,train_dir,test_dir,epochs=2,model_name=''):
    np.random.seed(10)
    train_loader = TieLoader(train_dir,0.2,0.3,t_size=32,mask_size=cfg.OUT_SIZE,sample_size=cfg.TILE_SIZE)
    test_loader = TieLoader(test_dir,0.2,0.3,t_size=32,mask_size=cfg.OUT_SIZE,sample_size=cfg.TILE_SIZE)
    data = T.tensor4(name='data')
    label = T.tensor3(name='label')
    net,X,params,l2_loss,add_params = net_generator(data,ndim=ndim,model_name=model_name)
    train_fn,test_fn,non_learn_params,net=make_train_fn(net,X,data,label,params,l2_loss,ndim=ndim,gm_num=4)
    fit(name,net,add_params,train_fn,test_fn,train_loader,test_loader,non_learn_params,epochs=epochs)
    
#print('net half')
#sys.stdout.flush()
#run_train('convh_net_no_bn_ndim6',FCN,6,'half_ties','test/half_ties')

#print('#############################################')
#print('net ties')
#sys.stdout.flush()
#run_train('conv2_net_no_bn_ndim6',FCN,6,'train_ties1','test_ties')

#print('baseline half')
#run_train('baseline',baseline,3,'half_ties','test/half_ties',epochs=1)

print('#############################################')
print('baseline ties')
run_train('baseline',baseline,3,'train_ties1','test_ties',epochs=1)

#print('#############################################')
#print('color net half')
#run_train('convh_net_color_ndim6',FCN_color,9,'half_ties','test/half_ties',epochs=2)

#print('#############################################')
#print('color net ties')
#run_train('conv2_color_ndim6',FCN_color,9,'train_ties1','test_ties',epochs=2)
