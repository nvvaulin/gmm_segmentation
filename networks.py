import numpy as np
import cv2
from loader import TieLoader,data_generator
import lasagne
from lasagne import layers as L
from lasagne.nonlinearities import rectify,tanh
from utils import NormedDense,L2NormLayer
from utils import get_network_str,save_weights,load_weights



def conv(data,num_filters,name,pad):
    return L.Conv2DLayer(data,filter_size=(3,3),num_filters=num_filters,
                        nonlinearity=None,pad=pad,
                        name='conv_'+name) 

def conv_nonl(data,num_filters,name,pad):
    res = conv(data,num_filters,name,pad=pad)
    res = L.BatchNormLayer(res,name='bn_'+name,alpha=0.05)
    res = L.NonlinearityLayer(res,rectify,name='relu_'+name) 
    return res

def convNet(datal,pad='same'):
    res = conv_nonl(datal,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = conv(res,cfg.ndim-1,'4',pad=pad)
    return res

def baseline_norm(data,ndim,verbose=True,model_name='',input_shape = (None,3,None,None),pad='same'):
    assert(ndim==4)
    print 'baseline_norm'
    datal = res = L.InputLayer(input_shape
                           ,data/256.-0.5
                           ,name='data')
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    if(model_name!=''):
        load_weights(res,'models/'+model_name)
    print get_network_str(res,incomings=True,outgoings=True)
    return res


def baseline(data,ndim,verbose=True,model_name='',input_shape = (None,3,None,None),pad='same'):
    assert(ndim==3)
    print('baseline')
    datal = res = L.InputLayer(input_shape
                           ,data/256.-0.5
                           ,name='data')
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    if(model_name!=''):
        load_weights(res,'models/'+model_name)
    print get_network_str(res,incomings=True,outgoings=True)
    return res



def conv4_net(data,ndim,verbose=True,model_name='',input_shape = (None,3,None,None),pad='same'):
    print('conv4_net')
    datal = res = L.InputLayer(input_shape
                           ,data/256.-0.5
                           ,name='data')
    res = conv_nonl(datal,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = conv(res,ndim-1,'4',pad=pad)
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    if(model_name!=''):
        load_weights(res,'models/'+model_name)
    print get_network_str(res,incomings=True,outgoings=True)
    return res



def conv4_net_dense(data,ndim,verbose=True,model_name='',input_shape = (None,3,None,None),pad='same'):
    print('conv4_net_dense')
    datal = res = L.InputLayer(input_shape
                           ,data/256.-0.5
                           ,name='data')
    res = conv_nonl(datal,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = conv(res,ndim-1,'4',pad=pad)
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    res = NormedDense(res,name='normed_dense')
    if(model_name!=''):
        load_weights(res,'models/'+model_name)
    print get_network_str(res,incomings=True,outgoings=True)
    return res




def UNet(data,ndim,verbose=True,model_name='',input_shape = (None,3,None,None),pad='same',logger=None):
    datal = res = L.InputLayer(input_shape
                           ,data
                           ,name='data')
    res = uNet.build(datal)
    res = L.Conv2DLayer(res, ndim-1, 1, nonlinearity=None)
    res.W.name = 'outW'
    res.b.name = 'outb'
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    if(model_name!=''):
        print 'load model '+'models/'+model_name
        load_weights(res,'models/'+model_name)
    print get_network_str(res,incomings=True,outgoings=True)
    if(not (logger is None)):        
        logger.write(get_network_str(res,incomings=True,outgoings=True)+'\n')
        logger.flush()
    return res

