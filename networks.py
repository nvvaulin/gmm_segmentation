import numpy as np
import cv2
from loader import TieLoader,data_generator
import lasagne
from lasagne import layers as L
from lasagne.nonlinearities import rectify,tanh
from utils import NormedDense,L2NormLayer
from utils import get_network_str,save_weights,load_weights,tee


def conv(data,num_filters,name,pad):
    return L.Conv2DLayer(data,filter_size=(3,3),num_filters=num_filters,
                        nonlinearity=None,pad=pad,
                        name='conv_'+name) 

def conv_nonl(data,num_filters,name,pad,use_bn=True):
    res = conv(data,num_filters,name,pad=pad)
    if(use_bn):
        res = L.BatchNormLayer(res,name='bn_'+name)
    res = L.NonlinearityLayer(res,rectify,name='relu_'+name) 
    return res

def baseline_norm(data,ndim,pad='same'):
    assert(ndim==4)
    res = L.DimshuffleLayer(data,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    return res

def baseline(data,ndim,pad='same'):
    assert(ndim==3)
    res = L.DimshuffleLayer(data,(0,2,3,1),name='transpose')
    return res

def conv4_net(data,ndim,pad='same'):
    res = conv_nonl(data,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = conv(res,ndim-1,'4',pad=pad)
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    return res

def conv4_net_dense(data,ndim,pad='same'):
    res = conv_nonl(data,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = conv(res,ndim-1,'4',pad=pad)
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    res = NormedDense(res,ndim,name='normed_dense')
    return res

def conv4_net_dense_color(data,ndim,pad='same'):
    res = conv_nonl(data,6,'1',pad= pad)
    res = conv_nonl(res,12,'2',pad=pad)
    res = conv_nonl(res,24,'3',pad=pad)
    res = L.concat([data,res],axis=1,name='concat')
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    res = NormedDense(res,ndim,name='normed_dense')
    return res

def UNet(data,ndim,pad='same'):
    res = uNet.build(data)
    res = L.Conv2DLayer(res, ndim-1, 1, nonlinearity=None)
    res = L.DimshuffleLayer(res,(0,2,3,1),name='transpose')
    res = L2NormLayer(res,1e-8,name='l2norm')
    return res       

def make_FCN(FCN_name,data,ndim,model_name='',input_shape = (None,3,None,None),pad='same',logger=None):
    if (isinstance(FCN_name,str)):
        tee('load finction '+FCN_name,logger)
        try:
            FCN = globals()[FCN_name] 
        except:
            raise NotImplementedError("No such function "+FCN_name)
    else:        
        tee('load finction '+FCN_name.__name__,logger)
        FCN=FCN_name
    datal = res = L.InputLayer(input_shape,
                               data/256.-0.5,
                               name='data')
    res = FCN(datal,ndim=ndim,pad=pad)
    if(model_name!=''):
        tee('load model '+model_name,logger)
        load_weights(res,model_name)
    tee(get_network_str(res,incomings=True,outgoings=True),logger)
    return res
