import theano
import theano.tensor as T
import lasagne
from lasagne import layers as L
from broadcast import BroadcastLayer,UnbroadcastLayer
from lasagne.nonlinearities import rectify
from lasagne.init import HeNormal,Constant



def softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def nonl_name(nonl):
    if(nonl == rectify):
        return 'relu'
    elif(nonl == softmax):
        return 'softmax'
    return str(nonl)
    
def make_conv(data,num_filters,filter_size=1,nonl=rectify,no_bias=False,name='no_name',with_batchnorm=False):
    if(with_batchnorm):
        if(no_bias):
            res = L.Conv2DLayer(data,filter_size=filter_size,num_filters=num_filters,
                                nonlinearity=None,b=None,pad='same',
                                name=name+'_conv(no_bias,'+str(filter_size)+')')
        else:
            res = L.Conv2DLayer(data,filter_size=filter_size,num_filters=num_filters,
                                nonlinearity=None,pad='same',
                                name=name+'_conv('+str(filter_size)+')')
        res = L.BatchNormLayer(res,name=name+'_bn')
        if not (nonl is None):
            res = L.NonlinearityLayer(res,nonl,name=name+'_'+nonl_name(nonl))
        return res
    else:
        if(no_bias):
            res = L.Conv2DLayer(data,filter_size=filter_size,num_filters=num_filters,
                                nonlinearity=nonl,b=None,pad='same',
                                name=name+'_conv(no_bias,'+str(filter_size)+')_nonl')
        else:
            res = L.Conv2DLayer(data,filter_size=filter_size,num_filters=num_filters,
                                nonlinearity=nonl,pad='same',
                                name=name+'_conv('+str(filter_size)+')_nonl')
        return res

def res_unit(data,num_filters,nonl = rectify,hid =None,name=''):
    if(hid is None):
        hid = num_filters
    res = make_conv(data,num_filters=hid,filter_size=3,no_bias=False,name=name+'_1')
    res = make_conv(res,num_filters=num_filters,filter_size=3,no_bias=False,name=name+'_2')
    res = make_conv(res,num_filters=num_filters,filter_size=3,no_bias=False,nonl=None,name=name+'_3')
    resid = L.Conv2DLayer(data,filter_size=1,num_filters=num_filters,nonlinearity=None,
                          name=name+'_resid')
    res = L.ElemwiseSumLayer([res,resid],name=name+'_sum')
    res = L.NonlinearityLayer(res,nonl,name=name+'_'+nonl_name(nonl))
    return res

def make_deconv(data,num_filters,filter_size=2,nonl=rectify,name='no_name',with_batchnorm=False):
    if(with_batchnorm):
        res = L.Deconv2DLayer(data,num_filters,filter_size,2,crop='valid',nonlinearity=None,name=name+'_deconv')
        res = L.BatchNormLayer(res,name=name+'_deconv_bn')
        if not(nonl is  None):
            res = L.NonlinearityLayer(res,nonl,name=name+'_nonl')
        return res
    else:
        return L.Deconv2DLayer(data,num_filters,filter_size,2,crop='valid',nonlinearity=nonl,name=name+'_deconv_nonl')

    
    
def gen_unet(data,num_filters,deep,name='unet',first=True,out_ndim=None):    
    name = name+str(deep)
    res1 = make_conv(data,num_filters,3,name=name+'_in',with_batchnorm=True)   
    
    if(deep == 1):
        return res1
    
    res2 = L.Pool2DLayer(res1,2,name=name+'_pool')
    res2 = gen_unet(res2,num_filters*2,deep-1,name[:-1],False)
    res2 = make_deconv(res2,num_filters*((2**(deep-1))-1),name=name,with_batchnorm=True)
    res = L.ConcatLayer([res2,res1],axis=1, cropping=(None, None, "center", "center"),name=name+'_concat')
    if(first ):
        res = L.Conv2DLayer(res,out_ndim,3,nonlinearity=None,name=name+'_conv',pad='same')
    else:
        res = make_conv(res,num_filters*(2**(deep-1)),3,name=name+'_out',with_batchnorm=True)   
    return res


def make_net(input_tensor,ndim,use_score=False):
    data_l = L.InputLayer((None,3,None,None)
                           ,input_tensor
                           ,name='data')
    features = gen_unet(data_l,6,3,name='unet',out_ndim=ndim)
    if(not use_score):
        return features
    general_dist = make_conv(features,12,name='general_dist_hid')
    general_dist = L.Conv2DLayer(general_dist,1,(1,1),pad='same',name='general_dist')
    net = L.concat([general_dist,features])
    return net
    