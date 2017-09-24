import theano
import theano.tensor as T
import lasagne
from lasagne import layers as L
from broadcast import BroadcastLayer,UnbroadcastLayer
from lasagne.nonlinearities import rectify



def softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def nonl_name(nonl):
    if(nonl == rectify):
        return 'relu'
    elif(nonl == softmax):
        return 'softmax'
    return str(nonl)
    
def make_conv(data,num_filters,filter_size=1,nonl=rectify,no_bias=False,name='no_name'):
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

def gen_unet(data,num_filters,deep,name='unet'):    
    name = name+str(deep)
    res1 = res_unit(data,num_filters,name=name+'_resnet1',hid=num_filters)   
    
    if(deep == 1):
        return res1
    
    res2 = L.Pool2DLayer(res1,2,name=name+'_pool')
    res2 = gen_unet(res2,num_filters*2,deep-1,name)
    res2 = L.Upscale2DLayer(res2,2,name=name+'_upscale')
    res1 = L.Conv2DLayer(res1,filter_size=1,num_filters=num_filters*2,nonlinearity=None,
                         name=name+'_conv(1)')
    res = L.ElemwiseSumLayer([res1,res2],name=name+'_sum')
    res = L.NonlinearityLayer(res,rectify,name=name+'_relu')
    res = res_unit(res,num_filters,name=name+'_resnet2',hid=num_filters)
    return res