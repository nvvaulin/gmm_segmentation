import numpy as np
from loader import TieLoader
from networks import baseline_norm,conv4_net,conv4_net_dense
from train_tools import make_train,make_test
from networks import baseline_norm,conv4_net,conv4_net_dense,conv4_net_dense_color
import theano
import theano.tensor as T
from sklearn.metrics import average_precision_score
import datetime
from utils import get_pp_pn
import os
from train_tools import fit
from test_tools import test_network


def train_network(FCN,cfg,logger_name='out.txt'):

    train_loader = TieLoader('data/ties32',
                             0.2,0.3,t_size=32,mask_size=cfg.OUT_SIZE,sample_size=cfg.TILE_SIZE,cache_samples=True)
    test_loader = TieLoader('data/test_ties32',
                            -0.1,0.45,t_size=32,mask_size=cfg.OUT_SIZE,sample_size=cfg.TILE_SIZE,cache_samples=True)

    data = T.tensor4(name='data')
    label = T.tensor3(name='label')
    net = FCN(data,ndim=cfg.ndim,model_name='%s%03d'%(cfg.NAME,cfg.TRAIN.EPOCH) if cfg.TRAIN.EPOCH > 0 else '',
              input_shape = (None,3,cfg.TILE_SIZE,cfg.TILE_SIZE),
              pad = 'same')

    non_learn_params={'min_cov' : theano.shared(1e-3),
                      'lr' : theano.shared(np.array(1e-2, dtype=theano.config.floatX)),
                      'width': theano.shared(4.)}

    train_fn = make_train(net,data,label,non_learn_params,cfg.gm_num,cfg.ndim)
    print 'train_fn compiled'
    test_fn = make_test(net,data,label,non_learn_params,cfg.gm_num,cfg.ndim)
    print 'test_fn compiled'

    metrix = { 'aps' : average_precision_score,
               'pp'  : lambda l,pred : np.ones(len(l))[(pred < 0.5)&(l<0.1)].sum()/np.ones(len(l))[l<0.1].sum(),
               'pn'  : lambda l,pred : np.ones(len(l))[(pred >= .5)&(l>0.9)].sum()/np.ones(len(l))[l>0.9].sum(),
               'int_pp_pn' : lambda l,pred : get_pp_pn(l,pred)}

    def update_params(epoch,params):
        if(epoch == 0):
            params['min_cov'].set_value(1e-4)
            params['lr'].set_value(5e-2)
        if(epoch == 4):
            params['min_cov'].set_value(1e-8)
            params['lr'].set_value(5e-3)
        if(epoch == 10):
            params['lr'].set_value(1e-3)

    fout = open(logger_name,'a')
    fout.write('################### train network '+cfg.NAME+ ' ' + str(datetime.datetime.now())+'################\n')    
    fout.write(str(cfg))
    fit(cfg.NAME,net,train_fn,test_fn,train_loader,test_loader,non_learn_params,
        update_params=update_params,
        metrix = metrix,
        logger=fout,
        train_esize = 1500,
        test_esize = 750,
        epochs=cfg.TRAIN.EPOCH_NUM-cfg.TRAIN.EPOCH)

    test_network(cfg.NAME,FCN,cfg.ndim,cfg.TRAIN.EPOCH_NUM-1,cfg.gm_num)
    fout.write('################### done #######################\n')

    
from easydict import EasyDict
cfg = EasyDict()
cfg.SEQ_LENGTH = 250
cfg.TILE_SIZE = 32
cfg.OUT_SIZE = 1
cfg.TRAIN = EasyDict()
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.EPOCH_SIZE = 1000
cfg.TRAIN.EPOCH_NUM = 20
cfg.gm_num = 4
cfg.ndim = 12
cfg.NAME = 'conv4_net_dense_color%i'%(cfg.ndim)#'conv_net_no_bn_ndim12010'#


train_network(conv4_net_dense_color,cfg)