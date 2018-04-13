import numpy as np
from loader import TieLoader
from networks import make_FCN
from train_tools import make_train,make_test
import theano
import theano.tensor as T
from sklearn.metrics import average_precision_score
import datetime
from utils import get_pp_pn
from train_tools import fit
from test_tools import test_network
import argparse
from easydict import EasyDict
from utils import tee
import os

def train_network(cfg):
    try:
        os.mkdir(cfg.NAME)
    except Exception as e:
        print 'cannot create dir ',cfg.NAME
        print e
    if( cfg.TRAIN.EPOCH < 0):
        epoch = np.array([int(i[:-len('.npz')]) for i in os.listdir(cfg.NAME+"/models")]).max()
        cfg.TRAIN.EPOCH = int(epoch)
        print 'start epoch ',int(epoch)
    
    
    logger_name = cfg.NAME+'/train.log'

    train_loader = TieLoader('data/train_ties%d'%cfg.DATASET.T_SIZE,
                             cfg.DATASET.TRAIN_MINR,
                             cfg.DATASET.TRAIN_MAXR,
                             t_size=cfg.DATASET.T_SIZE,
                             mask_size=cfg.OUT_SIZE,
                             sample_size=cfg.TILE_SIZE,
                             cache_samples=cfg.DATASET.CASHE_SAMPLES)
    test_loader = TieLoader('data/test_ties%d'%cfg.DATASET.T_SIZE,
                            cfg.DATASET.TEST_MINR,
                            cfg.DATASET.TEST_MAXR,
                            t_size=cfg.DATASET.T_SIZE,
                            mask_size=cfg.OUT_SIZE,
                            sample_size=cfg.TILE_SIZE,
                            cache_samples=cfg.DATASET.CASHE_SAMPLES)
    
    data = T.tensor4(name='data')
    label = T.tensor3(name='label')
    net = make_FCN(cfg.NETWORK,data,
               ndim=cfg.NDIM,
               model_name='%s/models/%03d'%(cfg.NAME,cfg.TRAIN.EPOCH) if cfg.TRAIN.EPOCH > 0 else '',
               input_shape = (None,3,cfg.TILE_SIZE,cfg.TILE_SIZE),
               pad = 'same')

    non_learn_params={'min_cov' : theano.shared(1e-8),
                      'lr' : theano.shared(np.array(1e-2, dtype=theano.config.floatX)),
                      'width': theano.shared(4.),
                      'total_grad_constraint': 10,
                      'histogram_bins' : 100,
                      'use_approx_grad' : True,
                      'ndim' : cfg.NDIM,
                      'gm_num' : cfg.GM_NUM}

    train_fn = make_train(net,data,label,non_learn_params)
    print 'train_fn compiled'
    test_fn = make_test(net,data,label,non_learn_params)
    print 'test_fn compiled'

    metrix = { 'aps' : average_precision_score,
               'int_pp_pn' : lambda l,pred : get_pp_pn(l,pred)}

    def update_params(epoch,params):
        if(epoch == 0):
            params['lr'].set_value(5e-2)
        if(epoch == 4):
            params['lr'].set_value(5e-3)
        if(epoch == 10):
            params['lr'].set_value(1e-3)

    logger = open(logger_name,'a')
    tee('\n################### train network '+cfg.NAME+ ' ' + str(datetime.datetime.now())+'################\n',logger)    
    tee('config\n'+str(cfg),logger)
    tee('non_learn_params\n'+str(non_learn_params),logger)
    fit(cfg.NAME,net,train_fn,test_fn,train_loader,test_loader,non_learn_params,
        update_params=update_params,
        metrix = metrix,
        logger=logger,
        train_esize = cfg.TRAIN.TRAIN_EPOCH_SIZE ,
        test_esize = cfg.TRAIN.TEST_EPOCH_SIZE ,
        epochs=cfg.TRAIN.EPOCH_NUM,
        start_epoch=cfg.TRAIN.EPOCH)

    tee('\n################### test network '+cfg.NAME+ ' ' + str(datetime.datetime.now())+'################\n',logger)    
    test_network(cfg.NAME,cfg.NETWORK,cfg.NDIM,cfg.TRAIN.EPOCH_NUM-1,cfg.GM_NUM,im_size=(320,240),train_size=100,test_size=400)
    tee('\n################### done #######################\n',logger)

cfg = EasyDict()
cfg.SEQ_LENGTH = 250
cfg.TILE_SIZE = 9
cfg.OUT_SIZE = 1
cfg.GM_NUM = 4
cfg.NDIM = 4
cfg.NAME_PREFIX = ''
cfg.NETWORK = ''
cfg.TRAIN = EasyDict()
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.TRAIN_EPOCH_SIZE = 1500
cfg.TRAIN.TEST_EPOCH_SIZE = 750
cfg.TRAIN.EPOCH_NUM = 20

cfg.DATASET = EasyDict()
cfg.DATASET.TRAIN_MINR = 0.2
cfg.DATASET.TRAIN_MAXR = 0.3
cfg.DATASET.TEST_MINR = 0.2
cfg.DATASET.TEST_MAXR = 0.3
cfg.DATASET.T_SIZE = 32
cfg.DATASET.CASHE_SAMPLES = True

def iter_all_keys_in_dict(prefix,d):
    res = []
    for k in d.keys():
        try:
            d[k].keys()
            res = res + iter_all_keys_in_dict((prefix+'.' if( len(prefix) > 0) else '')+k,d[k])
        except:
            names = (prefix+'.' if( len(prefix) > 0) else '')+k
            default = d[k]
            res.append((names,default))
            
    return res

def set_values_to_config(args,cfg):
    for arg in vars(args):
        val = getattr(args, arg)
        if(val is None):
            continue
        arg = arg.upper()
        c = cfg
        path = arg.split('.')
        for i in path[:-1]:
            c = c[i]
        c[path[-1]] = val
        print 'set %s to '%arg,val
    return cfg
    

parser = argparse.ArgumentParser(description='Train gmm segmentation network')
reqiered_args = ['ndim','network_prefix','network']
for name,default in iter_all_keys_in_dict('',cfg):
    if(name.lower() in reqiered_args):
        parser.add_argument('--'+name.lower(), type = type(default),default=None)
    else:
        parser.add_argument('--'+name.lower(), type = type(default),default=None,nargs='?')


args = parser.parse_args()
cfg = set_values_to_config(args,cfg)
cfg.NAME='experiments/'+cfg.NAME_PREFIX+cfg.NETWORK+'%d'%(cfg.NDIM)    
train_network(cfg)