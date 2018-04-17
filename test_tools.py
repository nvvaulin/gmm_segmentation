import os
import cv2
import numpy as np
from sklearn import mixture
from sklearn.metrics import average_precision_score
from dataset_tools import *
from gmm_op import get_gmm,calc_log_prob_gmm
import theano.tensor as T
from lasagne import layers as L
import theano
from multiprocessing import Pool
from utils import tee
from networks import make_FCN
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn import metrics
from utils import get_aps

class BGAlgorithm:
    def __init__(self):
        pass
    
    def train_batched(self,data,labels):
        pass
    
    def predict_batched(self,data):
        raise NotImplementedError

        
def map_fit_gmm(args):
    if(args[0] is None):
        return None
    else:
        return args[0].fit(args[1])
    
class GMMAlgorithm(BGAlgorithm):
    def __init__(self,FCN,gm_num,pool=Pool(4)):
        super(GMMAlgorithm,self).__init__()
        self.gm_num = gm_num
        self.pool = pool
        self.feature_fn = None
        self.reset()
        self.im_size = (-1,-1)
        self.predict_fn = self._make_predict_fn()
        self.min_samples_for_gmm = self.gm_num=10
        
    def _make_feature_fn(self,im_size):        
        data=T.tensor4()
        feature_net = FCN(data=data,input_shape=(1,3,im_size[1],im_size[0]))
        feature_sym = L.get_output(feature_net,deterministic=True)
        return theano.function([data],feature_sym,allow_input_downcast=True)
        
    def _make_predict_fn(self)
        data,m,c,w=T.matrix(),T.matrix(),T.matrix(),T.vector()
        return theano.function([data,m,c,w],1.-T.nnet.sigmoid(calc_log_prob_gmm(data,m,c,w)),allow_input_downcast=True)
        
    def _make_features(self,imgs):
        assert(imgs.shape[3] == 3)
        if(self.im_size[0] != imgs.shape[2] or self.im_size[1] != imgs.shape[1]):
            self.im_size = imgs.shape[2],imgs.shape[1]
            self.feature_fn = self._make_feature_fn(self.im_size)
        data = None
        for i in range(len(imgs)):
            tmp = feature_fn(np.transpose(imgs[i:i+1],(0,3,1,2)).astype(np.float32))[0]
            if(data is None):
                data = np.empty((len(imgs),)+tmp.shape,dtype=np.float32)
            data[i] = tmp
        return data
    
    def _predict(self,features):
        res = np.zeros(features.shape[:2],dtype=np.float32)
        for i in range(features.shape[0]):
            gmm = gmms[i]
            if (not (gmm is None)):
                res[i] = self.predict_fn(features[i],gmm.means_,gmm.covariances_,gmm.weights_)
            else:
                res[i] = -1
        return res    
        
    def _fit_gmms(self,features,labels):
        args = []
        for i in range(features.shape[0]):
            f = features[i][labels[i] < 30]
            if(len(f) > min_samples_for_gmm):
                gmm = mixture.GaussianMixture( covariance_type='diag',
                                               n_components=gm_num,
                                               max_iter=1000,
                                               warm_start=False)
            else:
                gmm = None
            args.append((gmm,f))
            
        if(self.pool is None):
            gmms = map(map_fit_gmm,args)
        else:
            gmms = self.pool.map(map_fit_gmm,args)
        return gmms
    
    def _flatten(self,features,labels):
        flat_features = np.transpose(features,(1,2,0,3)).reshape((-1,features.shape[0],features.shape[-1]))
        flat_labels = np.transpose(labels,(1,2,0)).reshape((-1,labels.shape[0]))
        return flat_features,flat_labels
        
    def reset():
        pass
    
    def train_batched(self,imgs,labels):
        features = self._make_features(imgs)
        flat_features,flat_labels = self._flatten(features,labels)
        self.gmms = self._fit_gmms(flat_features,flat_labels)
      
    def predict_batched(self,imgs):
        features = self._make_features(imgs)
        flat_features,flat_labels = self._flatten(features)
        flat_prediction = self._predict(flat_features)        
        prediction = np.transpose(flat_prediction,(1,0)).reshape(imgs.shape[:-1])
        return prediction

def bin_score(score,threshold = 0.5):
    res = np.zeros_like(score,dtype=np.int32)
    res[score > threshold] = 1
    return res

def make_test(algorithm,
              out_dir=None,
              dataset='dataset',
              train_size = 100,
              test_size=200,
              im_size = None,
              metrics = {'aps' : average_precision_score,
                         'f1' : lambda y,s : metrics.f1_score(y,bin_score(s)),
                         'acc' : lambda y,s : metrics.accuracy_score(y,bin_score(s))},
              logger=None):
        
    for in_dir in iterate_folders(dataset):
        tee(in_dir+':',logger)
        algorithm.reset()
        for names,imgs,labels in iterate_bathced(in_dir,test_size+train_size,im_size):                
            test_imgs,test_labels = imgs[train_size:],labels[train_size:]
            train_imgs,train_labels = imgs[:train_size],labels[:train_size]
            test_names = names[train_size:]
            imgs,labels = None,None            
            algorithm.train_batched(train_imgs,train_labels)
            prediction = algorithm.predict_batched(test_imgs)
            binary_prediction = binarise_prediction(prediction,0.5)
            mask = get_label_mask(label) & get_label_mask(binary_prediction)
            for _name,_metric in metrics:
                tee('%s : %f'%(_name,_metric(labels[mask],prediction[mask])),logger)
            if(not out_dir is None):
                make_path(out_dir+in_dir[len(dataset):])
                for i in range(len(test_imgs)):
                    cv2.imwrite(out_dir+'/'+test_names[i]+'.png',binary_prediction[i])
                    cv2.imwrite(out_dir+'/'+test_names[i]+'_true.png',test_labels[i])
                    cv2.imwrite(out_dir+'/'+test_names[i]+'_input.jpg',test_imgs[i])
            break

def test_network(name,network,ndim,epoch,gm_num,im_size=(320,240),train_size=100,test_size=300):
    data=T.tensor4()
    feature_net = make_FCN(network,
                           data=data,
                           ndim=ndim,
                           model_name='%s/models/%03d'%(name,epoch) if epoch >= 0 else '',
                           input_shape=(1,3,im_size[1],im_size[0]))
    feature_sym = L.get_output(feature_net,deterministic=True)
    feature_fn = theano.function([data],feature_sym,allow_input_downcast=True)
    data,m,c,w=T.matrix(),T.matrix(),T.matrix(),T.vector()
    predict_fn = theano.function([data,m,c,w],soft_predict_sym(data,m,c,w),allow_input_downcast=True)
    try:
        os.mkdir('%s/test'%(name))
    except:
        pass
    
    make_test_as_train(feature_fn,predict_fn,
                       out_dir='%s/test'%(name),
                       dataset='data/test',
                       gm_num=gm_num,
                       max_frames=train_size+test_size,
                       train_size=train_size,
                       im_size=im_size)
    calc_metric_all_folders('%s/test'%(name))
    
    