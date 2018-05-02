import os
import cv2
import numpy as np
from dataset_tools import *
from sklearn.metrics import average_precision_score
from sklearn import metrics
import theano
import theano.tensor as T
from gmm_op import get_gmm,calc_log_prob_gmm
from sklearn import mixture
from lasagne import layers as L
from multiprocessing import Pool
from utils import Logger

class BGAlgorithm(object):
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def train_batched(self,data,labels):
        pass
    
    def predict_batched(self,data,labels):
        raise NotImplementedError

    
def map_fit_gmm(args):
    if(args[0] is None):
        return None
    else:
        try:
            return args[0].fit(args[1])
        except:
            print args[1].shape
            print args[1]
            raise
    
class GMMAlgorithm(BGAlgorithm):
    def __init__(self,FCN,gm_num,pool=None):
        super(GMMAlgorithm,self).__init__()
        self.gm_num = gm_num
        self.pool = pool
        self.feature_fn = None
        self.reset()
        self.FCN = FCN
        self.im_size = (-1,-1)
        self.predict_fn = self._make_predict_fn()
        self.min_samples_for_gmm = self.gm_num=10
        
    def _make_feature_fn(self,im_size):        
        data=T.tensor4()
        feature_sym = self.FCN(data=data,input_shape=(1,3,im_size[1],im_size[0]))
        return theano.function([data],feature_sym,allow_input_downcast=True)
        
    def _make_predict_fn(self):
        data,m,c,w=T.matrix(),T.matrix(),T.matrix(),T.vector()
        return theano.function([data,m,c,w],1.-T.nnet.sigmoid(calc_log_prob_gmm(data,m,c,w)),allow_input_downcast=True)
        
    def _make_features(self,imgs):
        if(self.im_size[0] != imgs.shape[2] or self.im_size[1] != imgs.shape[1]):
            self.im_size = imgs.shape[2],imgs.shape[1]
            self.feature_fn = self._make_feature_fn(self.im_size)
        data = None
        for i in range(len(imgs)):
            tmp = self.feature_fn(np.transpose(imgs[i:i+1],(0,3,1,2)).astype(np.float32))[0]
            if(data is None):
                data = np.empty((len(imgs),)+tmp.shape,dtype=np.float32)
            data[i] = tmp
        return data
    
    def _predict(self,features):
        res = np.zeros(features.shape[:2],dtype=np.float32)
        for i in range(features.shape[0]):
            gmm = self.gmms[i]
            if (not (gmm is None)):
                res[i] = self.predict_fn(features[i],gmm.means_,gmm.covariances_,gmm.weights_)
            else:
                res[i] = -1
        return res    
        
    def _fit_gmms(self,features,labels):
        args = []
        for i in range(features.shape[0]):
            f = features[i][labels[i] < 30]
            if(len(f) > self.min_samples_for_gmm):
                gmm = mixture.GaussianMixture( covariance_type='diag',
                                               n_components=self.gm_num,
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
        
    def reset(self):
        pass
    
    def train_batched(self,imgs,labels):
        features = self._make_features(imgs)
        flat_features,flat_labels = self._flatten(features,labels)
        self.gmms = self._fit_gmms(flat_features,flat_labels)
      
    def predict_batched(self,imgs,labels):
        features = self._make_features(imgs)
        flat_features,flat_labels = self._flatten(features,labels)
        flat_prediction = self._predict(flat_features)        
        prediction = np.transpose(flat_prediction,(1,0)).reshape(imgs.shape[:-1])
        return prediction
 

def bin_score(score,threshold = 0.5):
    res = np.zeros_like(score,dtype=np.int32)
    res[score > threshold] = 1
    return res

def calc_prediction_metrics(prediction,labels,metrics):
    prediction,labels = prediction.flatten(),labels.flatten()    
    mask = ((labels > 240) | (labels < 10)) & (prediction>-0.001)
    
    binary_label = np.zeros_like(labels)
    binary_label[labels > 240] = 1
    binary_label,prediction = binary_label[mask],prediction[mask]
    return dict([(k,metrics[k](binary_label,prediction)) for k in metrics.keys()])

def make_test(algorithm,
              out_dir=None,
              dataset='dataset',
              train_size = 100,
              test_size=200,
              im_size = None,
              metrics = {'aps' : average_precision_score,
                         'f1' : lambda y,s : metrics.f1_score(y,bin_score(s)),
                         'acc' : lambda y,s : metrics.accuracy_score(y,bin_score(s))},
              logger=Logger('std'),
              only_with_motion = True):
    all_results = dict()
    total_results = dict([(k,[]) for k in metrics])
    for in_dir in iterate_folders(dataset):
        logger.log(in_dir+' : ')
        algorithm.reset()
        for names,imgs,labels in iterate_bathced(in_dir,test_size+train_size,im_size):                
            test_imgs,test_labels = imgs[train_size:],labels[train_size:]
            train_imgs,train_labels = imgs[:train_size],labels[:train_size]
            test_names = names[train_size:]
            if(only_with_motion and test_labels[test_labels > 240].size < 10):
                logger.log('skip',end=' ')
                continue        
            imgs,labels = None,None            
            algorithm.train_batched(train_imgs,train_labels)
            prediction = algorithm.predict_batched(test_imgs,test_labels)
            logger.log(test_labels[test_labels>240].size)
            batch_result = calc_prediction_metrics(prediction,test_labels,metrics)
            for k in batch_result:
                logger.log('%s : %f'%(k,batch_result[k]))
                total_results[k].append(batch_result[k])
            all_results[in_dir] = batch_result
            if(not out_dir is None):
                make_path(out_dir+in_dir[len(dataset):])
                for i in range(len(test_imgs)):
                    cv2.imwrite(out_dir+'/'+test_names[i]+'.png',binary_prediction[i])
                    cv2.imwrite(out_dir+'/'+test_names[i]+'_true.png',binary_label[i])
                    cv2.imwrite(out_dir+'/'+test_names[i]+'_input.jpg',test_imgs[i])
            break
    all_results['total'] = dict([(k,np.array(total_results[k]).mean()) for k in total_results])
    logger.log('total : ')
    for k in total_results:
        logger.log('%s : %f'%(k,all_results['total'][k]))
    return all_results
