import os
import cv2
import numpy as np
from sklearn import mixture

def make_features(feature_fn,imgs):
    data = None
    for i in range(len(imgs)):
        tmp = feature_fn(np.transpose(imgs[i:i+1],(0,3,1,2)).astype(np.float32))[0]
        if(data is None):
            data = np.empty((len(imgs),)+tmp.shape,dtype=np.float32)
        data[i] = tmp
    return data

def make_gmms(shape,gm_num):
    gmms = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            gmms.append(mixture.GaussianMixture(covariance_type='diag',
                               n_components=gm_num,
                               max_iter=1000,
                               warm_start=True))
    return gmms

    

def fit_gmms(features,gmms,masks = None):
    for i in range(features.shape[1]):
        for j in range(features.shape[2]):
            f = features[:,i,j]
            gmm = gmms[i*features.shape[2]+j]
            if not (masks is None):
                if(len(f[masks[:,i,j] < 30]) > gmm.n_components*3):
                    f = f[masks[:,i,j] < 30]
            gmm.fit(f)


def predict_pixelwise(features,gmms,predict_fn):
    res = np.zeros_like(features[:,:,:,0])
    for i in range(features.shape[1]):
        for j in range(features.shape[2]):
            gmm = gmms[i*features.shape[2]+j]
            res[:,i,j] = predict_fn(features[:,i,j,:],gmm.means_,gmm.covariances_,gmm.weights_)
    return res

