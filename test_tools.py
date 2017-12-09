import os
import cv2
import numpy as np
from sklearn import mixture

def resize(im,mask,size):
    im = cv2.resize(im,size)
    mask = cv2.resize(mask,size)
    mask[mask < 30] = 0
    mask[(mask >=30)&(mask <230)] = 255//2
    mask[(mask >=230)] = 255
    return im,mask
    
def iterate_video(folder,min_rate = 0.01):
    for name in [i[2:-4] for i in os.listdir(folder+'/input') if (i[-4:] == '.jpg')]:
        mask = cv2.imread(folder+'/groundtruth/gt'+name+'.png',0)
        if((mask[(mask>30)&(mask < 230)].size > mask.size//2) | (mask[mask > 230].size < min_rate*mask.size)):
            continue
        im = cv2.imread(folder+'/input/in'+name+'.jpg')
        yield im,mask
    
def iterate_bathced(folder,num_frames,size):
    imgs  = np.zeros((num_frames,size[1],size[0],3),dtype=np.uint8)
    masks = np.zeros((num_frames,size[1],size[0]),dtype=np.uint8)
    for i,(im,mask) in enumerate(iterate_video(folder)):
        im,mask = resize(im,mask,size)
        imgs[i % num_frames] = im
        masks[i % num_frames] = mask
        if((i+1)%num_frames == 0):
            yield imgs,masks

def draw(_imgs,_mask,_out,cols=10,rows=10):
    res =_imgs[:cols*rows,...].copy()
    _,h,w,c = res.shape
    mask = np.zeros_like(res)
    res = np.transpose(res.reshape((rows,cols,h,w,c)),(0,2,1,3,4)).reshape((rows*h,cols*w,c))
    mask[...,0] = _mask[:cols*rows,:,:]
    mask[...,2] = _out[:cols*rows,:,:]
    mask = np.transpose(mask.reshape((rows,cols,h,w,c)),(0,2,1,3,4)).reshape((rows*h,cols*w,c))
    plt.figure(figsize=(10,10))
    plt.imshow(np.concatenate((res,mask),axis=1))
    plt.show()
    
def make_path(p):
    dirs = p.split('/')
    tmp = ''
    for i in range(len(dirs)):
        tmp = tmp+dirs[i]+'/'
        if not(os.path.exists(tmp)):
            os.mkdir(tmp)    
            

def iretate_test_dataset(out_dir,dataset='dataset',max_frames=300,im_size = (320//2,240//2)):
    for d in os.listdir(dataset):
        if not(os.path.isdir(dataset+'/'+d)):
            continue
        folder = d+'/'+os.listdir(dataset+'/'+d)[0]
        out_folder = out_dir+'/'+folder
        make_path(out_folder)
        for imgs,masks in iterate_bathced(dataset+'/'+folder,max_frames,im_size):
            yield out_folder,imgs,masks
            break
       

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
            if not (masks is None):
                gmm = gmms[i*features.shape[2]+j]
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

