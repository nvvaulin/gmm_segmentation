import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def clip_ties(X,o_size):
        t_size = X.shape[-2]
        if(len(X.shape) == 3):
            return X[:,(t_size-o_size)//2:(t_size+o_size)//2,\
                      (t_size-o_size)//2:(t_size+o_size)//2]

        else:
            return X[:,(t_size-o_size)//2:(t_size+o_size)//2,\
                      (t_size-o_size)//2:(t_size+o_size)//2,:]

def resize(im,mask,size):
    def _resize(i):
        pi = Image.fromarray(i)
        pi = pi.resize(size)
        return np.array(pi)
    return _resize(im),_resize(mask)

def ties_to_image(ties,cols=None,rows=None):
    if(cols is None):
        length = len(ties)
        cols = int(np.ceil(np.sqrt(length)))
        rows = int(np.ceil(float(length)/float(cols)))
    if(len(ties.shape) == 3):
        c = 1
        l,h,w = ties.shape
    else:
        l,h,w,c = ties.shape
    ties = ties[:min(l,cols*rows)]
    im = np.zeros(((rows*cols,)+ties.shape[1:]),dtype=ties.dtype)
    im[:len(ties)] = ties
    im = np.transpose(im.reshape((rows,cols,h,w,c)),(0,2,1,3,4)).reshape((rows*h,cols*w,c))
    if(c==1):
        return im[...,0]
    else:
        return im
                
def image_to_ties(im,tie_w,tie_h):
    h,w = tie_h,tie_w
    cols = int(np.ceil(float(im.shape[1])/float(w)))
    rows = int(np.ceil(float(im.shape[0])/float(h)))
    c = 1 if len(im.shape) == 2 else im.shape[-1]
    im = im.reshape((im.shape[0],im.shape[1],c))
    ties = np.zeros((rows*h,cols*w,c),dtype=im.dtype)
    ties[:im.shape[0],:im.shape[1],:] = im
    ties = np.transpose(ties.reshape((rows,h,cols,w,c)),(0,2,1,3,4)).reshape(cols*rows,h,w,c)
    if(c==1):
        return ties[...,0]
    else:
        return ties
    

def iterate_folders(dataset):
    '''
    iterate over all dataset videos
    dataset : root path to dataset
    '''
    for subsets in os.listdir(dataset):
        if not(os.path.isdir(dataset+'/'+subsets)):
            continue
        for video in os.listdir(dataset+'/'+subsets):
            yield dataset+'/'+subsets+'/'+video
                
def iterate_video(folder,skip_first_unlabled=True):
    '''
    iterate over frames of video
    folder : path to video
    skip_first_unlabled : skip unlabeled frames at the beginning and end of video
    '''
    
    if(skip_first_unlabled):
        f = [int(i) for i in open(folder+'/temporalROI.txt').read().split(' ')]
    else:
        img_range = range(1,len([i for i in os.listdir(folder+'/input') if (i[-4:] == '.jpg')])+1)
        
    for i in range(f,all_img_num+1):
        name='%06d'%(i)
        mask = cv2.imread(folder+'/groundtruth/gt'+name+'.png',0)        
        im = cv2.imread(folder+'/input/in'+name+'.jpg')
        yield name,im,mask

def iterate_bathced(folder,num_frames,size=None):
    imgs = None
    masks = None
    frame_names = None
    for i,(name,im,mask) in enumerate(iterate_video(folder,True)):
        if(size is None):
            size = im.shape[1],im.shape[0]
        im,mask = resize(im,mask,size)
        if(imgs is None):
            imgs  = np.zeros((num_frames,size[1],size[0],3),dtype=np.uint8)
            masks = np.zeros((num_frames,size[1],size[0]),dtype=np.uint8)
            frame_names = ['' for i in range(num_frames)]
            
        imgs[i % num_frames] = im
        masks[i % num_frames] = mask
        frame_names[i % num_frames] = name
        if((i+1)%num_frames == 0):
            yield frame_names,imgs,masks

def make_path(p):
    dirs = p.split('/')
    tmp = ''
    for i in range(len(dirs)):
        tmp = tmp+dirs[i]+'/'
        if not(os.path.exists(tmp)):
            os.mkdir(tmp)    
            
def draw(ties,mask,cols=None,rows=None):
    im = ties_to_image(ties,cols,rows)
    mask = ties_to_image(mask,cols,rows)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(np.concatenate((im,mask),1))
    plt.show()
    

def binarise_prediction(prediction,threshold):
    result = np.zeros_like(prediction,dtype=np.uint8)
    result[prediction < -0.001] = 256/2
    result[prediction > threshold] = 255
    return result

def get_labeled_mask(label):
    return (label < 10) | (label > 240)