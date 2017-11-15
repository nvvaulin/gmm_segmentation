import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
   

class TieLoader:
    def __init__(self,path='train_ties',cols = 16,rows=16,t_size=48):
        self.t_size = t_size
        self.cols = cols
        self.rows = rows
        if(os.path.exists(path+'/list.txt')):
            self.img_list = [i[:-1] for i in open(path+'/list.txt')]
        else:
            self.img_list = self.list_files(path)
            f = open(path+'/list.txt','w')
            for i in self.img_list:
                f.write(i+'\n')
            f.close()
            
    def read_bg_for(self,path):
        print path[:path.rfind('/')+1]+'motion_input.jpg'
        im = cv2.imread(path[:path.rfind('/')+1]+'motion_input.jpg')
        if(im is None):
            return None
        cols = im.shape[1]//self.t_size
        rows = im.shape[0]//self.t_size
        h,w = self.t_size,self.t_size
        data  = np.zeros((cols*rows,3,h,w),dtype=np.uint8)
        for c in range(cols):
            for r in range(rows):
                if(r*self.cols+c < len(data)):
                    data[r*self.cols+c,:] = np.transpose(im[r*h:r*h+h,c*w:c*w+w,:],(2,0,1))
        return data[data.mean((1,2,3)) > 0.1]
        
        
        
    def list_files(self,path):
        res = []
        for i in os.listdir(path):
            if(os.path.isdir(path+'/'+i)):
                res = res+ self.list_files(path+'/'+i)
            elif(i[-3:] == 'jpg'):
                return [path+'/'+str(j) for j in [ int(k[:k.rfind('_')]) for k in os.listdir(path) if k.find('motion') < 0]]
        return res
    
    def load_random(self):
        i = int(np.random.randint(0,len(self.img_list)))
        im = cv2.imread(self.img_list[i]+'_input.jpg')
        mask = cv2.imread(self.img_list[i]+'_mask.jpg',0)
        h,w = im.shape[0]//self.rows,im.shape[1]//self.cols
        data  = np.zeros((self.cols*self.rows,4,h,w),dtype=np.uint8)
        for c in range(self.cols):
            for r in range(self.rows):
                if(r*self.cols+c < len(data)):
                    data[r*self.cols+c,1:] = np.transpose(im[r*h:r*h+h,c*w:c*w+w,:],(2,0,1))
                    data[r*self.cols+c,0] = mask[r*h:r*h+h,c*w:c*w+w]
        return data,self.read_bg_for(self.img_list[i])
        
from numpy import random as rnd
from numba import jit
@jit
def transform(data,tile_size,scale_range = [0.9,1.1],rot_range = 360.):
    scale_range = [0.9,1.1]
    rot_range = 360.
    w,h = tile_size
    res = np.empty((data.shape[0],data.shape[1],h,w),dtype=np.float32)
    angle = rot_range*(0.5-np.random.rand())
    scale = scale_range[0]+(scale_range[1]-scale_range[0])*rnd.rand() 
    flip_code = rnd.randint(4)-2
    center = rnd.randint(w/2)+w/2,rnd.randint(h/2)+h/2
    rot = cv2.getRotationMatrix2D(center,angle,scale)
    rot[0,2] -= center[0]-w/2.
    rot[1,2] -= center[1]-h/2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            res[i,j] = cv2.warpAffine(data[i,j].astype(np.float32),rot,(h,w))
            if(flip_code != -2):
                res[i,j] = cv2.flip(res[i,j],flip_code)
    return res

class GMMDataLoader():
    def __init__(self,dl,tile_size,out_size,seq_length,min_r= 0.25,max_r=0.4):
        self.dl = dl
        self.out_size = out_size
        self.tile_size = tile_size
        self.seq_length = seq_length
        self.min_r = min_r
        self.max_r = max_r
    
    def load_from_dl(self):
        y = np.zeros((self.seq_length,  self.tile_size[1],self.tile_size[0]),dtype=np.float32)+0.5
        data,motion = self.dl.load_random()
        lo = np.random.randint(0,len(data)-self.seq_length)
        tmp = data[lo:lo+self.seq_length]
        tmp = transform(tmp,self.tile_size,rot_range=30.)
        
        X = tmp[:,1:,:,:]
        y[tmp[:,0] > 250] = 1.
        y[tmp[:,0] <= 20] = 0.
        y = y[:,(self.tile_size[1]-self.out_size[1])//2:(self.tile_size[1]-self.out_size[1])//2+self.out_size[1],\
                (self.tile_size[0]-self.out_size[0])//2:(self.tile_size[0]-self.out_size[0])//2+self.out_size[0]]
        return X,y,motion
    
    
    def load_random(self):
        while(1):
            X,y,data =self.load_from_dl()
            ii = np.random.randint(len(data))
            means = y.mean((1,2))
            inx = means.argsort()
            if(means[inx[0]] < 0.1) and (y[(y>0.1)&(y<0.9)].size < 0.1*y.size):
                break
        if(means.mean() < self.min_r):
            if(data is None):
                continue
            for i in range(len(inx)//2):
                X[inx[i]] = data[ii][:,:self.tile_size[1],:self.tile_size[0]]
                ii = (ii+1) % len(data)
                y[inx[i]] = 1.
                if((means[inx[i:]].sum()+i+1)/len(means) > self.min_r):
                    break
        elif(means.mean() > self.max_r):
            for i in range(1,len(inx)//2):
                X[inx[-i]] = X[inx[0]]
                y[inx[-i]] = y[inx[0]]
                if((means[inx[:-i]].sum()+means[inx[0]]*i)/len(means) < self.max_r):
                    break
        means = y.mean((1,2))
        inx = means.argsort()
        return X,y
    

def draw_sample(X,y,cols=10,rows=10):
    X = np.transpose(X,(0,2,3,1))
    res = X[:cols*rows,(X.shape[1]-y.shape[1])//2:(X.shape[1]-y.shape[1])//2+y.shape[1],\
              (X.shape[2]-y.shape[2])//2:(X.shape[2]-y.shape[2])//2+y.shape[2]].copy()
    _,h,w,c = res.shape
    mask = np.zeros_like(res)
    res = np.transpose(res.reshape((rows,cols,h,w,c)),(0,2,1,3,4)).reshape((rows*h,cols*w,c))
    mask[:] = y[:cols*rows,:,:,None]
    mask = np.transpose(mask.reshape((rows,cols,h,w,c)),(0,2,1,3,4)).reshape((rows*h,cols*w,c))
    plt.figure(figsize=(10,10))
    plt.imshow(np.concatenate((res.astype(np.uint8),(mask*255).astype(np.uint8)),axis=1))
    plt.show()


    