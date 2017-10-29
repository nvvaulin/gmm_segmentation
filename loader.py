import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_path(p):
    dirs = p.split('/')
    tmp = ''
    for i in range(len(dirs)):
        tmp = tmp+dirs[i]+'/'
        if not(os.path.exists(tmp)):
            os.mkdir(tmp)    

def save_tie(t_im,t_mask,prefix,max_length):
    assert(len(t_im) == max_length)
    cols = int(np.sqrt(len(t_im)))
    rows = int(np.ceil(len(t_im)/float(cols)))
    
    w = t_im.shape[2]
    h = t_im.shape[1]
    res_im = np.zeros((rows*h,cols*w,3),dtype=np.uint8)
    res_mask = np.zeros((rows*h,cols*w),dtype=np.uint8)
    for c in range(cols):
        for r in range(rows):
            if(r*cols+c < len(t_im)):
                res_im[r*h:r*h+h,c*w:c*w+w,:] = t_im[r*cols+c,:]
                res_mask[r*h:r*h+h,c*w:c*w+w] = t_mask[r*cols+c]
    cv2.imwrite(prefix+'_input.jpg',res_im)
    cv2.imwrite(prefix+'_mask.jpg',res_mask)
    
def iterate_video(d):
    d = d[(d.r1<0.9)&(d.r2<0.9)]
    if(len(d) > 0):
        im_shape = cv2.imread(d.x.values[0]).shape
    for i in range(len(d)):
        yield cv2.imread(d.x.values[i]), cv2.imread(d.y.values[i],0)
    
def iterate_img(img,mask,t_size):
    for i in range(img.shape[0]//t_size):
        for j in range(img.shape[1]//t_size):
            lo = (i*t_size),(j*t_size)
            hi = (i+1)*t_size if (i+2)*t_size < img.shape[0] else None,\
                 (j+1)*t_size if (j+2)*t_size < img.shape[1] else None
            tie_mask = mask[lo[0]:hi[0],lo[1]:hi[1]]
            tie_img  = img[lo[0]:hi[0],lo[1]:hi[1]]
            yield tie_img,tie_mask
        
def process_tie(ties,tie_mask):
        r0 = np.zeros_like(tie_mask,dtype=np.float32)
        r1 = np.zeros_like(tie_mask,dtype=np.float32)
        r3 = np.zeros_like(tie_mask,dtype=np.float32)
        r0[tie_mask < 10] = 1.
        r1[(tie_mask > 10)&(tie_mask < 240)] = 1.
        r3[tie_mask > 240] = 1.
        if(r1.mean() > 0.7):
            return False
        if(r3.mean() > 0.001):
            return True
        return False
    
def create_dataset(path = 'dataset/data.csv',out_dir='ties256',t_size=64,max_length=256):
    data = pd.read_csv(path)
    for video_num,d in enumerate(data.groupby('id')):
        ties = []
        count = 0
        video_path = (out_dir+'/'+d[1].x.values[0])[:(out_dir+'/'+d[1].x.values[0]).rfind('/input')]
        make_path(video_path)
        for frame_num,(img,mask) in enumerate(iterate_video(d[1])):
            for i,(tie_img,tie_mask) in enumerate(iterate_img(img,mask,t_size)):
                if(len(ties)<=i):
                    ties.append([[tie_img],[tie_mask]])
                else:
                    ties[i][0].append(tie_img)
                    ties[i][1].append(tie_mask)
                if(len(ties[i][0]) >= max_length):
                    tie_img = np.array(ties[i][0],dtype=np.uint8)
                    tie_mask = np.array(ties[i][1],dtype=np.uint8)
                    if(process_tie(tie_img,tie_mask)):            
                        save_tie(tie_img,tie_mask,video_path+'/'+str(count),max_length=max_length)
                        count = count+1
                    ties[i] = [[],[]]
            print "\rvideo: %i frame: %i proper_ties: %i"%(video_num,frame_num,count),
        print ""
    
class TieLoader:
    def __init__(self,path='train_ties'):
        self.cols = 16
        self.rows = 16
        if(os.path.exists(path+'/list.txt')):
            self.img_list = [i[:-1] for i in open(path+'/list.txt')]
        else:
            self.img_list = self.list_files(path)
            f = open(path+'/list.txt','w')
            for i in self.img_list:
                f.write(i+'\n')
            f.close()
    
    def list_files(self,path):
        res = []
        for i in os.listdir(path):
            if(os.path.isdir(path+'/'+i)):
                res = res+ self.list_files(path+'/'+i)
            elif(i[-3:] == 'jpg'):
                return [path+'/'+str(j) for j in [ int(k[:k.rfind('_')]) for k in os.listdir(path)]]
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
        return data
        
        
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
        self.data = np.load('data.npz')["arr_0"]
        self.tile_size = tile_size
        self.seq_length = seq_length
        self.min_r = min_r
        self.max_r = max_r
    
    def load_from_dl(self):
        y = np.zeros((self.seq_length,  self.tile_size[1],self.tile_size[0]),dtype=np.float32)+0.5
        data = self.dl.load_random()
        lo = np.random.randint(0,len(data)-self.seq_length)
        tmp = data[lo:lo+self.seq_length]
        tmp = transform(tmp,self.tile_size)        
        X = tmp[:,1:,:,:]
        y[tmp[:,0] > 250] = 1.
        y[tmp[:,0] <= 20] = 0.
        y = y[:,(self.tile_size[1]-self.out_size[1])//2:(self.tile_size[1]-self.out_size[1])//2+self.out_size[1],\
                (self.tile_size[0]-self.out_size[0])//2:(self.tile_size[0]-self.out_size[0])//2+self.out_size[0]]
        return X,y
    
    
    def load_random(self):
        while(1):
            X,y =self.load_from_dl()
            means = y.mean((1,2))
            inx = means.argsort()
            if(means[inx[0]] < 0.1) and (y[(y>0.1)&(y<0.9)].size < 0.1*y.size):
                break
        if(means.mean() < self.min_r):
            for i in range(len(inx)//2):
                X[inx[i]] = self.data[np.random.randint(len(self.data))][:,:self.tile_size[1],:self.tile_size[0]]
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
        return X[inx],y[inx]

    def create_data(self):
        all_imgs = None        
        for i,sample in enumerate(self.iterate_all_imgs()):
            imgs = self.get_move_tile(sample,self.tile_size)
            if(all_imgs is None):
                all_imgs = imgs
            else:
                all_imgs = np.concatenate((all_imgs,imgs),0)
            print '\r',i,len(all_imgs),
        np.savez("data.npz",all_imgs)

    def iterate_all_imgs(self):
        data = self.vl.data[self.vl.data.r4 > 0.05]
        for i in data.index:
            sample = self.vl.load_data(data[data.index == i],is_color=True)
            yield sample[1]
    
    def get_move_tile(self,sample,t_size):
        mask = np.zeros_like(sample[0],dtype=np.float32)
        mask[sample[0] > 240] = 1.
        mask = np.cumsum(mask,1)
        mask = np.cumsum(mask,0)
        w,h = t_size
        mask = (mask[h:,w:]+mask[:-h,:-w] - mask[h:,:-w] - mask[:-h,w:])/(t_size[0]*t_size[1])
        mask = mask[::t_size[1],::t_size[0]]
        inx = np.arange(0,mask.size,dtype=np.int)
        inx = inx[mask.flatten() > 0.93]
        lo_y,lo_x = np.unravel_index(inx,mask.shape)
        lo_y*=t_size[1]
        lo_x*=t_size[0]
        hi_x=lo_x+w
        hi_y=lo_y+h
        mask = (hi_x<=mask.shape[1]*t_size[0])&(hi_y<=mask.shape[0]*t_size[1])
        hi_x = hi_x[mask]
        hi_y = hi_y[mask]
        lo_x = lo_x[mask]
        lo_y = lo_y[mask]
        imgs = np.zeros((len(lo_x),3,t_size[1],t_size[0]),dtype=np.uint8)
        for i in range(len(imgs)):
            imgs[i] = sample[1:,lo_y[i]:hi_y[i],lo_x[i]:hi_x[i]]
        return imgs
    

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
