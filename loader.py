import os
import cv2
import numpy as np
from dataset_tools import *
   
class TieLoader:
    def __init__(self,path,min_r,max_r,t_size=48,sample_size = 32,mask_size = 32):
        self.t_size = t_size
        self.min_r = min_r
        self.max_r = max_r
        self.sample_size = sample_size
        self.mask_size = mask_size
        if(os.path.exists(path+'/list.txt') and os.path.exists(path+'/motion.txt')):
            self.img_list = [i[:-1] for i in open(path+'/list.txt')]
            self.motion_list = [i[:-1] for i in open(path+'/motion.txt')]
        else:
            self.img_list = []
            self.motion_list = []
            for d in iterate_folders(path):
                self.img_list=self.img_list+[d+'/'+i[:i.rfind('_')] for i in os.listdir(d) if i[-len('input.jpg'):] == 'input.jpg']
                self.motion_list = self.motion_list+[d+'/'+i for i in os.listdir(d) if i[-len('motion.jpg'):] == 'motion.jpg']
            f = open(path+'/list.txt','w')
            for i in self.img_list:
                f.write(i+'\n')
            f.close()
            f = open(path+'/motion.txt','w')
            for i in self.motion_list:
                f.write(i+'\n')
            f.close()
            
    

            
    def load_motion(self,sample_path):
        p = sample_path[:sample_path.rfind('/')]
        m_paths = [i for i in self.motion_list if i[:min(len(i),len(p))] == p]
        if(len(m_paths) == 0):
            return None
        data = image_to_ties(cv2.imread(m_paths[np.random.randint(len(m_paths))]),self.t_size,self.t_size)
        data = data[data.sum((1,2,3)) > 10 ]
        return clip_ties(data,self.sample_size)
            
    
    def balance_tie(self,path,ties,mask):
        means = (mask.astype(np.float32)/255.).mean((1,2))
        inx = means.argsort()
        if(means[inx[0]] < 0.1):
            if(means.mean() < self.min_r):
                data = self.load_motion(path)
                if(data is None):
                    return None,None
                ii = np.random.randint(len(data))
                for i in range(len(inx)//2):
                    ties[inx[i]] = data[ii][:,:self.t_size,:self.t_size]
                    ii = ii+1
                    if(ii >= len(data)):
                        data = np.concatenate((data,self.load_motion(path)))
                    mask[inx[i]] = 255
                    if((means[inx[i:]].sum()+i+1)/len(means) > self.min_r):
                        return ties,mask
            elif(means.mean() > self.max_r):
                for i in range(1,len(inx)//2):
                    ties[inx[-i]] = ties[inx[0]]
                    mask[inx[-i]] = mask[inx[0]]
                    if((means[inx[:-i]].sum()+means[inx[0]]*i)/len(means) < self.max_r):
                        return ties,mask
        return None,None
        
    
    def load_sample(self,path):
        tie = image_to_ties(cv2.imread(path+'_input.jpg'),self.t_size,self.t_size)
        tie = clip_ties(tie,self.sample_size)
        mask = image_to_ties(cv2.imread(path+'_mask.png',0),self.t_size,self.t_size)
        mask = clip_ties(mask,self.mask_size)
        tie,mask = self.balance_tie(path,tie,mask)
        return tie,mask
    
    def iterate(self,shuffle=False):
        if(shuffle):
            np.random.shuffle(self.img_list)
        for path in self.img_list:
            ties,mask = self.load_sample(path)
            if(ties is None):
                continue
            yield ties,mask
            
        

def data_generator(gmm_loader,
                   epoch_size,
                   shuffle=False):
    for i,(ties,mask) in enumerate(gmm_loader.iterate(shuffle)): 
        if(i >= epoch_size):
            break
        ties = np.transpose(ties,(0,3,1,2)).astype(np.float32)
        mask = mask.astype(np.float32)/255.
        mask[mask <= 0.1] = 0
        mask[mask >= 0.9] = 1.
        mask[(mask < 0.9)&(mask > 0.1)] = 0.5
        yield ties,mask
        