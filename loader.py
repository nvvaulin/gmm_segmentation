import os
import cv2
import numpy as np
from dataset_tools import *

def list_all_img(dir):
    res = []
    for i in os.listdir(dir):
        p = dir+'/'+i
        if(os.path.isdir(p)):
            res=res+list_all_img(p)
        elif (p[-4:] in ['.png','.jpg','.bmp']):
            res.append(p)
    return res


class IMDB:
    def __init__(self,all_paths):
        all_img = []
        self.paths = all_paths
        self.data = dict()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        if not(i in self.data):
            self.data.update({i:np.fromfile(i,dtype=np.uint8)})
        arr = self.data[i]
        if(cv2.__version__[0] == '3'):
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            return cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_UNCHANGED)
    
    def keys(self):
        return self.paths
            
    
def clip_random(imgs,masks=None,s=None):
    y = int(np.random.randint(imgs.shape[1]-s+1))
    x = int(np.random.randint(imgs.shape[2]-s+1))
    if(masks is None):
        return imgs[:,y:y+s,y:y+s]
    else:
        return imgs[:,y:y+s,y:y+s],masks[:,y:y+s,y:y+s]
    
class TieLoader:
    def __init__(self,path,min_r,max_r,t_size=48,sample_size = 32,mask_size = 32,cache_samples=False):
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
        self.motion = IMDB(self.motion_list)
        if(cache_samples):
            self.samples = IMDB([i+'_input.jpg' for i in self.img_list]+[i+'_mask.png' for i in self.img_list])
        else:
            self.samples = None
            
    def load_motion(self,sample_path):
        p = sample_path[:sample_path.rfind('/')]
        m_paths = [i for i in self.motion_list if i[:min(len(i),len(p))] == p]
        if(len(m_paths) == 0):
            return None
        data = image_to_ties(self.motion[m_paths[np.random.randint(len(m_paths))]],self.t_size,self.t_size)
        data = data[data.sum((1,2,3)) > 10 ]
        return clip_random(data,s=self.sample_size)
            
    
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
            else:
                return ties,mask
        return None,None
        
  
    
    def load_sample(self,path):
        if not (self.samples is None):
            tie = image_to_ties(self.samples[path+'_input.jpg'],self.t_size,self.t_size)
            mask = image_to_ties(self.samples[path+'_mask.png'],self.t_size,self.t_size)
        else:
            tie = image_to_ties(cv2.imread(path+'_input.jpg'),self.t_size,self.t_size)
            mask = image_to_ties(cv2.imread(path+'_mask.png',0),self.t_size,self.t_size)
        tie,mask = clip_random(tie,mask,self.sample_size)
        mask = clip_ties(mask,self.mask_size)
        tie,mask = self.balance_tie(path,tie,mask)
        return tie,mask

    
    def iterate(self,shuffle=False):
        img_list = [i for i in self.img_list]
        if(shuffle):
            np.random.shuffle(img_list)
        for path in img_list:
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


class PatchLoader(object):
    def __init__(self, root, t_size, seq_l, min_m, max_m, out_size):
        assert (out_size // 2) * 2 + 1 == out_size, 'out_size must be odd'
        self.root, self.seq_l, self.out_size = root, seq_l, out_size
        self.min_m, self.max_m = np.ceil(min_m * float(seq_l)), np.floor(max_m * float(seq_l))
        self.t_size = t_size
        self.patches = []
        self.motions = []
        self.hist = []
        self.motion_names = []
        pad = self.out_size // 2
        for video in iterate_folders(root):
            for p in os.listdir(video):
                prefix = video + '/' + p + '/'
                hist = cv2.imread(prefix + 'hist.png', 0)
                if (hist is None):
                    continue
                hist = hist[pad:hist.shape[0] - pad, pad:hist.shape[1] - pad]
                if (hist.max() < self.min_m):
                    continue
                self.hist.append(hist)
                all_jpg = [prefix + i[:i.rfind('.')] for i in os.listdir(prefix) if i.find('jpg') > 0]
                self.motions.append([i for i in all_jpg if i.find('motion') > 0][0])
                self.patches.append([i for i in all_jpg if i.find('motion') < 0])
                self.motion_names.append(np.array([i[:-1] for i in open(prefix + 'motion_info.txt')]))

    def load(self, path):
        patches = cv2.imread(path + '.jpg')
        patches = image_to_ties(patches, self.t_size, self.t_size)
        if (os.path.exists(path + '.png')):
            mask = image_to_ties(cv2.imread(path + '.png', 0), self.t_size, self.t_size)
        else:
            mask = np.zeros_like(patches[:, :, :, 0])
        return patches, mask

    def load_patch(self, patch_inx, inx):
        path = self.patches[patch_inx][inx]
        names = [int(i) for i in path[path.rfind('/') + 1:].split('_')]
        names = np.array(['%06d' % i for i in range(names[0], names[1] + 1)])
        patches, mask = self.load(path)
        return patches, mask, names

    def load_motion(self, patch_inx):
        patches, mask = self.load(self.motions[patch_inx])
        names = self.motion_names[patch_inx]
        patches = patches[:len(names)]
        mask = mask[:len(names)]
        return patches, mask, names

    def balance_tie(self, patch_inx, patches, mask, x, y, names):
        means = np.zeros(len(mask))
        means[mask[:, y, x] > 240] = 1
        if (means.sum() > self.max_m):
            bg_patch = patches[means < 0.5]
            bg_mask = mask[means  < 0.5]
            bg_names = names[means < 0.5]
            if(len(bg_mask) == 0):
                return None, None, None
            samples_to_add = int(means.sum()-self.max_m)
            inx = np.random.choice(np.arange(0, len(bg_mask)).astype(np.int32), samples_to_add)
            patches = np.concatenate((patches[means < 0.5], bg_patch[inx],patches[means >= 0.5][samples_to_add:]),
                                     axis=0)
            mask = np.concatenate((mask[means < 0.5], bg_mask[inx],mask[means >= 0.5][samples_to_add:]), axis=0)
            names = np.concatenate(( names[means < 0.5], bg_names[inx],names[means >= 0.5][samples_to_add:]), axis=0)
            return patches, mask, names
        elif (means.sum() < self.min_m):
            m_patch, m_mask, m_names = self.load_motion(patch_inx)
            m_patch = m_patch[m_mask[:, y, x] > 240]
            m_names = m_names[m_mask[:, y, x] > 240]
            m_mask = m_mask[m_mask[:, y, x] > 240]
            samples_to_add = int(self.min_m - means.sum())
            inx = np.random.choice(np.arange(0, len(m_mask)).astype(np.int32), samples_to_add)
            patches = np.concatenate((patches[means < 0.5][samples_to_add:], patches[means >= 0.5], m_patch[inx]),
                                     axis=0)
            mask = np.concatenate((mask[means < 0.5][samples_to_add:], mask[means >= 0.5], m_mask[inx]), axis=0)
            names = np.concatenate((names[means < 0.5][samples_to_add:], names[means >= 0.5], m_names[inx]), axis=0)
            return patches, mask, names
        else:
            return patches, mask, names

    def load_sample(self, patch_inx, inx=None):
        pos = tuple([int(i) for i in self.patches[patch_inx][inx].split('/')[-2].split('_')])
        hist = self.hist[patch_inx].copy()
        p = np.zeros_like(hist, dtype=np.float32).flatten()
        p[hist.flatten() >= self.min_m] = 1.
        y, x = np.unravel_index(int(np.random.choice(np.arange(hist.size), p=p / p.sum())), hist.shape)
        patches, mask, names = self.load_patch(patch_inx, inx)
        patches, mask, names = patches[:self.seq_l], mask[:self.seq_l], names[:self.seq_l]
        patches, mask, names = self.balance_tie(patch_inx, patches, mask, x + self.out_size // 2,
                                                y + self.out_size // 2, names)
        if(patches is None):
            return None,None,None,None
        pos = (pos[0] + x, pos[1] + y)
        return patches[:, y:y + self.out_size, x:x + self.out_size], mask[:, y:y + self.out_size,
                                                                     x:x + self.out_size], names, pos

    def get_position(self, patch_inx):
        path = self.motions[patch_inx]


def test_patch_loader(original_data, dir, patch_size, max_seq_l):
    o_size = int(np.random.randint(1, (patch_size - 1) // 2)) * 2 + 1
    seq_l = int(np.random.randint(5, np.sqrt(max_seq_l))) ** 2
    max_m = np.random.randint(1,seq_l)
    min_m = np.random.randint(0,max_m)
    min_m,max_m = min_m / float(seq_l), max_m / float(seq_l)
    pl = PatchLoader(dir, patch_size, seq_l,min_m,max_m , o_size)
    p_inx = np.random.randint(len(pl.patches))
    inx = np.random.randint(len(pl.patches[p_inx]))
    path = pl.patches[p_inx][inx]
    for i in range(10):
        im, mask, names, pos = pl.load_sample(p_inx, inx)
        if not (im is None):
            break

    o_imgs = []
    o_mask = []
    path = original_data+'/'.join(path[len(dir)+1:].split('/')[:2])
    for i in names:
        o_imgs.append(cv2.imread(path+'/input/in%s.jpg'%(i))[pos[1]:pos[1]+o_size,pos[0]:pos[0]+o_size])
        o_mask.append(cv2.imread(path+'/groundtruth/gt%s.png'%(i),0)[pos[1]:pos[1]+o_size,pos[0]:pos[0]+o_size])
    o_mask =np.array(o_mask)
    o_imgs = np.array(o_imgs)
    draw(o_imgs,o_mask)
    draw(im,mask)
    print np.array(o_mask.astype(np.float32)-mask.astype(np.float32)).sum()
    print np.array(o_imgs[:20].astype(np.float32)-im[:20].astype(np.float32)).sum()
    try:
        if (im is None):
            raise ValueError('always None')
        if (im.shape != (seq_l, o_size, o_size, 3) or mask.shape != (seq_l, o_size, o_size)):
            print 'im_shape', im.shape, ',mask_shape', mask.shape
            raise ValueError('wrong shape')
        if not (min_m <= (mask[:, o_size // 2, o_size // 2] > 240).mean() <= max_m):
            print 'balanse', (mask[:, o_size // 2, o_size // 2] > 240).mean(), np.ceil(min_m * seq_l)
            raise ValueError('wrong balanse')
    except Exception as e:
        print 'max_m=%f,min_m=%f,seq_l=%d,o_size=%d,p_inx=%d,inx=%d,path=%s' % (
        max_m, min_m, seq_l, o_size, p_inx, inx, path)
        if not (im is None):
            draw(im, mask)
        raise e

np.random.seed(0)
for i in range(100):
    print i
    test_patch_loader('data/test/', 'out', 32, 256)