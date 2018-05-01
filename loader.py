import os
import cv2
import numpy as np
from dataset_tools import iterate_folders,iterate_bathced,make_path,draw,image_to_patches,patches_to_image

def images_to_patches(imgs,cols,rows,h,w):
    '''
    samples x rows*h x cols*w x c -> patches x samples x 
    '''
    assert len(imgs.shape) == 4
    c = imgs.shape[-1]
    return np.transpose(imgs.reshape((-1,rows,h,cols,w,c)),(1,3,0,2,4,5)).reshape(cols*rows,len(imgs),h,w,c)
    
def process_batch(imgs,masks,w,h,min_roi):
    '''
    return 
    patches - n_samples x length x h x w x 3
    patches_mask - n_samples x length x h x w
    positions - n_samples x [x,y]
    '''
    cols = imgs.shape[2]//w
    rows = imgs.shape[1]//h
    imgs = imgs[:,:h*rows,:w*cols,:]
    masks = masks[:,:h*rows,:w*cols]
    imgs = images_to_patches(imgs,cols,rows,h,w)
    masks = images_to_patches(masks.reshape(masks.shape+(1,)),cols,rows,h,w)[...,0]
    positions = np.zeros((1,rows*h,cols*w,2),dtype=np.int32)
    positions[0,:,:,0] += np.arange(cols*w)[None,:]
    positions[0,:,:,1] += np.arange(rows*h)[:,None]
    positions = images_to_patches(positions,cols,rows,h,w)
    positions = positions[:,0,0,0,:]
    roi = np.zeros_like(masks)
    roi[(masks > 240) | (masks < 10)] = 1
    roi = roi.mean((1,2,3))
    imgs,masks,positions = imgs[roi>=min_roi],masks[roi>=min_roi],positions[roi>=min_roi]
    return imgs,masks,positions

def get_motion(patches,masks,positions,min_motion,names):
    '''
    return n_samples list of
    patches - n_motion_patches x h x w x 3
    patches_mask - n_motion_patches x h x w
    positions - n_motion_patches x [x,y] positions of patches
    index - n_motion_patches index of patches
    '''
    motion_mask = np.zeros_like(masks)
    motion_mask[(masks > 240)] = 1
    motion_mask = motion_mask.mean((2,3))
    motion = []
    names = np.array(names)
    for i in range(len(masks)):
        m = motion_mask[i] >= min_motion
        p = (positions[i][0],positions[i][1])
        motion.append((p,patches[i][m],masks[i][m],names[m]))
    return motion

def generate_patches(dataset,imdb,length,patch_size,min_motion,min_roi):
    def save_motion_hist(key,hist):
        key = imdb+key
        make_path(key[:key.rfind('/')])
        cv2.imwrite(key+'hist.png',hist)
    def save(key,img,mask):
        key = imdb+key
        make_path(key[:key.rfind('/')])
        img = patches_to_image(img)
        cv2.imwrite(key+'.jpg',img)
        if not(mask is None):
            mask = patches_to_image(mask)
            cv2.imwrite(key+'.png',mask)
            
    for path in iterate_folders(dataset):
        print 'processing',path
        all_motion = dict()
        in_dir = path[len(dataset):]
        for batch_num,(names,imgs,masks) in enumerate(iterate_bathced(path,length)):
            patches, masks,positions = process_batch(imgs,masks,patch_size,patch_size,min_roi)
            if(len(patches) == 0):
                continue
            for t,m,p in zip(patches,masks,positions):                    
                key = '%s/%d_%d/%s_%s'%(in_dir,p[0],p[1],names[0],names[-1])
                save(key,t, None if m[m>30].size == 0 else m)
                
            motion = get_motion(patches,masks,positions,min_motion,names)
            for position,m_patch,m_mask,m_names in motion:
                if not(position in all_motion):
                    all_motion[position] = ([],[],[])
                all_motion[position][0].append(m_patch)
                all_motion[position][1].append(m_mask)
                all_motion[position][2].append(m_names)
            
        for position,(patches,mask,names) in all_motion.iteritems():
            patches = np.concatenate(tuple(patches))
            mask = np.concatenate(tuple(mask))
            names  = np.concatenate(tuple(names))
            key = '%s/%d_%d/'%(in_dir,position[0],position[1])
            if(len(patches) > 0):
                motion_info = open(imdb+key+'motion_info.txt','w')
                for n in names:
                    motion_info.write(n+'\n')
                motion_info.close()
                save(key+'motion',patches,mask)
                motion_hist = np.zeros_like(mask,dtype=np.int32)
                motion_hist[mask > 240] = 1
                motion_hist = np.clip(motion_hist.sum(0),0,255).astype(np.uint8)
                save_motion_hist(key,motion_hist)
                

class IMDB:
    def __init__(self):
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

class PatchLoader(object):
    def __init__(self, root, t_size, seq_l, min_m, max_m, out_size,cashe_samples=False):
        assert (out_size // 2) * 2 + 1 == out_size, 'out_size must be odd'
        self.root, self.seq_l, self.out_size = root, seq_l, out_size
        self.cashe_samples = cashe_samples
        self.cashed_data = IMDB()
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

    def load(self, path,cashe=False):
        if(cashe):
            patches = self.cashed_data[path + '.jpg']
        else:
            patches = cv2.imread(path + '.jpg')
        patches = image_to_patches(patches, self.t_size, self.t_size)
        if (os.path.exists(path + '.png')):
            if(cashe):
                mask = self.cashed_data[path + '.png']
            else:
                mask = cv2.imread(path + '.png', 0)
            mask = image_to_patches(mask,self.t_size, self.t_size)
        else:
            mask = np.zeros_like(patches[:, :, :, 0])
        return patches, mask

    def load_patch(self, patch_inx, inx):
        path = self.patches[patch_inx][inx]
        names = [int(i) for i in path[path.rfind('/') + 1:].split('_')]
        names = np.array(['%06d' % i for i in range(names[0], names[1] + 1)])
        patches, mask = self.load(path,self.cashe_samples)
        return patches, mask, names

    def load_motion(self, patch_inx):
        patches, mask = self.load(self.motions[patch_inx],True)
        names = self.motion_names[patch_inx]
        patches = patches[:len(names)]
        mask = mask[:len(names)]
        return patches, mask, names

    def balance_patch(self, patch_inx, patches, mask, x, y, names):
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
        patches, mask, names = self.balance_patch(patch_inx, patches, mask, x + self.out_size // 2,
                                                y + self.out_size // 2, names)
        if(patches is None):
            return None,None,None,None
        pos = (pos[0] + x, pos[1] + y)
        return patches[:, y:y + self.out_size, x:x + self.out_size], mask[:, y:y + self.out_size,
                                                                     x:x + self.out_size], names, pos

    def iterate(self,shuffle=False):
        img_list = np.arange(len(self.patches)).astype(np.int32)
        if(shuffle):
            np.random.shuffle(img_list)
        for patch_inx in img_list:
            inx = int(np.random.randint(len(self.patches[patch_inx])))
            patches,mask,names,pos = self.load_sample(patch_inx,inx)
            if(patches is None):
                continue
            yield patches,mask

def data_generator(gmm_loader,
                   epoch_size,
                   shuffle=False):
    for i,(patches,mask) in enumerate(gmm_loader.iterate(shuffle)): 
        if(i >= epoch_size):
            break
        patches = np.transpose(patches,(0,3,1,2)).astype(np.float32)
        mask = mask.astype(np.float32)/255.
        mask[mask <= 0.1] = 0
        mask[mask >= 0.9] = 1.
        mask[(mask < 0.9)&(mask > 0.1)] = 0.5
        yield patches,mask
        