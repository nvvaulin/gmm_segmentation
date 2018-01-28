from pybgs import BackgroundSubtraction
import numpy as np

class BgSubstructor:
    def __init__(self,params):
        self.bg_sub = BackgroundSubtraction()
        self.init = False
        self.params = params
        self.i = 0 

    def update(self,features,img):
        img = img.astype(np.float32)
        features = features.astype(np.float32)
        high_threshold_mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
        low_threshold_mask = np.zeros_like(high_threshold_mask)
        if(not  self.init):
            self.bg_sub.init_model(features,img, self.params)
            self.init = True
        self.bg_sub.subtract(self.i,features, img, low_threshold_mask, high_threshold_mask)
        self.i = self.i+1
        return low_threshold_mask