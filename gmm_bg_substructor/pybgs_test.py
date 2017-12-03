import numpy as np
import cv2
import pybgs

params = { 
 	'algorithm': 'grimson_gmm', 
 	'low': 3.0 * 3.0,
 	'high': 3.0 * 3.0 * 2,
 	'alpha': 0.01,
 	'max_modes': 3,
 	'channels': 2,
	'variance': 36.,
	'bg_threshold': 0.75,
	'min_variance': 4.,
	'variance_factor': 5.}


bg_sub = pybgs.BackgroundSubtraction()	
camera_source = cv2.VideoCapture()
camera_source.open(0)

i = 0
error, img = camera_source.read()
img = img[:,:,:2].astype(np.float32)
high_threshold_mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
low_threshold_mask = np.zeros_like(high_threshold_mask)
bg_sub.init_model(img, params)

while cv2.waitKey(30) == -1:
    error, img = camera_source.read()
    img = img[:,:,:2].astype(np.float32)
    bg_sub.subtract(i, img, low_threshold_mask, high_threshold_mask)
    bg_sub.update(i, img, high_threshold_mask)
    cv2.imshow('foreground', low_threshold_mask)
    i += 1

 
