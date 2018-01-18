import numpy as np
import cv2
import pybgs

#params = { 
# 	'algorithm': 'grimson_gmm', 
# 	'low': 3.0 * 3.0,
# 	'high': 3.0 * 3.0 * 2,
# 	'alpha': 0.01,
# 	'max_modes': 3,
# 	'channels': 2,
#	'variance': 36.,
#	'bg_threshold': 0.75,
#	'min_variance': 4.,
#	'variance_factor': 5.}



params = { 
 	'algorithm': 'FTSG', 
	'th': 30, 
	'nDs': 5,
	'nDt': 5,
	'nAs': 5,
	'nAt': 5,
	'bgAlpha': 0.004,
	'fgAlpha': 0.5,
	'tb': 3,
	'tf': 20,
	'tl': 0.1,
	'init_variance': 15
}


bg_sub = pybgs.BackgroundSubtraction()	
camera_source = cv2.VideoCapture()
camera_source.open(0)

i = 0
error, img = camera_source.read()
img = img.astype(np.float32)
high_threshold_mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
low_threshold_mask = np.zeros_like(high_threshold_mask)
bg_sub.init_model(img[:,:,:2],img, params)

while cv2.waitKey(30) == -1:
    error, img = camera_source.read()
    img = img.astype(np.float32)
    bg_sub.subtract(i, img[:,:,:2],img, low_threshold_mask, high_threshold_mask)
    cv2.imshow('foreground1', low_threshold_mask)
    i += 1

 
