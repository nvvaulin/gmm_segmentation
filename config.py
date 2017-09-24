from easydict import EasyDict
import numpy as np

cfg = EasyDict()
cfg.LABEL_VALUES = [0,50,85,170,255]
cfg.LABELS = [0,0,0,0,1]
cfg.LABEL_NUM = len(np.unique(np.array(cfg.LABELS)))
cfg.SEQ_LENGTH = 3
cfg.IS_COLOR  = True
cfg.MOTION_RATE = 0.9
cfg.TILE_SIZE = (200,200)
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.EPOCH_SIZE = 100