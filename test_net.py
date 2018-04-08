from test_tools import test_network
from networks import baseline_norm,conv4_net,conv4_net_dense,conv4_net_dense_color
import argparse
from easydict import EasyDict

parser = argparse.ArgumentParser(description='test network.')
parser.add_argument('--epoch', type=int)
parser.add_argument('--network',type=str)
parser.add_argument('--prefix',type=str)
parser.add_argument('--ndim',type=int)
parser.add_argument('--gm_num',type=int,default=4)
args = parser.parse_args()

name=args.prefix+args.network+'%d'%(args.ndim)
test_network(name,args.network,args.ndim,args.epoch,args.gm_num,im_size=(320,240),train_size=100,test_size=400)