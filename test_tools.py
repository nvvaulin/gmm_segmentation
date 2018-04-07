import os
import cv2
import numpy as np
#from gmm_bg_substructor import BgSubstructor
from sklearn import mixture
from sklearn.metrics import average_precision_score
from dataset_tools import *
from gmm_op import get_gmm,calc_log_prob_gmm
import theano.tensor as T
from lasagne import layers as L
import theano
from multiprocessing import Pool
from utils import tee
from networks import make_FCN
from sklearn.metrics import precision_recall_curve

def calc_metrics_imgs(predict,label):
    predict,label = predict.flatten(),label.flatten()
    mask = (label>230)|(label < 50)
    p = (predict.astype(np.float32)/255.)[mask]
    y = (label.astype(np.float32)/255.)[mask]
    y[y>0.5] = 1.
    y[y<=0.5] = 0.
    bp = np.zeros_like(p)
    bp[p>0.5] = 1.
    TP = (bp*y).sum()
    TN = ((1-bp)*(1-y)).sum()
    FP = (bp*(1.0-y)).sum()
    FN = ((1.0-bp)*y).sum()
    AveragePrecision = 0#average_precision_score(y,p)
    return np.array([TP,TN,FP,FN],dtype=np.int64),AveragePrecision
    

def print_results(results):
    s = ''
    for k in results.keys():
        s=s+k+str(': ')+str(results[k])+'\n'
    return s

def calc_metrics_folder(data_dir):
    S = np.array([0,0,0,0],dtype=np.int64)
    AveragePrecision = 0.0
    nums = [int(i[:-4]) for i in os.listdir(data_dir) if i.find('true') < 0 and  i.find('input') < 0]
    for i in nums:
        m = cv2.imread(data_dir+'/%i_true.png'%(i))
        p = cv2.imread(data_dir+'/%i.png'%(i))
        _s = calc_metrics_imgs(p,m)
        AveragePrecision += _s[1]
        S+=_s[0]
        
    TP,TN,FP,FN = S[0],S[1],S[2],S[3]
    if(S.min() <= 0):
        results = dict( AveragePrecision = 0,\
                        Recall = np.nan,\
                        Sp = np.nan,\
                        FPR = np.nan,\
                        FNR = np.nan,\
                        PWC =  np.nan,\
                        F_Measure  =  np.nan,\
                        Precision  = np.nan)
    else:
        results = dict( AveragePrecision = AveragePrecision/float(len(nums)),\
                        Recall = TP / float(TP + FN),\
                        Sp = TN / float(TN + FP),\
                        FPR = FP / float(FP + TN),\
                        FNR = FN / float(TP + FN),\
                        PWC =  100 * (FN + FP) / float(TP + FN + FP + TN),\
                        F_Measure  =  (2 * (TP / float(TP + FP)) * (TP / float(TP + FN))) / (TP / float(TP + FP) +  TP / float(TP + FN)),\
                        Precision  = TP / float(TP + FP))

    print data_dir
    print print_results(results)
    return results

def calc_metric_all_folders(data_dir):
    res = []
    f = open(data_dir+'.txt','w')
    list_dirs = []
    for j in os.listdir(data_dir):
        if(os.path.isdir(data_dir+'/'+j)):
            list_dirs = list_dirs+[data_dir+'/'+j+'/'+i for i in os.listdir(data_dir+'/'+j)]
    
    for folder in list_dirs:
        results = calc_metrics_folder(folder)
        if not (results is None):
            res.append(results)
            f.write(folder+'\n')
            f.write(print_results(results))
    results = dict()
    for k in res[0].keys():
        results[k] = np.array([i[k] for i in res if np.isfinite(i[k])]).mean()
    f.write('total result\n')
    f.write(print_results(results))
    f.close()
    print 'total result'
    print print_results(results)

def soft_predict_sym(features,means,covars,weights):
    return 1.-T.nnet.sigmoid(calc_log_prob_gmm(features,means,covars,weights))

def make_features(feature_fn,imgs):
    data = None
    for i in range(len(imgs)):
        tmp = feature_fn(np.transpose(imgs[i:i+1],(0,3,1,2)).astype(np.float32))[0]
        if(data is None):
            data = np.empty((len(imgs),)+tmp.shape,dtype=np.float32)
        data[i] = tmp
    return data

def map_fit_gmm(args):
    if(args[0] is None):
        return None
    else:
        return args[0].fit(args[1])
    
def fit_gmms(features,labels,gm_num,min_samples_for_gmm,pool):
    args = []
    for i in range(features.shape[0]):
        f = features[i][labels[i] < 30]
        if(len(f) > min_samples_for_gmm):
            gmm = mixture.GaussianMixture( covariance_type='diag',
                                           n_components=gm_num,
                                           max_iter=1000,
                                           warm_start=False)
        else:
            gmm = None
        args.append((gmm,f))
    gmms = pool.map(map_fit_gmm,args)
    return gmms

def predict(features,gmms,predict_fn):
    res = np.zeros(features.shape[:2],dtype=np.float32)
    for i in range(features.shape[0]):
        gmm = gmms[i]
        if (not (gmm is None)):
            res[i] = predict_fn(features[i],gmm.means_,gmm.covariances_,gmm.weights_)
        else:
            res[i] = -1
    return res

    
def calc_aps(predicted,labels):
    labels = labels.flatten()
    predicted = predicted.flatten()
    mask = ((labels < 30) | (labels > 240)) & (predicted >= 0.)
    true = np.zeros_like(labels,dtype=np.float32)
    true[labels > 240] = 1.
    if(len(true[true > 0.9]) == 0):
        true[mask][0] = 1.
        predicted[mask][0] = 0.5
    return average_precision_score(true[mask],predicted[mask])

def fit_and_predict(imgs,masks,feature_fn,predict_fn,train_size,gm_num,pool,min_samples_for_gmm=50):
    data = make_features(feature_fn,imgs)
    flat_data = np.transpose(data,(1,2,0,3)).reshape((-1,data.shape[0],data.shape[-1]))
    flat_masks = np.transpose(masks,(1,2,0)).reshape((-1,masks.shape[0]))
    gmms = fit_gmms(flat_data[:,:train_size],flat_masks[:,:train_size],
                    gm_num=gm_num,
                    pool=pool,
                    min_samples_for_gmm=min_samples_for_gmm)
    prediction = predict(flat_data[:,train_size:],gmms,predict_fn)
    prediction = np.transpose(prediction,(1,0)).reshape(masks[train_size:].shape)
    return prediction


def make_test_as_train(feature_fn,predict_fn,
                       gm_num,
                       out_dir,
                       dataset='dataset',
                       max_frames=200,
                       im_size = (320,240),
                       train_size = 100):
    try:
        os.mkdir(out_dir)
    except:
        pass
    aps_log = open(out_dir+'/aps.txt','w')
    pool = Pool(4)
    print 'run'
    for in_dir,out_dir in iterate_folders(dataset,out_dir):
        print in_dir
        for i,(imgs,masks) in enumerate(iterate_bathced(in_dir,max_frames,im_size)):
            print 'process dir ' + in_dir
            if((masks[(masks>30) & (masks < 240)].size > 0.9*masks.size) or
               (masks[(masks>240)].size < 10)):
                print 'skip'
                continue
            prediction = fit_and_predict(imgs,masks,feature_fn,predict_fn,train_size,gm_num,pool=pool)
            aps = calc_aps(prediction,masks[train_size:])
            tee(str(in_dir)+' aps = '+str(aps),aps_log)
            print 'save to '+out_dir
            imgs = imgs[train_size:]
            masks = masks[train_size:]
            threshold = 0.5
            prediction[prediction > threshold] = 1.
            prediction[prediction <= threshold] = 0.
            prediction = (prediction*255).astype(np.uint8)
            for i in range(len(imgs)):
                cv2.imwrite(out_dir+'/'+str(i)+'.png',prediction[i])
                cv2.imwrite(out_dir+'/'+str(i)+'_true.png',masks[i])
                cv2.imwrite(out_dir+'/'+str(i)+'_input.jpg',imgs[i])
            break
        print ''
        break
    print 'test complete'

    

def test_network(name,network,ndim,epoch,gm_num,im_size=(320,240),train_size=100,test_size=300):
    data=T.tensor4()
    feature_net = make_FCN(network,
                           data=data,
                           ndim=ndim,
                           model_name='%s%03d'%(name,epoch) if epoch >= 0 else '',
                           input_shape=(1,3,im_size[1],im_size[0]))
    feature_sym = L.get_output(feature_net,deterministic=True)
    feature_fn = theano.function([data],feature_sym,allow_input_downcast=True)
    data,m,c,w=T.matrix(),T.matrix(),T.matrix(),T.vector()
    predict_fn = theano.function([data,m,c,w],soft_predict_sym(data,m,c,w),allow_input_downcast=True)
    make_test_as_train(feature_fn,predict_fn,
                       out_dir='results/'+name,
                       dataset='data/test',
                       gm_num=gm_num,
                       max_frames=train_size+test_size,
                       train_size=train_size,
                       im_size=im_size)
    calc_metric_all_folders('results/'+name)
    
    
def find_gmm_params(feature_fn,
              dataset='../gmm_segmentation/test_dataset',
              max_frames=300,
              im_size = (320//2,240//2),gm_num=4):
    all_covars = []
    all_weights = []
    all_comp = []
    all_masks = []
    for k,in_dir in enumerate(iterate_folders(dataset)):
        for i,(imgs,masks) in enumerate(iterate_bathced(in_dir,max_frames,im_size)):
            print in_dir,'generate_features,',
            data = make_features(feature_fn,imgs)
            gmms = make_gmms(imgs.shape[1:-1],gm_num)
            print 'fit gmms'
            fit_gmms(data,gmms,None)
            cov = np.empty((len(gmms),)+gmms[0].covariances_.shape,dtype=np.float32)
            weights = np.empty((len(gmms),len(gmms[0].weights_)),dtype=np.float32)
            dists = np.empty((len(gmms),len(imgs),len(gmms[0].weights_)),dtype=np.float32)
            data = data.reshape((len(imgs),-1,gmms[i].means_.shape[1]))
            for j in range(len(gmms)):
                cov[j] = gmms[j].covariances_
                weights[j] = gmms[j].weights_
                dists[j] = ((data[:,j,None,:]-gmms[j].means_[None,:,:])**2).sum(-1)
            print 'cov',cov.mean(),cov.std()
            masks = masks.reshape((len(masks),-1))
            masks = np.transpose(masks,(1,0))
            print 'cov bg',dists[masks < 0.5].min(-1).mean(),dists[masks < 0.1].min(-1).std()
            print 'cov motion',dists[masks > 0.5].min(-1).mean(),dists[masks > 0.9].min(-1).std()
            print 'std motion',data[masks > 0.9].std()
            break
    print 'test complete'
    
    
    
def make_bgs_test(feature_fn,params,out_name,dataset='test_dataset',max_l = 1000,im_size=None,verbose=False,skip_frames=200):
    def make_bgs_features(feature_fn,x):
        return feature_fn(np.transpose(x,(0,3,1,2)))[0]
    out_dir = 'results/'+out_name+'_'+params['algorithm']
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    f = open('params.txt','w')
    f.write('params\n'+str(params)+'\n')
    f.write('max_l = %i,skip_frames=%i')
    f.close()
    jj = 0
    try:
        for d_in,d_out in iterate_folders(dataset,out_dir):
            jj+=1
            if(jj<1):
                continue
            try:
                bgs = BgSubstructor(params)
                i = 0
                prev =None
                for im,mask in iterate_video(d_in,skip_first_unlabled=True):
                    print '%s %d                   \r'%(d_in,i),
                    if not (im_size is None):
                        im,mask = resize(im,mask,im_size)
                    if(prev is None):
                        prev = im
                        continue
                    #tmp = np.concatenate((im[np.newaxis],prev[np.newaxis]),0)
                    features = make_bgs_features(feature_fn,im.astype(np.float32))
                    prev = im
                    pred = bgs.update(features.astype(np.float32),im.astype(np.float32))
                    cv2.imwrite(d_out+'/%d.png'%(i),pred)
                    cv2.imwrite(d_out+'/%d_true.png'%(i),mask)
                    cv2.imwrite(d_out+'/%d_input.png'%(i),im)
                    if(verbose):
                        cv2.imshow('pred',pred)
                        cv2.imshow('true',mask)
                        cv2.imshow('input',im)
                        cv2.waitKey(1)
                    if(i >= max_l):
                        break
                    i+=1
                print '%s %d\r'%(d_in,i),
            finally:
                del bgs
    finally:
        if(verbose):
            cv2.destroyAllWindows()
    print 'done'
    
# params = { 'algorithm': 'grimson_gmm', 
#             'low': 1.,#*24*24,
#             'high': 3,#.*24*24,
#             'alpha': 0.01,
#             'max_modes': 5,
#             'channels': 12,
#             'variance': .01,
#             'bg_threshold': 0.075,
#             'min_variance': .005,
#             'variance_factor': 1.}    
    
# params = { 
#     'algorithm': 'FTSG', 
#     'th': 30, 
#     'nDs': 5,
#     'nDt': 5,
#     'nAs': 5,
#     'nAt': 5,
#     'bgAlpha': 0.004,
#     'fgAlpha': 0.05,
#     'tb': 15,
#     'tf': 0.00001,
#     'tl': 0.01,
#     'init_variance' : 0.01
# } 
# make_bgs_test(feature_fn,
#                out_name=cfg.NAME,
#                params=params,
#                dataset='../gmm_segmentation/test_dataset',
#                im_size=(320,240),
#                verbose=True)


# params = { 
#     'algorithm': 'FTSG', 
#     'th': 30, 
#     'nDs': 5,
#     'nDt': 5,
#     'nAs': 5,
#     'nAt': 5,
#     'bgAlpha': 0.004,
#     'fgAlpha': 0.5,
#     'tb': 4,
#     'tf': 20,
#     'tl': 0.1,
#     'init_variance': 15
# }
# make_bgs_test(lambda x : np.transpose(x,(0,2,3,1)),
#                params,
#                out_name='baseline',
#                dataset='../gmm_segmentation/test_dataset',
#                im_size=(320,240),
#                verbose=True)