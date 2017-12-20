# distutils: language = c++
# distutils: sources = FTSGAlgorithm.cpp tools.cpp FluxTensorMethod.cpp BackgroundGaussian.cpp ForegroundGaussian.cpp Pixel.cpp SplitGaussian.cpp Image.cpp GrimsonGMM.cpp FTSG.cpp
# distutils: libraries = opencv_core opencv_highgui opencv_features2d opencv_imgproc opencv_photo opencv_superres opencv_ts 

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp cimport bool

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef extern from "types_c.h":
    # C MACRO defined as an external int variable
    cdef int IPL_DEPTH_8U
    cdef int IPL_DEPTH_32F

    ctypedef struct IplImage:
        char *imageData
        int  widthStep
        int width
        int height

    cdef struct CvSize:
        int width
        int height

cdef extern from "core_c.h":
    IplImage *cvCreateImageHeader(CvSize size, int depth, int channels)
    void cvReleaseImageHeader(IplImage** image)
    void cvSetData(IplImage * arr, void* data, int step)

cdef extern from "Image.hpp":
    cdef cppclass ImageBase:
        ImageBase()
        ImageBase(IplImage* img)
        void ReleaseMemory(bool b)
        IplImage* Ptr()        

    cdef cppclass Image(ImageBase):
        Image()
        Image(IplImage* img)      

    cdef cppclass BwImage(ImageBase):
        BwImage()
        BwImage(IplImage* img)
        

cdef extern from "Bgs.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass Bgs:
        void InitModel(const Image& data)

        void Initalize(const BgsParams& param)

        void Subtract(int frame_num, const Image& data, const Image& image,
                      BwImage& low_threshold_mask, BwImage& high_threshold_mask)
        void Update(int frame_num, const Image& data,
                    const BwImage& update_mask)

cdef extern from "BgsParams.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass BgsParams:
        pass

cdef extern from "GrimsonGMM.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass GrimsonParams(BgsParams):
        pass
    cdef cppclass GrimsonGMM(Bgs):
        pass  


cdef extern from "FTSG.hpp" namespace "Algorithms::BackgroundSubtraction":
    cdef cppclass FTSGParams(BgsParams):
        pass
    cdef cppclass FTSG(Bgs):
        pass  

cdef extern from "create_params_wrapper.hpp":
    GrimsonParams CreateGrimsonGMMParams(int width, int height,
    float low_threshold, float high_threshold, float alpha, float max_modes,int channels,float bg_threshold, float variance,float min_variance,float variance_factor)

    FTSGParams CreateFTSGParams(int width, int height,
				float th, int nDs, int nDt, int nAs, int nAt,
				float bgAlpha, float fgAlpha, float tb, float tf, float tl)

    void set_image_data(ImageBase* updated_image, IplImage* updated_ipl_image, 
                        float* data_ptr, int step)

    void set_image_data(ImageBase* updated_image, IplImage* updated_ipl_image, 
                        unsigned char* data_ptr, int step)


cdef class BackgroundSubtraction:
    cdef Bgs* bg
    cdef IplImage* iplimage
    cdef IplImage* ipldata
    cdef IplImage* low_mask_iplimage
    cdef IplImage* high_mask_iplimage
    cdef Image image
    cdef Image data
    cdef BwImage low_mask_image
    cdef BwImage high_mask_image

    def __init__(self):
        self.image.ReleaseMemory(False)
        self.data.ReleaseMemory(False)
        self.low_mask_image.ReleaseMemory(False)
        self.high_mask_image.ReleaseMemory(False)

    def __dealloc__(self):
        cvReleaseImageHeader(&self.ipldata)
        cvReleaseImageHeader(&self.iplimage)
        cvReleaseImageHeader(&self.low_mask_iplimage)
        cvReleaseImageHeader(&self.high_mask_iplimage)
        del self.bg

    def init_model(self,np.float32_t[:, :, :] data, np.float32_t[:, :, :] image, params):
        if(params['algorithm'] == 'FTSG'):
            self.bg = new FTSG()
            ftsg_params = CreateFTSGParams(
	    	image.shape[1], image.shape[0], 
	    	params['th'], params['nDs'], params['nDt'], params['nAs'], params['nAt'], 
	    	params['bgAlpha'], params['fgAlpha'],params['tb'],params['tf'],params['tl'])
            self.bg.Initalize(ftsg_params)
        else:            
            self.bg = new GrimsonGMM()
            grimson_gmm_params = CreateGrimsonGMMParams(
	    	image.shape[1], image.shape[0], 
	    	params['low'], params['high'], 
	    	params['alpha'], params['max_modes'],params['channels'],params['bg_threshold'],params['variance'],params['min_variance'],params['variance_factor'])
            self.bg.Initalize(grimson_gmm_params)
        
        cdef CvSize size
        size.width = image.shape[1]
        size.height = image.shape[0]
        self.iplimage = cvCreateImageHeader(size, IPL_DEPTH_32F, image.shape[2])
        self.ipldata = cvCreateImageHeader(size, IPL_DEPTH_32F, data.shape[2])
        self.low_mask_iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 1)
        self.high_mask_iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 1)

        set_image_data(&self.image, self.iplimage, 
                       &image[0, 0, 0], image.strides[0])

        set_image_data(&self.data, self.ipldata, 
                       &data[0, 0, 0], data.strides[0])
        self.bg.InitModel(self.data)

    def subtract(self, frame_num, np.float32_t[:, :, :] data, np.float32_t[:, :, :] image, 
                 np.uint8_t[:, :] low_threshold_mask, 
                 np.uint8_t[:, :] high_threshold_mask):

        set_image_data(<ImageBase *>(&self.image), self.iplimage, 
                       &image[0, 0, 0], image.strides[0])
        set_image_data(<ImageBase *>(&self.data), self.ipldata, 
                       &data[0, 0, 0], data.strides[0])
        set_image_data(<ImageBase *>(&self.low_mask_image), self.low_mask_iplimage, 
                       &low_threshold_mask[0, 0], low_threshold_mask.strides[0])
        set_image_data(<ImageBase *>(&self.high_mask_image), self.high_mask_iplimage, 
                       &high_threshold_mask[0, 0], high_threshold_mask.strides[0])
        self.bg.Subtract(frame_num, self.data, self.image, self.low_mask_image, self.high_mask_image)

