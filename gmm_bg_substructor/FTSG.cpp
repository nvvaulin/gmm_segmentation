
#include "FTSG.hpp"
#include <iostream>

using namespace Algorithms::BackgroundSubtraction;

FTSG::FTSG()
{}

FTSG::~FTSG()
{}

void FTSG::Initalize(const BgsParams& _param)
{
	const FTSGParams& param = (FTSGParams&)_param; 
	FTSGParams::GMMParams p = param.getGMMParams();
	algorithm.setSplitGaussianParam(p.bgAlpha, p.fgAlpha, p.tb, p.tf, p.tl,p.init_variance);
	FTSGParams::FLUXParams f = param.getFLUXParams();
	algorithm.setFluxTensorParam(f.th, f.nDs, f.nDt, f.nAs, f.nAt);
}


void FTSG::InitModel(const Image& data)
{}

void FTSG::Update(int frame_num, const Image& data,  const BwImage& update_mask)
{}

///////////////////////////////////////////////////////////////////////////////
//Input:
//  data - a pointer to the data of a RGB image of the same size
//Output:
//  output - a pointer to the data of a gray value image of the same size 
//					(the memory should already be reserved) 
//					values: 255-foreground, 125-shadow, 0-background
///////////////////////////////////////////////////////////////////////////////
void FTSG::Subtract(int frame_num, const Image& features,const Image& image, BwImage& low_threshold_mask, BwImage& high_threshold_mask)
{
	cv::Mat image_mat(image.Ptr(),true);
	image_mat.convertTo(image_mat,CV_8UC3);
	cv::Mat features_mat(features.Ptr(),true);

	cv::Mat background(low_threshold_mask.Ptr(),false);
	cv::Mat foreground(high_threshold_mask.Ptr(),false);
	algorithm.update(features_mat,image_mat,result,background,foreground,flux);
//	cv::imshow("input",input_img);
//	cv::imshow("result",result);
//	cv::imshow("foreground",foreground);
//	cv::imshow("flux",flux);
//	cv::waitKey(10);
	
}

