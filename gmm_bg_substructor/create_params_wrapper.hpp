#ifndef _CREATE_PARAMS_WRAPPER_HPP_
#define _CREATE_PARAMS_WRAPPER_HPP_

#include "GrimsonGMM.hpp"
#include "Image.hpp"
#include <cv.h>
#include <cxcore.h>

/**
 * @brief Updates ImageBase and IplImage data pointers.
 * 
 * @param updated_image 
 * @param updated_ipl_image
 * @param data_ptr 				data pointer
 * @param step 					step between adjacent rows in bytes
 */	
void set_image_data(
	ImageBase* updated_image, 
	IplImage* updated_ipl_image, 
	float* data_ptr, int step)
{
	cvSetData(updated_ipl_image, data_ptr, step);
	(*updated_image) = updated_ipl_image;
}
void set_image_data(
	ImageBase* updated_image, 
	IplImage* updated_ipl_image, 
	unsigned char* data_ptr, int step)
{
	cvSetData(updated_ipl_image, data_ptr, step);
	(*updated_image) = updated_ipl_image;
}

using namespace Algorithms::BackgroundSubtraction;

GrimsonParams CreateGrimsonGMMParams(int width, int height,
	float low_threshold, float high_threshold, 	
	float alpha, float max_modes,int channels,float bg_threshold, float variance,float min_variance,float variance_factor)
{
	Algorithms::BackgroundSubtraction::GrimsonParams params;
	params.SetFrameSize(width, height);
	params.LowThreshold() = low_threshold;
	params.HighThreshold() = high_threshold;
	params.Alpha() = alpha;
	params.MaxModes() = max_modes;
	params.Channels() = channels;
	params.Variance() = variance;
	params.BgThreshold() = bg_threshold;
	params.VarianceFactor() = variance_factor;
	params.MinVariance() = min_variance;
	return params;
}
#endif
