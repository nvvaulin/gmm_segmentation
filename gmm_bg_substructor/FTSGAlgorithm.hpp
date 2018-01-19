/*
 * FTSGmethod.hpp
 *
 *  Created on: 21 lis 2015
 *      Author: Piotr Janus
 */
#ifndef SRC_FTSG_ALGORITHM_HPP_
#define SRC_FTSG_ALGORITHM_HPP_

#include <opencv2/core/core.hpp>
#include <vector>
#include "SplitGaussian.hpp"
#include "FluxTensorMethod.hpp"

using namespace cv;

class FTSGAlgorithm {
public:
	FTSGAlgorithm();
	~FTSGAlgorithm();
	cv::Mat flux;

	void update(const Mat & input_fetures,const Mat & input_img, Mat & background, Mat & foreground);

	void setSplitGaussianParam(double bgAlpha, double fgAlpha, double tb, double tf, double tl,double init_variance){
		splitGaussian = std::unique_ptr<SplitGaussian>(new SplitGaussian(bgAlpha, fgAlpha, tb, tf, tl,init_variance));
	}

	void setFluxTensorParam(double th, int nDs, int nDt, int nAs, int nAt){
		fluxTensor = std::unique_ptr<FluxTensorMethod>(new FluxTensorMethod(nDs, nDt, nAs, nAt, th));
	}

private:
	std::unique_ptr<SplitGaussian> splitGaussian;
	std::unique_ptr<FluxTensorMethod> fluxTensor;
	std::vector<std::vector< bool> > updateMask;

	void reset(const Mat & input_features,const Mat & input_img);

	void fusion(const Mat & flux, const Mat & background, const Mat & foreground, Mat & fusion, Mat & staticFg);

	void gaussianUpdate(const Mat & input, bool largeChange);

};

#endif /* SRC_FTSG_ALGORITHM_HPP_ */
