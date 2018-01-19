#ifndef FTSG_
#define FTSG_

#include "Bgs.hpp"
#include "FTSGAlgorithm.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

namespace Algorithms
{
namespace BackgroundSubtraction
{


class FTSGParams : public BgsParams
{

public:
	struct GMMParams{
		double bgAlpha;
		double fgAlpha;
		double tb;
		double tf;
		double tl;
		double init_variance;
	};

	struct FLUXParams{
		double th;
		int nDs;
		int nDt;
		int nAs;
		int nAt;
	};

	FTSGParams::GMMParams getGMMParams() const{
		return gmmParams;
	};
	FTSGParams::FLUXParams getFLUXParams() const{
		return fluxParams;
	};
	void SetFLUXParams(double th, int nDs, int nDt, int nAs, int nAt){
		fluxParams.th = th;
		fluxParams.nDs = nDs;
		fluxParams.nDt = nDt;
		fluxParams.nAs = nAs;
		fluxParams.nAt = nAt;
	}
	void SetGMMPatams(double bgAlpha, double fgAlpha, double tb, double tf, double tl,float init_variance){
		gmmParams.bgAlpha = bgAlpha;
		gmmParams.fgAlpha = fgAlpha;
		gmmParams.tb = tb;
		gmmParams.tf = tf;
		gmmParams.tl = tl;
		gmmParams.init_variance = init_variance;
	}

private:
	FLUXParams fluxParams;
	GMMParams gmmParams;
};

class FTSG : public Bgs
{

	FTSGAlgorithm algorithm;
public:
	FTSG();
	~FTSG();

	void Initalize(const BgsParams& param);
	void InitModel(const Image& data);
	void Update(int frame_num, const Image& data,  const BwImage& update_mask);
	void Subtract(int frame_num, const Image& data,const Image& image,  BwImage& low_threshold_mask, BwImage& high_threshold_mask);	
};

};
};

#endif






