#include "ForegroundGaussian.hpp"

double ForegroundGaussian::ALPHA = 0;
double ForegroundGaussian::T_F = 0;

#include <iostream>

void ForegroundGaussian::initialise(double *gaussian_mean)
{
    for(int i = 0; i < num_channels; ++i)
    {
        mean[i] = gaussian_mean[i];
    }
    initFlag = true;
}

void ForegroundGaussian::getMean(double * gaussianMean)
{
    for(int i = 0; i < num_channels; ++i)
    {
        gaussianMean[i]= mean[i];
    }
}

void ForegroundGaussian::update(double *rgb)
{
	if(!initFlag){
		initialise(rgb);
		return;
	}

    for(int i = 0; i < num_channels; ++i)
    {   mean[i] *= (1 - ALPHA);
        mean[i] += ALPHA*rgb[i];
    }
}

bool ForegroundGaussian::checkPixelMatch(double *rgb) // max((I-u)-mean)^2 - T_F^2)
{
	if(!initFlag)
		return false;

	double dist = malahidanDistance(rgb, &mean[0], num_channels);
	return dist < T_F*T_F;
}
