#include <iostream>

#include "tools.hpp"
#include "BackgroundGaussian.hpp"

using namespace std;

double BackgroundGaussian::INITIAL_VARIANCE = 0;
double BackgroundGaussian::ALPHA = 0;
double BackgroundGaussian::T_B = 0;


void BackgroundGaussian::initialise(double weight, double *gaussianMean, double stdDeviation)
{
    this->weight = weight;
    for(int i = 0; i < num_channels; ++i)
    {
        mean[i] = gaussianMean[i];
    }
    this->stdDeviation = stdDeviation;
}

void BackgroundGaussian::getMean(double * gaussianMeans)
{
    for(int i = 0; i < num_channels; ++i)
    {
        gaussianMeans[i] = mean[i];
    }
}

void BackgroundGaussian::updateUnmatched()
{
    weight *= (1 - ALPHA);
}

void BackgroundGaussian::updateMatched(double *rgb)
{
    //Weight
    weight *=  (1 - ALPHA);
    weight += ALPHA;

    //RGB means
    for(int i = 0; i < num_channels; ++i)
    {   mean[i] *= (1 - ALPHA);
        mean[i] += ALPHA*rgb[i];
    }

    //Standard deviation
    double dist = malahidanDistance(rgb, &mean[0], num_channels);
    stdDeviation *= stdDeviation;
    stdDeviation *= (1 - ALPHA);
    stdDeviation += ALPHA*dist;
    stdDeviation = sqrt(stdDeviation);
}

double BackgroundGaussian::checkPixelMatch(double *rgb, double sigma) // max((I-u)^2 - T_B*sig^2)
{
	sigma *= sigma;
	double* diff = new double[num_channels];
	double max;
    for(int i = 0; i < num_channels; ++i)
    {
    	diff[i] = rgb[i]-mean[i];
    	diff[i] *= diff[i];
    	diff[i] -= T_B*sigma;

    	if(i==0) max = diff[i];
    	else max = (diff[i] > max) ? diff[i] : max;
    }
    delete[] diff;
    return max;
}

