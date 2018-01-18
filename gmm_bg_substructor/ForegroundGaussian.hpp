/*
 * single_gaussian.h
 *
 *  Created on: 21 lis 2015
 *      Author: Piotr Janus
 */

#ifndef SRC_FOREGROUND_GAUSSIAN_HPP_
#define SRC_FOREGROUND_GAUSSIAN_HPP_

#include <math.h>
#include <cstring>
#include <memory>
#include "tools.hpp"
#include <vector>
using namespace std;
class ForegroundGaussian {

private:
	static double ALPHA;
	static double T_F;

    vector<double> mean;
    bool initFlag;
    int num_channels;

public:
    ForegroundGaussian(int _num_channels) :
        initFlag(false),
        mean(vector<double>(_num_channels)),
        num_channels(_num_channels)
    {};
	~ForegroundGaussian(){
	};
    void initialise(double *gaussianMean);
    void getMean(double * gaussianMean);

    void update(double *rgb);
    bool checkPixelMatch(double *rgb);

    static void setAlpha(double alpha) {ForegroundGaussian::ALPHA = alpha; }
    static void setTF(double th) {ForegroundGaussian::T_F = th; }
};

#endif /* SRC_FOREGROUND_GAUSSIAN_HPP_ */
