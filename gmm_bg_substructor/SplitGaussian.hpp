#ifndef SRC_SPLIT_GAUSSIANS_HPP_
#define SRC_SPLIT_GAUSSIANS_HPP_

#include <opencv2/core/core.hpp>
#include <vector>
using namespace cv;
using namespace std;
#include "Pixel.hpp"
#include "tools.hpp"

class SplitGaussian
{
    private:
        vector<vector<SplitGaussianPixel::Pixel> > pixels;
        int height;
        int width;
        bool initFlag;
        int numOfBgGaussians = 0;


    public:

	double init_variance;
        SplitGaussian(double bgAlpha, double fgAlpha, double tb, double tf, double tl,double init_variance);
        ~SplitGaussian();

        bool isInitialized() { return initFlag; }

        void setFrameSize(int width, int height,int num_channels);
        void backgroundInit(int row, int col, uchar * rgb);
        void foregroundInit(int row, int col, uchar * rgb);

        void detection(const Mat & input, Mat & background, Mat & foreground, bool * isLargeChange);
        void update(int x, int y, bool updateFlag, double * rgb);

        void initialize(const Mat & input);

        void addNewGaussian(int x, int y, double weight, double * means, double sd);

};

#endif /* SRC_SPLIT_GAUSSIANS_HPP_ */
