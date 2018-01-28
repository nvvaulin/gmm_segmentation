#ifndef SRC_PIXEL_HPP_
#define SRC_PIXEL_HPP_

#include <vector>

#include "tools.hpp"
#include "BackgroundGaussian.hpp"
#include "ForegroundGaussian.hpp"

using namespace std;
namespace SplitGaussianPixel{
class Pixel
{
    private:
		ForegroundGaussian fgGaussian;
		int num_channels;
    public:
        vector <BackgroundGaussian> bgGaussians;
        Pixel(int _num_channels=1):
            fgGaussian(ForegroundGaussian(_num_channels)),
            num_channels(_num_channels)
        {};
        void insertBackgroundGaussian(double weight, double * means, double std_dev);
        void initialiseForeground(double * means);

        bool isForeground(double * rgb);
        bool isBackground(double * rgb);

        void update(bool updateMask, double * rgb);

};
}

#endif /* SRC_PIXEL_HPP_ */
