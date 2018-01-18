#include <iostream>
#include <cstdlib>
#include <ctime>

#include "tools.hpp"
#include "SplitGaussian.hpp"

SplitGaussian::SplitGaussian(double bgAlpha, double fgAlpha, double tb, double tf, double tl,double _init_variance) :
	pixels(NULL),
        height(0),
        width(0),
	initFlag(false),
	init_variance(_init_variance)

{
	ForegroundGaussian::setAlpha(fgAlpha);
	ForegroundGaussian::setTF(tf);

	BackgroundGaussian::setAlpha(bgAlpha);
	BackgroundGaussian::setTB(tb);
}

SplitGaussian::~SplitGaussian()
{
}

void SplitGaussian::setFrameSize(int width, int height,int num_channels){
	this->width = width;
	this->height = height;

	pixels = vector<vector<SplitGaussianPixel::Pixel> >(height);
    for(int i=0; i < height; ++i)
	{
        pixels[i] = vector<SplitGaussianPixel::Pixel>(width,SplitGaussianPixel::Pixel(num_channels));
	}
}

void SplitGaussian::update(int x, int y, bool updateFlag, double * rgb){

	pixels[x][y].update(updateFlag, rgb);

}

void SplitGaussian::detection(const Mat & input, Mat & background, Mat & foreground, bool * isLargeChange){

	static Mat prevBackground;

	double* input_rgb = new double[input.channels()];

	const float * input_pixel_ptr;
	uchar * background_pixel_ptr;
	uchar * foreground_pixel_ptr;
	uchar * prev_bg_pixel_ptr;

	uchar bg_pixel, fg_pixel;
	double prev_bg_pixel, total_diff = 0;

	for(int row = 0; row < input.rows; ++row)
	{
		input_pixel_ptr = input.ptr<float>(row);
		background_pixel_ptr = background.ptr(row);
		foreground_pixel_ptr = foreground.ptr(row);

		if(!prevBackground.empty())
			prev_bg_pixel_ptr = prevBackground.ptr(row);

	    for(int col = 0; col < input.cols; ++col)
	    {
		for(unsigned c = 0; c < input.channels();++c){
	    		input_rgb[c] = (double)*input_pixel_ptr++;
		}

	    	bg_pixel = (pixels[row][col].isBackground(input_rgb)) ? BLACK : WHITE;
	    	fg_pixel = (pixels[row][col].isForeground(input_rgb)) ? WHITE : BLACK;

	    	*background_pixel_ptr++ = bg_pixel;
	    	*foreground_pixel_ptr++ = fg_pixel;

	    	if(!prevBackground.empty()){
	    		prev_bg_pixel = (double)*prev_bg_pixel_ptr++;
	    		double diff = (double)bg_pixel - prev_bg_pixel;
	    		diff = (diff < 0) ? (-1*diff) : diff;
	    		total_diff += diff;
	    	}
	    }
	}
	delete[] input_rgb;


	total_diff = total_diff/(input.rows*input.cols);
	*isLargeChange = total_diff > 20;
	
	//cout << total_diff << endl;

	prevBackground = background.clone();
}

void SplitGaussian::initialize(const Mat & input){

	static int numOfBgGaussians = 0;

	double* input_rgb = new double[input.channels()];
	const float * input_pixel_ptr;

	for(int row = 0; row < input.rows; ++row)
	{
		input_pixel_ptr = input.ptr<float>(row);

	    for(int col = 0; col < input.cols; ++col)
	    {
			for(unsigned c=0; c < input.channels();++c){
					input_rgb[c] = (double)*input_pixel_ptr++;
			}
	    	pixels[row][col].insertBackgroundGaussian(0.2, input_rgb, init_variance);
	    }
	}
	delete[] input_rgb;
	numOfBgGaussians++;
	if(numOfBgGaussians == 5)
		initFlag = true;

}

void SplitGaussian::addNewGaussian(int x, int y, double weight, double * means, double sd){
	pixels[x][y].insertBackgroundGaussian(weight, means, sd);
}

