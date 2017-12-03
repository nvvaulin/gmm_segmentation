/****************************************************************************
*                                                                             
*   This program is free software: you can redistribute it and/or modify     
*    it under the terms of the GNU General Public License as published by     
*    the Free Software Foundation, either version 3 of the License, or        
*    (at your option) any later version.                                      
*                                                                             
*    This program is distributed in the hope that it will be useful,          
*    but WITHOUT ANY WARRANTY; without even the implied warranty of           
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            
*    GNU General Public License for more details.                             
*                                                                             
*    You should have received a copy of the GNU General Public License        
*    along with this program. If not, see <http://www.gnu.org/licenses/>.     
*                                                                             
******************************************************************************/

/****************************************************************************
*
* Image.hpp
*
* Purpose:  C++ wrapper for OpenCV IplImage which supports simple and 
*						efficient access to the image data
*
* Author: Donovan Parks, September 2007
*
* Based on code from: 
*  http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html
******************************************************************************/

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <cv.h>
#include <cxcore.h>
#include <vector>
#include <cstring>
using namespace std;

// --- Constants --------------------------------------------------------------

const unsigned char NUM_CHANNELS = 3;


// --- Pixel Types ------------------------------------------------------------

class Pixel
{
public:
	Pixel() {;}
	Pixel(std::vector<float> _data) : data(_data)
	{}

	Pixel& operator=(const Pixel& rhs)
	{
		data=rhs.data;
		return *this;
	}

	inline float& operator()(const int _ch)
	{
		return data[_ch];
	}

	inline float operator()(const int _ch) const
	{
		return data[_ch];
	}

	std::vector<float> data;
};


// --- Image Types ------------------------------------------------------------

class ImageBase
{
public:
 ImageBase(IplImage* img = NULL) { imgp = img; m_bReleaseMemory = true; }
  ~ImageBase();

	void ReleaseMemory(bool b) { m_bReleaseMemory = b; }

	IplImage* Ptr() { return imgp; }
	const IplImage* Ptr() const { return imgp; }

	void operator=(IplImage* img) 
	{ 
		imgp = img;
	}

	// copy-constructor
	ImageBase(const ImageBase& rhs)
	{	
		// it is very inefficent if this copy-constructor is called
		assert(false);
	}

	// assignment operator
	ImageBase& operator=(const ImageBase& rhs)
	{
		// it is very inefficent if operator= is called
		assert(false);

		return *this;
	}

	virtual void Clear() = 0;

protected:
	IplImage* imgp;
	bool m_bReleaseMemory;
};


class Image : public ImageBase
{
public:
	Image(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img) 
	{ 
		imgp = img;
	}

	// channel-level access using image(row, col, channel)
	inline float& operator()(const int r, const int c, const int ch)
	{
		return (float &)imgp->imageData[r*imgp->widthStep+(c*imgp->nChannels+ch)*sizeof(float)];
	}

	inline float operator()(const int r, const int c, const int ch) const
	{
		return (float)imgp->imageData[r*imgp->widthStep+(c*imgp->nChannels+ch)*sizeof(float)];
	}

	// RGB pixel-level access using image(row, col)
	inline Pixel& operator()(const int r, const int c) 
	{
	    assert(false);
	}

	inline Pixel operator()(const int r, const int c) const
	{
			vector<float> res(imgp->nChannels,0);
			std::memcpy((void*)&res[0],
				    (const void*)&imgp->imageData[r*imgp->widthStep+c*imgp->nChannels*sizeof(float)],
				    imgp->nChannels*sizeof(float));
    		return Pixel(res);
	}
};

class BwImage : public ImageBase
{
public:
	BwImage(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img) 
	{ 
		imgp = img;
	}

	// pixel-level access using image(row, col)
	inline unsigned char& operator()(const int r, const int c)
	{
		return (unsigned char &)imgp->imageData[r*imgp->widthStep+c];
	}

	inline unsigned char operator()(const int r, const int c) const
	{
		return (unsigned char)imgp->imageData[r*imgp->widthStep+c];
	}
};

class BwImageFloat : public ImageBase
{
public:
	BwImageFloat(IplImage* img = NULL) : ImageBase(img) { ; }

	virtual void Clear()
	{
		cvZero(imgp);
	}

	void operator=(IplImage* img) 
	{ 
		imgp = img;
	}

	// pixel-level access using image(row, col)
	inline float& operator()(const int r, const int c)
	{
		return (float &)imgp->imageData[r*imgp->widthStep+c*sizeof(float)];
	}

	inline float operator()(const int r, const int c) const
	{
		return (float)imgp->imageData[r*imgp->widthStep+c*sizeof(float)];
	}
};



// --- Image Functions --------------------------------------------------------

void DensityFilter(BwImage& image, BwImage& filtered, int minDensity, unsigned char fgValue);

#endif
