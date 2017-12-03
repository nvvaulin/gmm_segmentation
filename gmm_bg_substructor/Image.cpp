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
*  http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.hpptml
******************************************************************************/

#include "Image.hpp"

ImageBase::~ImageBase()
{ 
	if(imgp != NULL && m_bReleaseMemory)
		cvReleaseImage(&imgp);
	imgp = NULL;	
}
