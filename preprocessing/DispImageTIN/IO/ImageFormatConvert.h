
#ifndef __IMAGE_FORMAT_CONVERT_OPEN_IMAGE_IO_H__
#define __IMAGE_FORMAT_CONVERT_OPEN_IMAGE_IO_H__

#pragma once

#include <cstdint>

// main for COpenImageIO(use lib)

// auto contrast
// from 16bit to 8bit
// ratio = 0.03 is too large
void AutoContrast(unsigned short * pSrcImage, int nRows, int nCols, int nColors, 
				  unsigned char * pTarImage, double ratioL = 0.001, double ratioR = 0.001);

// auto contrast
// from 8bit to 8bit
// use the origin data
void AutoContrast(unsigned char * pSrcImage, int nRows, int nCols, int nColors, double ratio = 0.001);

// Zoom out image
template<typename T>
void ZoomOutImage(T * pImage, int nSrcRows, int nSrcCols, int nColors, int nZoom, T * pZoomImg)
{
	int nRows = nSrcRows/nZoom;
	int nCols = nSrcCols/nZoom;
	
	#pragma omp parallel for
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nSrcIndex = ((int64_t)(i) * nZoom * nSrcCols + (int64_t)j * nZoom) * nColors + (int64_t)k;
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				pZoomImg[nIndex] = pImage[nSrcIndex];
			}
		}
	}
}

// Rotate 180°
template<typename T>
void RotaeImageInverse(T * pImage, int nRows, int nCols, int nColors, T * pRotateImg)
{
	#pragma omp parallel for
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nSrcIndex = ((int64_t)(nRows -1 - i) * nCols + (int64_t)(nCols - 1 -j)) * nColors + (int64_t)k;
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				pRotateImg[nIndex] = pImage[nSrcIndex];
			}
		}
	}
}

//TODO
template<typename T>
void rgb2gray(T * pRgb, int nRows, int nCols, T * pGray)
{
	double weight[3] = {0.2989, 0.5870, 0.1140};
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			double fValue = 0;
			for (int k = 0; k < 3; ++k){
				fValue += pRgb[(i * nCols + j) * 3 + k] * weight[k];
			}
			pGray[i * nCols + j] = (T)fValue;
		}
	}
}
#endif //__IMAGE_FORMAT_CONVERT_OPEN_IMAGE_IO_H__
