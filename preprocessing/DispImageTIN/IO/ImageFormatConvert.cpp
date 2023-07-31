
#include "ImageFormatConvert.h"

#include <vector>

void AutoContrast(unsigned short* pSrcImage, int nRows, int nCols, int nColors, 
				  unsigned char* pTarImage, double ratioL, double ratioR)
{
	const int nByteCount = 65536;
	
	std::vector<int> histCount;
	histCount.resize(nByteCount * nColors);
	
	for (int i = 0; i < nByteCount * nColors; ++i){
		histCount[i] = 0;
	}
	
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				unsigned short usValue = pSrcImage[nIndex];
				++histCount[usValue + k * nByteCount];
			}
		}
	}
	
	int64_t nSize = (int64_t)nRows * nCols;
	
	std::vector<unsigned short> minValue;
	minValue.resize(nColors);
	
	std::vector<unsigned short> maxValue;
	maxValue.resize(nColors);
	
	for (int i = 0; i < nColors; ++i){
		minValue[i] = 0;
		maxValue[i] = 0;
	}
	for (int k = 0; k < nColors; ++k){
		int nLeftCount = 0;
		for(int i = 0; i < nByteCount; ++i){
			nLeftCount += histCount[i + k * nByteCount];
			if (nLeftCount > nSize * ratioL){
				if (i > 0){
					minValue[k] = i - 1;
				}
				else{
					minValue[k] = 0;
				}
				break;
			}
		}
		
		int nRightCount = 0;
		for(int i = nByteCount - 1; i >=0; --i){
			nRightCount += histCount[i + k * nByteCount];
			if (nRightCount > nSize * ratioR){
				if (i < nByteCount - 1){
					maxValue[k] = i + 1;
				}
				else{
					maxValue[k] = nByteCount - 1;
				}
				break;
			}
		}
	}
	
	#pragma omp parallel for
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				unsigned short usValue = pSrcImage[nIndex];
				if (usValue < minValue[k]){
					pTarImage[nIndex] = 0;
				}
				else if (usValue > maxValue[k]){
					pTarImage[nIndex] = 255;
				}
				else{
					double fValue = 255.0 * (usValue - minValue[k])/(double)(maxValue[k] - minValue[k]);
					pTarImage[nIndex] = (unsigned char)fValue;
				}
			}
		}
	}
}

void AutoContrast(unsigned char* pSrcImage, int nRows, int nCols, int nColors, double ratio)
{
	const int nByteCount = 256;
	
	std::vector<int> histCount;
	histCount.resize(nByteCount * nColors);
	
	for (int i = 0; i < nByteCount * nColors; ++i){
		histCount[i] = 0;
	}
	
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				unsigned char usValue = pSrcImage[nIndex];
				++histCount[usValue + k * nByteCount];
			}
		}
	}
	
	int64_t nSize = (int64_t)nRows * nCols;
	
	std::vector<unsigned char> minValue;
	minValue.resize(nColors);
	
	std::vector<unsigned char> maxValue;
	maxValue.resize(nColors);
	
	for (int i = 0; i < nColors; ++i){
		minValue[i] = 0;
		maxValue[i] = 0;
	}
	for (int k = 0; k < nColors; ++k){
		int nLeftCount = 0;
		for(int i = 0; i < nByteCount; ++i){
			nLeftCount += histCount[i + k * nByteCount];
			if (nLeftCount > nSize * ratio){
				if (i > 0){
					minValue[k] = i - 1;
				}
				else{
					minValue[k] = 0;
				}
				break;
			}
		}
		
		int nRightCount = 0;
		for(int i = nByteCount - 1; i >=0; --i){
			nRightCount += histCount[i + k * nByteCount];
			if (nRightCount > nSize * ratio){
				if (i < nByteCount - 1){
					maxValue[k] = i + 1;
				}
				else{
					maxValue[k] = nByteCount - 1;
				}
				break;
			}
		}
	}
	
	#pragma omp parallel for
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			for (int k = 0; k < nColors; ++k){
				int64_t nIndex = ((int64_t)(i) * nCols + (int64_t)j) * nColors + (int64_t)k;
				unsigned char usValue = pSrcImage[nIndex];
				if (usValue < minValue[k]){
					pSrcImage[nIndex] = 0;
				}
				else if (usValue > maxValue[k]){
					pSrcImage[nIndex] = 255;
				}
				else{
					double fValue = 255.0 * (usValue - minValue[k])/(double)(maxValue[k] - minValue[k]);
					pSrcImage[nIndex] = (unsigned char)fValue;
				}
			}
		}
	}
}
