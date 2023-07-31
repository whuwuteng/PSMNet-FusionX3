
#include "OpenImageoiio.h"

#include <stdio.h>
#include <string.h>

#include <iostream>

#include "ImageFormatConvert.h"

bool COpenImageIO::Open(const char* lpstrPathName, unsigned int flag)
{
	if (flag == modeRead){
		m_pImageIn = OIIO::ImageInput::open(lpstrPathName);
		if (! m_pImageIn){
			std::cout << "can not open: " << lpstrPathName << std::endl;
			return false;
		}
		m_ImgSpec = m_pImageIn->spec();
		m_bWriteCreate = false;
		return true;
	}
	else if (flag == modeCreate){
		m_pImageOut = OIIO::ImageOutput::create(lpstrPathName);
		if (! m_pImageOut){
			std::cout << "can not open: " << lpstrPathName << std::endl;
			return false;
		}
		m_strImage = std::string(lpstrPathName);
		m_bWriteCreate = false;
		return true;
	}
	else{
		return false;
	}
}

bool COpenImageIO::Read(unsigned char * pBuf, int pxlBytes, int sRow, int sCol, int rows, int cols)
{
	return m_pImageIn->read_tiles(0, 0, sCol, sCol + cols, sRow, sRow + rows, 0, 1, OIIO::TypeDesc::UINT8, pBuf);
}

bool COpenImageIO::Write(unsigned char *pBuf, int rowIdx)
{
	if (! m_bWriteCreate){
		// default
		m_ImgSpec.format = OIIO::TypeDesc::UCHAR;
		m_ImgSpec.attribute("compression", "none");
		m_pImageOut->open(m_strImage, m_ImgSpec);
		m_bWriteCreate = true;
	}
	return m_pImageOut->write_scanline(rowIdx, 0, m_ImgSpec.format, pBuf);
}

void COpenImageIO::Close() {
	if (m_bWriteCreate){
		m_pImageOut->close();
	}
	else{
		m_pImageIn->close();
	}
}

int COpenImageIO::GetDataType()
{
	IMGBASETYPE iType;
	switch (m_ImgSpec.format){
		case OIIO::TypeDesc::UNKNOWN :
			iType = IMGBASETYPE::UNKNOWN;
			break;
		case OIIO::TypeDesc::UCHAR :
			iType = IMGBASETYPE::UCHAR;
			break;
		case OIIO::TypeDesc::USHORT :
			iType = IMGBASETYPE::USHORT;
			break;
		case OIIO::TypeDesc::FLOAT:
			iType = IMGBASETYPE::FLOAT;
			break;
		default:
			iType = IMGBASETYPE::UNKNOWN;
			break;
	}
	return iType;
}

bool COpenImageIO::SetDataType(COpenImageIO::IMGBASETYPE iType)
{
	switch (iType){
		case IMGBASETYPE::UCHAR :
			m_ImgSpec.format = OIIO::TypeDesc::UCHAR;
			break;
		case IMGBASETYPE::USHORT :
			m_ImgSpec.format = OIIO::TypeDesc::USHORT;
			break;
		case IMGBASETYPE::FLOAT:
			m_ImgSpec.format = OIIO::TypeDesc::FLOAT;
			break;
		default:
			m_ImgSpec.format = OIIO::TypeDesc::UNKNOWN;
			break;
	}
	if (m_ImgSpec.format == OIIO::TypeDesc::UNKNOWN){
		printf("not support format\n");
		return false;
	}
	return true;
}

// surpport > 4GB image
bool COpenImageIO::Read(void* pBuf, COpenImageIO::IMGBASETYPE iType, int nZoom,  double autocontrast_left, double autocontrast_right)
{
	//std::cout << m_ImgSpec.format << std::endl;
	if (iType == IMGBASETYPE::UCHAR){
		if(m_ImgSpec.format == OIIO::TypeDesc::UCHAR){
			if (nZoom == 1){
				m_pImageIn->read_image(OIIO::TypeDesc::UCHAR, pBuf);
			}
			else{
				int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
				unsigned char * pImage = new unsigned char[nSize];
				memset(pImage, 0, sizeof(unsigned char) * nSize);
				
				m_pImageIn->read_image(OIIO::TypeDesc::UCHAR, pImage);
				
				unsigned char * pZoomImg = (unsigned char *)pBuf;
				ZoomOutImage(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				
				delete []pImage; 		pImage = NULL;
			}
		}
		else if (m_ImgSpec.format == OIIO::TypeDesc::USHORT){
			int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
			unsigned short * pImage = new unsigned short[nSize];
			memset(pImage, 0, sizeof(unsigned short) * nSize);
			
			m_pImageIn->read_image(OIIO::TypeDesc::USHORT, pImage);
			
			if (nZoom == 1){
				AutoContrast(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, (unsigned char *)pBuf, autocontrast_left, autocontrast_right);				
			}
			else{
				unsigned char * pImage8 = new unsigned char[nSize];
				memset(pImage8, 0, sizeof(unsigned char) * nSize);
				
				AutoContrast(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, pImage8, autocontrast_left, autocontrast_right);
				
				unsigned char * pZoomImg = (unsigned char *)pBuf;
				ZoomOutImage(pImage8, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				delete []pImage8;				pImage8 = NULL;
			}
			delete []pImage; 		pImage = NULL;
			
		}
		else{
			printf("not support format\n");
			return false;
		}
	}
	else if (iType == IMGBASETYPE::USHORT){
		if (m_ImgSpec.format == OIIO::TypeDesc::USHORT){
			if (nZoom == 1){
				m_pImageIn->read_image(OIIO::TypeDesc::USHORT, pBuf);
			}
			else{
				int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
				unsigned short * pImage = new unsigned short[nSize];
				memset(pImage, 0, sizeof(unsigned short) * nSize);
				
				m_pImageIn->read_image(OIIO::TypeDesc::USHORT, pImage);
				
				unsigned short * pZoomImg = (unsigned short *)pBuf;
				ZoomOutImage(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				
				delete []pImage; 		pImage = NULL;
			}
		}
		else{
			printf("not support format\n");
			return false;
		}
	}
	else if (iType == IMGBASETYPE::SHORT){
		if (m_ImgSpec.format == OIIO::TypeDesc::SHORT){
			if (nZoom == 1){
				m_pImageIn->read_image(OIIO::TypeDesc::SHORT, pBuf);
			}
			else{
				int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
				short * pImage = new short[nSize];
				memset(pImage, 0, sizeof(short) * nSize);
				
				m_pImageIn->read_image(OIIO::TypeDesc::UCHAR, pImage);
				
				short * pZoomImg = (short *)pBuf;
				ZoomOutImage(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				
				delete []pImage; 		pImage = NULL;
			}
		}
		else{
			printf("not support format\n");
			return false;
		}
	}
	else if (iType == IMGBASETYPE::FLOAT){
		if (m_ImgSpec.format == OIIO::TypeDesc::FLOAT){
			if (nZoom == 1){
				m_pImageIn->read_image(OIIO::TypeDesc::FLOAT, pBuf);
			}
			else{
				int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
				float * pImage = new float[nSize];
				memset(pImage, 0, sizeof(float) * nSize);
				
				m_pImageIn->read_image(OIIO::TypeDesc::FLOAT, pImage);
				
				float * pZoomImg = (float *)pBuf;
				ZoomOutImage(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				
				delete []pImage; 		pImage = NULL;
			}
		}
		else{
			printf("not support format\n");
			return false;
		}
	}
	else if (iType == IMGBASETYPE::DOUBLE){
		if (m_ImgSpec.format == OIIO::TypeDesc::DOUBLE){
			if (nZoom == 1){
				m_pImageIn->read_image(OIIO::TypeDesc::DOUBLE, pBuf);
			}
			else{
				int64_t nSize = (int64_t)(m_ImgSpec.width) * m_ImgSpec.height * m_ImgSpec.nchannels;
				double * pImage = new double[nSize];
				memset(pImage, 0, sizeof(double) * nSize);
				
				m_pImageIn->read_image(OIIO::TypeDesc::DOUBLE, pImage);
				
				double * pZoomImg = (double *)pBuf;
				ZoomOutImage(pImage, m_ImgSpec.height, m_ImgSpec.width, m_ImgSpec.nchannels, nZoom, pZoomImg);
				
				delete []pImage; 		pImage = NULL;
			}
		}
		else{
			printf("not support format\n");
			return false;
		}
	}
	else{
		printf("not support format\n");
		return false;
	}
	return true;
}

bool COpenImageIO::Write(void* pBuf)
{
	if (! m_bWriteCreate){
		// unsigned char is default
		if (m_ImgSpec.format == OIIO::TypeDesc::UNKNOWN){
			m_ImgSpec.format = OIIO::TypeDesc::UCHAR;
		}
		// for micmac
		m_ImgSpec.attribute("compression", "none");
		
		m_pImageOut->open(m_strImage, m_ImgSpec);
		m_bWriteCreate = true;
	}	
	return m_pImageOut->write_image(m_ImgSpec.format, pBuf);
}
