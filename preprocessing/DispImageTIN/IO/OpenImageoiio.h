#ifndef __SAVE_IMAGE_OPEN_IMAGE_IO_H__
#define __SAVE_IMAGE_OPEN_IMAGE_IO_H__

#pragma once

#include <OpenImageIO/imageio.h>

// to do : write the image io class
// use h and cpp model to make sure the code can be extended later and keep clean
class COpenImageIO
{
public:
	COpenImageIO() {
		// need to check
		//m_pImage = NULL;
	}
	virtual ~COpenImageIO(){
	}
	// open mode
	enum OPENFLAGS { modeRead = 0x0000, modeCreate = 0x1000 };
	// image format(support format)
	enum IMGBASETYPE {UNKNOWN, UCHAR, USHORT, SHORT, FLOAT, DOUBLE};
public:
	bool Open(const char* lpstrPathName, unsigned int flag = modeRead);
	int GetRows() {
		return m_ImgSpec.height;
	}
	int GetCols() {
		return m_ImgSpec.width;
	}
	void SetRows(int nRows) {
		m_ImgSpec.height = nRows;
	}
	void SetCols(int nCols) {
		m_ImgSpec.width = nCols;
	}
	int GetPixelBytes() {
		return m_ImgSpec.nchannels;
	}
	void SetPixelBytes(int nPixelBytes) {
		m_ImgSpec.nchannels = nPixelBytes;
	}
	// datatype
	int GetDataType();
	bool SetDataType(IMGBASETYPE iType = UCHAR);
	// io same with spvzimage
	bool Read(unsigned char * pBuf, int pxlBytes, int sRow, int sCol, int rows, int cols);
	// only works for 0...n
	bool Write(unsigned char *pBuf, int rowIdx);
	
	// io for full image read
	// Zoom rate is 1.0/nZoom(nZoom >= 1)
	bool Read(void * pBuf, IMGBASETYPE iType = UCHAR, int nZoom = 1, double autocontrast_left = 0.001, double autocontrast_right = 0.001);
	bool Write(void * pBuf);
	void Close();
protected:
	std::unique_ptr<OIIO::ImageInput> m_pImageIn;
	std::unique_ptr<OIIO::ImageOutput> m_pImageOut;
	OIIO::ImageSpec m_ImgSpec;
	std::string m_strImage;
	bool m_bWriteCreate;
};

// save float image
// already defined in spvzimage.h
inline bool SaveImg(const char * pszName, unsigned char * pImage, int nRows, int nCols, int nColors)
{
	std::unique_ptr<OIIO::ImageOutput> out;
	out = OIIO::ImageOutput::create(pszName);
	if (! out){
		std::cout << "can not open" << pszName << std::endl;
		return false;
	}
	
	OIIO::ImageSpec spec_out(nCols, nRows, nColors, OIIO::TypeDesc::UCHAR);
	
	// for micmac
	//spec_out.attribute("compression", "none");
	
	out->open(pszName, spec_out);
	out->write_image(OIIO::TypeDesc::UCHAR, pImage);
	out->close();
	
	return true;
}

// save float image
inline bool SaveImg(const char * pszName, float * pImage, int nRows, int nCols, bool bCompress = false)
{
	std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(pszName);
	if (! out){
		std::cout << "can not open" << pszName << std::endl;
		return false;
	}
	
	OIIO::ImageSpec spec_out(nCols, nRows, 1, OIIO::TypeDesc::FLOAT);
	
	// for micmac
	if (! bCompress){
		spec_out.attribute("compression", "none");
	}
	
	out->open(pszName, spec_out);
	out->write_image(OIIO::TypeDesc::FLOAT, pImage);
	out->close();
	
	return true;
}

// save float image
inline bool SaveImg(const char * pszName, unsigned short * pImage, int nRows, int nCols, bool bCompress = false)
{
	std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(pszName);
	if (! out){
		std::cout << "can not open" << pszName << std::endl;
		return false;
	}
	
	OIIO::ImageSpec spec_out(nCols, nRows, 1, OIIO::TypeDesc::USHORT);
	
	// for micmac
	if (! bCompress){
		spec_out.attribute("compression", "none");
	}
	
	out->open(pszName, spec_out);
	out->write_image(OIIO::TypeDesc::USHORT, pImage);
	out->close();
	
	return true;
}
#endif // __SAVE_IMAGE_OPEN_IMAGE_IO_H__
