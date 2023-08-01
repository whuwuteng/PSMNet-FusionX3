
#include <stdio.h>
#include <string.h>

#include <OpenImageIO/imageio.h>

#include "OpenImageoiio.h"

#include "triangle.h"

#include "InterpolateTIN.h"

const unsigned char MARK_LABEL = 255;

/*****Compute the circumradius of the triangle (v0,v1,v2)*****/
double get_CircumRadius(NODE * pNode, int v0, int v1, int v2, double circenter[2])
{
    int    i;
    double A[2], B[2], C[2], A2, B2, C2, a, b, S[2], s1, s2;
    
    for (i = 0; i < 2; i++) {
        A[i] = (pNode + v0)->crd[i];
        B[i] = (pNode + v1)->crd[i];
        C[i] = (pNode + v2)->crd[i];
    }
    A2 = A[0] * A[0] + A[1] * A[1];
    B2 = B[0] * B[0] + B[1] * B[1];
    C2 = C[0] * C[0] + C[1] * C[1];
    a = (B[0] * C[1] - B[1] * C[0]) - (A[0] * C[1] - A[1] * C[0]) + (A[0] * B[1] - A[1] * B[0]);
    if (fabs(a) < 1.0e-6) {
        circenter[0] = ((pNode + v0)->crd[0] + (pNode + v1)->crd[0] + (pNode + v2)->crd[0]) / 3.0;
        circenter[1] = ((pNode + v0)->crd[1] + (pNode + v1)->crd[1] + (pNode + v2)->crd[1]) / 3.0;
        return 0;
    }
    b = A2*(B[0] * C[1] - B[1] * C[0]) - B2*(A[0] * C[1] - A[1] * C[0]) + C2*(A[0] * B[1] - A[1] * B[0]);
    S[0] = 0.5*((B2*C[1] - B[1] * C2) - (A2*C[1] - A[1] * C2) + (A2*B[1] - A[1] * B2));
    S[1] = 0.5*((B[0] * C2 - B2*C[0]) - (A[0] * C2 - A2*C[0]) + (A[0] * B2 - A2*B[0]));
    s1 = circenter[0] = S[0] / a;
    s2 = circenter[1] = S[1] / a;
    
    return sqrt(b / a + s1*s1 + s2*s2);
}

bool PointInTriangle(NODE * pNode, TRIG * pTrig, int x, int y)
{
    NODE * pNode1 = pNode + pTrig->vtx[0];
    NODE * pNode2 = pNode + pTrig->vtx[1];
    NODE * pNode3 = pNode + pTrig->vtx[2];
    
    float signOfTrig = (pNode2->crd[0] - pNode1->crd[0]) * (pNode3->crd[1] - pNode1->crd[1]) - (pNode2->crd[1] - pNode1->crd[1]) * (pNode3->crd[0] - pNode1->crd[0]);
    
    float signOfAB = (pNode2->crd[0] - pNode1->crd[0]) * (y - pNode1->crd[1]) - (pNode2->crd[1] - pNode1->crd[1]) * (x - pNode1->crd[0]);
    float signOfCA = (pNode1->crd[0] - pNode3->crd[0]) * (y - pNode3->crd[1]) - (pNode1->crd[1] - pNode3->crd[1]) * (x - pNode3->crd[0]);
    float signOfBC = (pNode3->crd[0] - pNode2->crd[0]) * (y - pNode2->crd[1]) - (pNode3->crd[1] - pNode2->crd[1]) * (x - pNode2->crd[0]);
    /*
     * if (fabs(signOfAB) < 1.0e-3 || fabs(signOfCA) < 1.0e-3 || fabs(signOfBC) < 1.0e-3){
     * return true;
}
*/
    bool d1 = (signOfAB * signOfTrig >= 0);
    bool d2 = (signOfCA * signOfTrig >= 0);
    bool d3 = (signOfBC * signOfTrig >= 0);
    return ((d1 == d2) && (d2 == d3));
}

double Uncertainty( NODE * pNode, TRIG * pTrig, int x, int y)
{
    NODE * pNode1 = pNode + pTrig->vtx[0];
    NODE * pNode2 = pNode + pTrig->vtx[1];
    NODE * pNode3 = pNode + pTrig->vtx[2];

    double x1 = pNode1->crd[0];
    double y1 = pNode1->crd[1];

    double x2 = pNode2->crd[0];
    double y2 = pNode2->crd[1];

    double x3 = pNode3->crd[0];
    double y3 = pNode3->crd[1];

    double L = x1 * y2 + x3 * y1 + x2 * y3 - x3 * y2 - x1 * y3 - x2 * y1;

    double a1 = (y2 - y3)/L;
    double a2 = (y3 - y1)/L;
    double a3 = (y1 - y2)/L;

    double b1 = (x3 - x2)/L;
    double b2 = (x1 - x3)/L;
    double b3 = (x2 - x1)/L;

    double c1 = (x2 * y3 - x3 * y2)/L;
    double c2 = (x3 * y1 - x1 * y3)/L;
    double c3 = (x1 * y2 - x2 * y1)/L;

    double m1 = a1 * x + b1 * y + c1;
    double m2 = a2 * x + b2 * y + c2;
    double m3 = a3 * x + b3 * y + c3;

    double M = m1 * m1 + m2 * m2 + m3 * m3;

    return M;
}

bool TriangleConsistency( NODE * pNode, TRIG * pTrig, double *pDisp, int nRows, int nCols)
{
    // a threshold for the incontinuty
    double disp_continuty = 3.0 * 256.0;

    NODE * pNode1 = pNode + pTrig->vtx[0];
    NODE * pNode2 = pNode + pTrig->vtx[1];
    NODE * pNode3 = pNode + pTrig->vtx[2];

    int x1 = int(pNode1->crd[0] + 0.5);
    int y1 = int(pNode1->crd[1] + 0.5);

    int x2 = int(pNode2->crd[0] + 0.5);
    int y2 = int(pNode2->crd[1] + 0.5);

    int x3 = int(pNode3->crd[0] + 0.5);
    int y3 = int(pNode3->crd[1] + 0.5);

    if (fabs(pDisp[y1 * nCols + x1] - pDisp[y2 * nCols + x2]) < disp_continuty && \
        fabs(pDisp[y1 * nCols + x1] - pDisp[y3 * nCols + x3]) < disp_continuty && \
        fabs(pDisp[y2 * nCols + x2] - pDisp[y3 * nCols + x3]) < disp_continuty){
            return true;
    }
    else{
        return false;
    }
}

// use the TIN to give a more dense disparity
int main(int argc, char const *argv[])
{
	if (argc != 4){
		std::cout << "CreateDispDenseTIN SrcImage TarImage uncertainty" << std::endl;
		return -1;
	}
	
	char szSrcImg[512] = { 0 };
	char szTarImg[512] = { 0 };
    char szCmpImg[512] = { 0 };
	
	strcpy(szSrcImg, argv[1]);
	strcpy(szTarImg, argv[2]);
    strcpy(szCmpImg, argv[3]);
	
	COpenImageIO srcImage;
	if (! srcImage.Open(szSrcImg)){
        std::cout << "Can not open: " << szSrcImg << std::endl;
        return -1;
    }
	
	int nRows = srcImage.GetRows();
	int nCols = srcImage.GetCols();

	unsigned short * pImage = new unsigned short[nRows * nCols];
	memset(pImage, 0, sizeof(unsigned short) * nRows * nCols);
	
	srcImage.Read(pImage, COpenImageIO::USHORT);
	srcImage.Close();
	
	triangulateio inputTri;
    memset(&inputTri, 0, sizeof(triangulateio));
    
    inputTri.pointlist = (REAL *)malloc(nRows * nCols * 2 * sizeof(REAL));
    
    unsigned char * pMark = new unsigned char[nRows * nCols];
    memset(pMark, 0, sizeof(unsigned char) * nRows * nCols);
    
    double * pDisp = new double[nRows * nCols];
    memset(pDisp, 0, sizeof(double) * nRows * nCols);

    double * pUnCertainty = new double[nRows * nCols];
    memset(pUnCertainty, 0, sizeof(double) * nRows * nCols);
    
    int nValid = 0;
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            if (pImage[i * nCols + j] > 0) {
                pMark[i * nCols + j] = MARK_LABEL;
                pDisp[i * nCols + j] = pImage[i * nCols + j];
                
                inputTri.pointlist[nValid * 2] = j;
                inputTri.pointlist[nValid * 2 + 1] = i;
                
                ++nValid;
            }
        }
    }
    delete []pImage; pImage = NULL;
     
    inputTri.numberofpoints = nValid;
    
    triangulateio outputTri;
    memset(&outputTri, 0, sizeof(triangulateio));
    
    triangulate("ze", &inputTri, &outputTri, NULL);
    
    if (inputTri.pointlist) free(inputTri.pointlist);
    if (inputTri.pointmarkerlist) free(inputTri.pointmarkerlist);
    if (inputTri.edgelist) free(inputTri.edgelist);
    if (inputTri.edgemarkerlist) free(inputTri.edgemarkerlist);
    if (inputTri.trianglelist) free(inputTri.trianglelist);
    
    NODE * pNode = new NODE[nValid];
    
    double * pHeight = new double[nValid];
    memset(pHeight, 0, sizeof(double) * nValid);
    
    for (int i = 0; i < outputTri.numberofpoints; ++i) {
        pNode[i].ord = i;
        pNode[i].crd[0] = outputTri.pointlist[i * 2];
        pNode[i].crd[1] = outputTri.pointlist[i * 2 + 1];
        
        int ix = int(pNode[i].crd[0] + 0.5);
        int iy = int(pNode[i].crd[1] + 0.5);
        
        pHeight[i] = pDisp[iy * nCols + ix];
    }
    
    int nTrig = outputTri.numberoftriangles;
    TRIG * pTrig = new TRIG[nTrig];
    
    for (int i = 0; i < nTrig; ++i) {
        (pTrig + i)->ord = i;
        for (int j = 0; j < 3; j++) {
            int k = outputTri.trianglelist[i * 3 + j];
            (pTrig + i)->vtx[j] = k;
            int ntemp = (pNode + k)->n_neigh_t;
            (pNode + k)->neigh_t.push_back((pTrig + i)->ord);
            (pNode + k)->n_neigh_t = ntemp + 1;
        }
    }
    
    if (outputTri.pointlist)			free(outputTri.pointlist);
	if (outputTri.pointmarkerlist)		free(outputTri.pointmarkerlist);
	if (outputTri.edgelist)				free(outputTri.edgelist);
	if (outputTri.edgemarkerlist)		free(outputTri.edgemarkerlist);
	if (outputTri.trianglelist)			free(outputTri.trianglelist);
	
    for (int i = 0; i < nTrig; ++i) {
        (pTrig + i)->circumR = get_CircumRadius(pNode, pTrig[i].vtx[0], pTrig[i].vtx[1], pTrig[i].vtx[2], (pTrig + i)->circenter);
    }
    
    std::vector< std::vector<int> > TrigLabel;
    TrigLabel.resize(nRows * nCols);
    
    for (int i = 0; i < nTrig; ++i) {
        int nMinX = std::max(int(pTrig[i].circenter[0] - pTrig[i].circumR + 0.5) - 1, 0);
        int nMaxX = std::min(int(pTrig[i].circenter[0] + pTrig[i].circumR + 0.5) + 1, nCols - 1);
        
        int nMinY = std::max(int(pTrig[i].circenter[1] - pTrig[i].circumR + 0.5) - 1, 0);
        int nMaxY = std::min(int(pTrig[i].circenter[1] + pTrig[i].circumR + 0.5) + 1, nRows - 1);
        
        for (int m = nMinY; m <= nMaxY; ++m) {
            for (int n = nMinX; n <= nMaxX; ++n) {
                double rdistance = (n - pTrig[i].circenter[0]) * (n - pTrig[i].circenter[0]) + (m - pTrig[i].circenter[1]) * (m - pTrig[i].circenter[1]);
                rdistance = sqrt(rdistance);
                
                if (rdistance <= pTrig[i].circumR){
                    TrigLabel[m * nCols + n].push_back(i);
                }
            }
        }
    }
    
    for (int i = 0; i < nTrig; ++i) {
        pTrig[i].CalcTrigPlane(pNode, pHeight);
    }
    
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            if (pMark[i * nCols + j] == MARK_LABEL){
                pUnCertainty[i * nCols + j] = 1.0;
            }
            else{
                int nCircle = TrigLabel[i * nCols + j].size();
                if (nCircle > 0){
                    for (int m = 0; m < nCircle; ++m) {
                        TRIG * pTrigOverLap = pTrig + TrigLabel[i * nCols + j][m];
                        if (PointInTriangle(pNode, pTrigOverLap, j, i)){
                            // only check the evalueation
                            if (TriangleConsistency(pNode, pTrigOverLap, pDisp, nRows, nCols)){
                                // too large
                                pDisp[i * nCols + j] = pTrigOverLap->Interpolate(j, i);

                                //calculate the uncertainty
                                pUnCertainty[i * nCols + j] = Uncertainty(pNode, pTrigOverLap, j, i);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    
    
    unsigned short * pTarImage = new unsigned short[nRows * nCols];
    memset(pTarImage, 0, sizeof(unsigned short) * nRows * nCols);
    
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            pTarImage[i * nCols + j] = (unsigned short)(pDisp[i * nCols + j] + 0.5);
        }
    }
    
    delete []pDisp; pDisp = NULL;
    delete []pMark; pMark = NULL;
	
	COpenImageIO tarImg;
	if ( !tarImg.Open(szTarImg, COpenImageIO::modeCreate)){
        std::cout << "Can not open: " << szTarImg << std::endl;
		return -1;
	}
	tarImg.SetRows(nRows);
	tarImg.SetCols(nCols);
	tarImg.SetDataType(COpenImageIO::USHORT);
	tarImg.SetPixelBytes(1);

	tarImg.Write(pTarImage);
	tarImg.Close();
	
	delete []pTarImage;		pTarImage = NULL;
	
    unsigned char * pUnCertaintyImg = new unsigned char[nRows * nCols];
    memset(pUnCertaintyImg, 0, sizeof(unsigned char) * nRows * nCols);

    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            if (pUnCertainty[i * nCols + j] > 0){
                pUnCertaintyImg[i * nCols + j] = (unsigned char)(pUnCertainty[i * nCols + j] * 255 + 0.5);
            }
        }
    }

    delete []pUnCertainty;    pUnCertainty = NULL;
    SaveImg(szCmpImg, pUnCertaintyImg, nRows, nCols, 1);

    delete []pUnCertaintyImg; pUnCertaintyImg = NULL;
    return 0;
}
