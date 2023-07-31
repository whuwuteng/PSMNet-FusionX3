
#include "InterpolateTIN.h"

#include <Eigen/Core>
#include <Eigen/Dense>

NODE::NODE()
{
	ord = 0;
	crd[0] = 0;
	crd[1] = 0;
	n_neigh_t = 0;
	neigh_t = std::vector<int>(0);
	neigh_t.reserve(64);
}

NODE::NODE(const NODE & C)
{
	ord = C.ord;
	crd[0] = C.crd[0];
	crd[1] = C.crd[1];
	n_neigh_t = C.n_neigh_t;
	neigh_t = C.neigh_t;
}


NODE::~NODE()
{

}

NODE & NODE::operator=(const NODE& C)
{
	this->ord = C.ord;
	this->crd[0] = C.crd[0];
	this->crd[1] = C.crd[1];
	this->n_neigh_t = C.n_neigh_t;
	this->neigh_t = C.neigh_t;
	return *this;
}

TRIG::TRIG()
{
	ord = 0;
	vtx[0] = 0;
	vtx[1] = 0;
	vtx[2] = 0;
	circenter[0] = 0;
	circenter[1] = 0;
	circumR = 0;
}

TRIG::TRIG(const TRIG & C)
{
	ord = C.ord;
	vtx[0] = C.vtx[0];
	vtx[1] = C.vtx[1];
	vtx[2] = C.vtx[2];
	circenter[0] = C.circenter[0];
	circenter[1] = C.circenter[1];
	circumR = C.circumR;
}

TRIG::~TRIG()
{

}

TRIG & TRIG::operator=(const TRIG& C)
{
	this->ord = C.ord;
	this->vtx[0] = C.vtx[0];
	this->vtx[1] = C.vtx[1];
	this->vtx[2] = C.vtx[2];
	this->circenter[0] = C.circenter[0];
	this->circenter[1] = C.circenter[1];
	this->circumR = C.circumR;
	return *this;
}

void TRIG::CalcTrigPlane(NODE * pNode, double * pHeight)
{
	Eigen::Matrix3d AMatrix;
	Eigen::Vector3d bVector;

	for (int i = 0; i < 3; ++i) {
		NODE * pItem = pNode + vtx[i];
		AMatrix(i, 0) = pItem->crd[0];
		AMatrix(i, 1) = pItem->crd[1];
		AMatrix(i, 2) = 1.0;

		bVector(i) = *(pHeight + vtx[i]);
	}

	Eigen::Vector3d XVector = AMatrix.inverse() * bVector;

	a = XVector(0);
	b = XVector(1);
	c = XVector(2);
}

double TRIG::Interpolate(double x, double y)
{
	double z = a * x + b * y + c;
	return z;
}
