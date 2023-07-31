//////////////////////////////////////////////////////////////////////
#ifndef INTERPOLATETIN_H__WHU_WUTENG_2018_04_03_15_25_35
#define INTERPOLATETIN_H__WHU_WUTENG_2018_04_03_15_25_35

#pragma once

#include <vector>

class NODE
{
public:
	NODE();
	virtual ~NODE();
	NODE(const NODE & C);

public:
	int    ord;
	double crd[2];

	int    n_neigh_t;
	std::vector<int> neigh_t;

public:
	NODE & operator = (const NODE& C);
};

class TRIG
{
public:
	TRIG();
	virtual ~TRIG();
	TRIG(const TRIG & C);

public:
	int    ord;
	int    vtx[3];
	double circenter[2];
	double circumR;

public:
	TRIG & operator = (const TRIG& C);
	void CalcTrigPlane(NODE * pNode, double * pHeight);
	double Interpolate(double x, double y);

protected:
	double a;
	double b;
	double c;
};
#endif // INTERPOLATETIN_H__WHU_WUTENG_2018_04_03_15_25_35
