#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf_msh.h"
#include "natural_array.h"
#include "math_type.h"
#include "geom_entity.h"

// 1st-order derivative using central difference.
inline double fder1(double fl, double fm, double fr, double dxl, double dxr)
{
	const double dxrl = dxr / dxl;
	const double dxlr = dxl / dxr;
	return (dxlr*fr - dxrl * fl - (dxlr - dxrl)*fm) / (dxl + dxr);
}

// 2nd-order derivative using central difference.
inline double fder2(double fl, double fm, double fr, double dxl, double dxr)
{
	const double inv_dxl = 1.0 / dxl;
	const double inv_dxr = 1.0 / dxr;
	return 2.0 / (dxl + dxr) * (fl * inv_dxl + fr * inv_dxr - fm * (inv_dxl + inv_dxr));
}

const Scalar T0 = 300.0; // K
const Scalar P0 = 101325.0; // Pa
const Scalar rho = 1.225; //kg/m^3
const Scalar U0 = 1.0; // m/s
const Scalar nu = 1e-5; //m^2/s

const Scalar g = 9.80665; // m/s^2
const Vector f(0.0, 0.0, -g);

size_t iter = 10;
const size_t MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

Array1D<Point> pnt;
Array1D<Face> face;
Array1D<Cell> cell;

void init()
{
	//Read grid
	XF_MSH mesh;
	mesh.readFromFile("grid/fluent.msh");
	const size_t NumOfPnt = mesh.numOfNode();
	const size_t NumOfFace = mesh.numOfFace();
	const size_t NumOfCell = mesh.numOfCell();
	pnt.resize(NumOfPnt);
	face.resize(NumOfFace);
	cell.resize(NumOfCell);

	// I.C.

}

void solve()
{
	bool converged = false;
	while(!converged)
	{

	}
}

int main(int argc, char *argv[])
{
	init();
	solve();
	return 0;
}
