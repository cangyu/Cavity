#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf.hpp"
#include "natural_array.hpp"
#include "math_type.hpp"

struct Point
{
	Vector coordinate;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Face
{
	Vector center;
	Scalar area;
	Array1D<Point*> vertex;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Cell
{
	Vector center;
	Array1D<Point*> vertex;
	Array1D<Face*> surface;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Patch
{
	std::string name;
	int BC;
	Array1D<Face *> surface;
};

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

//Load grid
const XF::MESH mesh("grid/fluent.msh");
const size_t NumOfPnt = mesh.numOfNode();
const size_t NumOfFace = mesh.numOfFace();
const size_t NumOfCell = mesh.numOfCell();

Array1D<Point> pnt(NumOfPnt);
Array1D<Face> face(NumOfFace);
Array1D<Cell> cell(NumOfCell);
Array1D<Patch> patch; // The group of boundary faces

void init()
{
	// Read grid data
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		const auto &n_src = mesh.node(i);
		auto &n_dst = pnt(i);

		const auto &c_src = n_src.coordinate;
		auto &c_dst = n_dst.coordinate;
		c_dst.x() = c_src.x();
		c_dst.y() = c_src.y();
		c_dst.z() = c_src.z();
	}
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		const auto &f_src = mesh.face(i);
		auto &f_dst = face(i);

		const auto &c_src = f_src.center;
		auto &c_dst = f_dst.center;
		c_dst.x() = c_src.x();
		c_dst.y() = c_src.y();
		c_dst.z() = c_src.z();

		f_dst.area = f_src.area;
	}
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		const auto &c_src = mesh.cell(i);
		auto &c_dst = cell(i);

		const auto &centroid_src = c_src.center;
		auto &centroid_dst = c_dst.center;
		centroid_dst.x() = centroid_src.x();
		centroid_dst.y() = centroid_src.y();
		centroid_dst.z() = centroid_src.z();
	}

	// I.C.
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		auto &n_dst = pnt(i);
	}
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);
	}
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c_dst = cell(i);
	}

	// B.C.

}

void solve()
{
	bool converged = false;
	while (!converged)
	{

	}
}

int main(int argc, char *argv[])
{
	init();
	solve();
	return 0;
}
