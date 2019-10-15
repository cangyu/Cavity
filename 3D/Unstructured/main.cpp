#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf.hpp"
#include "tecplot.hpp"
#include "geom_entity.hpp"

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
const Scalar rho0 = 1.225; //kg/m^3
const Scalar U0 = 1.0; // m/s
const Scalar V0 = 0.0; // m/s
const Scalar W0 = 0.0; // m/s
const Scalar nu = 1e-5; //m^2/s

const Scalar g = 9.80665; // m/s^2
const Vector f(0.0, 0.0, -g);

size_t iter = 0;
const size_t MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

// Grid
const XF::MESH mesh("grid/fluent.msh");
const size_t NumOfPnt = mesh.numOfNode();
const size_t NumOfFace = mesh.numOfFace();
const size_t NumOfCell = mesh.numOfCell();

// Physical variables located at geom entities
Array1D<Point> pnt(NumOfPnt);
Array1D<Face> face(NumOfFace);
Array1D<Cell> cell(NumOfCell);
Array1D<Patch> patch; // The group of boundary faces

void loadMesh()
{
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

		const size_t N1 = f_src.node.size();
		f_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			f_dst.vertex(i) = &pnt(f_src.node(i));
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

		const size_t N1 = c_src.node.size();
		c_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			c_dst.vertex(i) = &pnt(c_src.node(i));

		const size_t N2 = c_src.face.size();
		c_dst.surface.resize(N2);
		for (size_t i = 1; i <= N2; ++i)
			c_dst.surface(i) = &face(c_src.face(i));
	}
	size_t NumOfPatch = 0;
	for (size_t i = 1; i <= mesh.numOfZone(); ++i)
	{
		const auto &z_src = mesh.zone(i);

		const auto curZoneIdx = z_src.ID;
		auto curZonePtr = z_src.obj;
		auto curFace = dynamic_cast<XF::FACE*>(curZonePtr);
		if (curFace == nullptr)
			continue;

		if (curFace->identity() != XF::SECTION::FACE || curFace->zone() != curZoneIdx)
			throw std::runtime_error("Inconsistency detected.");

		if (XF::BC::INTERIOR != XF::BC::str2idx(z_src.type))
			++NumOfPatch;
	}
	patch.resize(NumOfPatch);
	size_t cnt = 0;
	for (int curZoneIdx = 1; curZoneIdx <= mesh.numOfZone(); ++curZoneIdx)
	{
		const auto &curZone = mesh.zone(curZoneIdx);
		auto curFace = dynamic_cast<XF::FACE*>(curZone.obj);
		if (curFace == nullptr || XF::BC::INTERIOR == XF::BC::str2idx(curZone.type))
			continue;

		auto &p_dst = patch[cnt];
		p_dst.name = curZone.name;
		p_dst.BC = XF::BC::str2idx(curZone.type);
		p_dst.surface.resize(curFace->num());
		const auto loc_first = curFace->first_index();
		const auto loc_last = curFace->last_index();
		for (auto i = loc_first; i <= loc_last; ++i)
			p_dst.surface.at(i - loc_first) = &face(i);

		++cnt;
	}
}

void IC()
{
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		auto &n_dst = pnt(i);
		n_dst.density = rho0;
		n_dst.velocity = { 0, 0, 0 };
		n_dst.pressure = P0;
		n_dst.temperature = T0;
	}
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);
		f_dst.density = rho0;
		f_dst.velocity = { 0, 0, 0 };
		f_dst.pressure = P0;
		f_dst.temperature = T0;
	}
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c_dst = cell(i);
		c_dst.density = rho0;
		c_dst.velocity = { 0, 0, 0 };
		c_dst.pressure = P0;
		c_dst.temperature = T0;
	}
}

void BC()
{
	for (auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
				f->velocity = { U0, V0, W0 };
		}
		else
		{
			for (auto f : e.surface)
				f->velocity = { 0.0, 0.0, 0.0 };
		}
	}
}

void init()
{
	loadMesh();
	IC();
	BC();
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
