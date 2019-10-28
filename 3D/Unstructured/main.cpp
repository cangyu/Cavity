#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf.hpp"
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

void readMSH()
{
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		const auto &n_src = mesh.node(i);
		auto &n_dst = pnt(i);
		n_dst.index = i;

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
		f_dst.index = i;

		const auto &c_src = f_src.center;
		auto &c_dst = f_dst.center;
		c_dst.x() = c_src.x();
		c_dst.y() = c_src.y();
		c_dst.z() = c_src.z();

		f_dst.area = f_src.area;

		const size_t N1 = f_src.includedNode.size();
		f_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			f_dst.vertex(i) = &pnt(f_src.includedNode(i));
	}
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		const auto &c_src = mesh.cell(i);
		auto &c_dst = cell(i);
		c_dst.index = i;

		const auto &centroid_src = c_src.center;
		auto &centroid_dst = c_dst.center;
		centroid_dst.x() = centroid_src.x();
		centroid_dst.y() = centroid_src.y();
		centroid_dst.z() = centroid_src.z();

		const size_t N1 = c_src.includedNode.size();
		c_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			c_dst.vertex(i) = &pnt(c_src.includedNode(i));

		const size_t N2 = c_src.includedFace.size();
		c_dst.surface.resize(N2);
		for (size_t i = 1; i <= N2; ++i)
			c_dst.surface(i) = &face(c_src.includedFace(i));
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

void readTECPLOT()
{
	// Only the solution part is extracted, and its connectivity
	// should be consistent with that given by existing mesh.

}

void writeTECPLOT()
{
	static const size_t RECORD_PER_LINE = 10;

	const std::string fn = "flow" + std::to_string(iter) + ".dat";
	std::ofstream fout(fn);
	if (fout.fail())
		throw std::runtime_error("Failed to open target output file: \"" + fn + "\"!");

	fout << R"(TITLE="3D Cavity flow at t=)" + std::to_string(t) + R"(s")" << std::endl;
	fout << "FILETYPE=FULL" << std::endl;
	fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;
	fout << "ZONE NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL, [4-9]=CELLCENTERED)" << std::endl;

	// X-Coordinates
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).coordinate.x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Y-Coordinates
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).coordinate.y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Z-Coordinates
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).coordinate.z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Density
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).density;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).velocity.x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).velocity.y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).velocity.z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).pressure;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).temperature;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Connectivity Information
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		const auto v = cell(i).vertex;
		for (const auto &e : v)
			fout << '\t' << e->index;
		fout << std::endl;
	}

	fout.close();
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
	readMSH();
	IC();
	BC();
}

void solve()
{
	bool converged = false;
	writeTECPLOT();
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
