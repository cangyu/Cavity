#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include "xf.h"
#include "Eigen/Dense"
#include "natural_array.h"
#include "math_type.h"

using namespace GridTool;

struct Cell;
struct Point
{
	size_t index;

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
	size_t index;
	bool atBdry;

	Vector center;
	Scalar area;
	Array1D<Point*> vertex;
	Cell *c0, *c1;
	Vector r0, r1;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;

	Tensor tau;

	Vector rhoU;
};
struct Cell
{
	size_t index;

	Vector center;
	Scalar volume;
	Array1D<Point*> vertex;
	Array1D<Face*> surface;
	Array1D<Vector> S;
	Array1D<Cell*> adjCell;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;

	Vector rhoU;
	Vector rhoU_star;

	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> J; // Used for computing gradients using Least-squares approach.
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> QR;
};
struct Patch
{
	std::string name;
	int BC;
	Array1D<Face*> surface;
};

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
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

// Physical variables located at geom entities
Array1D<Point> pnt;
Array1D<Face> face;
Array1D<Cell> cell;
Array1D<Patch> patch; // The group of boundary faces

void readMSH()
{
	// Load ANSYS Fluent mesh using external independent package.
	const XF::MESH mesh("grid0.msh");

	// Update counting of geom elements.
	NumOfPnt = mesh.numOfNode();
	NumOfFace = mesh.numOfFace();
	NumOfCell = mesh.numOfCell();

	// Allocate memory for geom information and related physical variables.
	pnt.resize(NumOfPnt);
	face.resize(NumOfFace);
	cell.resize(NumOfCell);

	// Update point information.
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		const auto &n_src = mesh.node(i);
		auto &n_dst = pnt(i);

		// Assign node index.
		n_dst.index = i;

		const auto &c_src = n_src.coordinate;
		auto &c_dst = n_dst.coordinate;

		// Node location.
		c_dst.x() = c_src.x();
		c_dst.y() = c_src.y();
		c_dst.z() = c_src.z();
	}

	// Update face information.
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		const auto &f_src = mesh.face(i);
		auto &f_dst = face(i);

		// Assign face index.
		f_dst.index = i;
		f_dst.atBdry = f_src.atBdry;

		const auto &c_src = f_src.center;
		auto &c_dst = f_dst.center;

		// Face center location.
		c_dst.x() = c_src.x();
		c_dst.y() = c_src.y();
		c_dst.z() = c_src.z();

		// Face area.
		f_dst.area = f_src.area;

		// Face included nodes.
		const size_t N1 = f_src.includedNode.size();
		f_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			f_dst.vertex(i) = &pnt(f_src.includedNode(i));

		// Face adjacent 2 cells.
		f_dst.c0 = f_src.leftCell ? &cell(f_src.leftCell) : nullptr;
		f_dst.c1 = f_src.rightCell ? &cell(f_src.rightCell) : nullptr;
	}

	// Update cell information.
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		const auto &c_src = mesh.cell(i);
		auto &c_dst = cell(i);

		// Assign cell index.
		c_dst.index = i;

		const auto &centroid_src = c_src.center;
		auto &centroid_dst = c_dst.center;

		// Cell center location.
		centroid_dst.x() = centroid_src.x();
		centroid_dst.y() = centroid_src.y();
		centroid_dst.z() = centroid_src.z();

		// Cell included nodes.
		const size_t N1 = c_src.includedNode.size();
		c_dst.vertex.resize(N1);
		for (size_t i = 1; i <= N1; ++i)
			c_dst.vertex(i) = &pnt(c_src.includedNode(i));

		// Cell included faces.
		const size_t N2 = c_src.includedFace.size();
		c_dst.surface.resize(N2);
		for (size_t i = 1; i <= N2; ++i)
			c_dst.surface(i) = &face(c_src.includedFace(i));

		// Cell adjacent cells.
		c_dst.adjCell.resize(N2);
		for (size_t i = 1; i <= N2; ++i)
		{
			auto adjIdx = c_src.adjacentCell(i);
			c_dst.adjCell(i) = adjIdx ? &cell(adjIdx) : nullptr;
		}
	}

	// Cell center to face center vectors 
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);

		if (f_dst.c0)
		{
			f_dst.r0.x() = f_dst.center.x() - f_dst.c0->center.x();
			f_dst.r0.y() = f_dst.center.y() - f_dst.c0->center.y();
			f_dst.r0.z() = f_dst.center.z() - f_dst.c0->center.z();
		}

		if (f_dst.c1)
		{
			f_dst.r1.x() = f_dst.center.x() - f_dst.c1->center.x();
			f_dst.r1.y() = f_dst.center.y() - f_dst.c1->center.y();
			f_dst.r1.z() = f_dst.center.z() - f_dst.c1->center.z();
		}
	}

	// Count valid patches.
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

	// Update boundary patch information.
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

void writeTECPLOT()
{
	static const size_t RECORD_PER_LINE = 10;

	const std::string fn("flow" + std::to_string(iter) + ".dat");
	std::ofstream fout(fn);
	if (fout.fail())
		throw std::runtime_error("Failed to open target output file: \"" + fn + "\"!");

	fout << R"(TITLE="3D Cavity flow at t=)" + std::to_string(t) + R"(s")" << std::endl;
	fout << "FILETYPE=FULL" << std::endl;
	fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;
	fout << "ZONE T=\"Interior\", NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL, [4-9]=CELLCENTERED)" << std::endl;

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

// Extract the solution only. Both size and 
// connectivity should be consistent with existing mesh.
void readTECPLOT(const std::string &fn)
{
	std::ifstream fin(fn);
	if (fin.fail())
		throw std::runtime_error("Failed to open target input file \"" + fn + "\"!");

	std::string tmp;
	Scalar var;

	/* Skip header */
	std::getline(fin, tmp);
	std::getline(fin, tmp);
	std::getline(fin, tmp);
	std::getline(fin, tmp);

	/* Skip coordinates */
	for (int k = 0; k < 3; ++k)
		for (size_t i = 1; i <= NumOfPnt; ++i)
			fin >> var;

	/* Load cell-centered data */
	// Density
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).density;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).velocity.x();

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).velocity.y();

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).velocity.z();

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).pressure;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).temperature;

	/* Skip connectivity info */
	// Finalize
	fin.close();
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

	// Compute J and its QR decomposition for each cell
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		auto &curJ = c.J;
		const size_t nF = c.surface.size();
		curJ.resize(nF, Eigen::NoChange);
		for (size_t j = 1; j <= nF; ++j)
		{
			auto curFace = c.surface(j);
			auto curCell = c.adjCell(j);

			if (curFace->atBdry)
				curJ.row(j - 1) << curFace->center.x() - c.center.x(), curFace->center.y() - c.center.y(), curFace->center.z() - c.center.z();
			else
				curJ.row(j - 1) << curCell->center.x() - c.center.x(), curCell->center.y() - c.center.y(), curCell->center.z() - c.center.z();
		}
		c.QR = curJ.householderQr();
	}
}

Scalar calcTimeStep()
{
	Scalar ret = 0.0;

	return ret;
}

bool checkConvergence()
{
	bool ret = false;

	return ret;
}

void calcGradients()
{
	for (auto &c : cell)
	{
		const size_t nF = c.surface.size();
		Eigen::VectorXd dphi(nF);

		// Gradient of Density.
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
				dphi(i - 1) = curFace->density - c.density;
			else
				dphi(i - 1) = curCell->density - c.density;
		}
		auto drho = c.QR.solve(dphi);
		c.density_gradient.x() = drho(0);
		c.density_gradient.y() = drho(1);
		c.density_gradient.z() = drho(2);
	}
}

void SIMPLE()
{
	/********************************************** Prediction Step ***************************************************/
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		const size_t Nf = c.surface.size();

		Vector convection_flux;
		Vector pressure_flux;
		Vector viscous_flux;
		for (size_t j = 1; j <= Nf; ++j)
		{
			const auto curFace = c.surface(j);

			convection_flux += curFace->rhoU * dot_product(curFace->velocity, c.S(j));

			pressure_flux -= c.S(j) * curFace->pressure;

			Vector tmp;
			dot_product(c.S(j), curFace->tau, tmp);
			viscous_flux += tmp;
		}

		c.rhoU_star = c.rhoU + (pressure_flux + viscous_flux - convection_flux) * (dt / c.volume);
	}

	/********************************************** Correction Step ***************************************************/
	

	/*************************************************** Update *******************************************************/

}

void solve()
{
	static const size_t OUTPUT_GAP = 100;

	bool converged = false;
	writeTECPLOT();
	while (!converged)
	{
		std::cout << "Iter" << ++iter << ":" << std::endl;
		dt = calcTimeStep();
		std::cout << "\tt=" << t << "s, dt=" << dt << "s" << std::endl;
		calcGradients();
		SIMPLE();
		t += dt;
		converged = checkConvergence();
		if (converged || !(iter % OUTPUT_GAP))
			writeTECPLOT();
	}
	std::cout << "Converged!" << std::endl;
}

int main(int argc, char *argv[])
{
	init();
	solve();

	return 0;
}
