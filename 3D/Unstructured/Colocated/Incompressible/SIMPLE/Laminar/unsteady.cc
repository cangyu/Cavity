#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include "xf.h"
#include "Eigen/Dense"

using namespace GridTool;

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;

struct Tensor : public Eigen::Matrix<Scalar, 3, 3>
{
	Scalar xx() const { return this->operator()(0, 0); }
	Scalar xy() const { return this->operator()(0, 1); }
	Scalar xz() const { return this->operator()(0, 2); }
	Scalar yx() const { return this->operator()(1, 0); }
	Scalar yy() const { return this->operator()(1, 1); }
	Scalar yz() const { return this->operator()(1, 2); }
	Scalar zx() const { return this->operator()(2, 0); }
	Scalar zy() const { return this->operator()(2, 1); }
	Scalar zz() const { return this->operator()(2, 2); }

	Scalar &xx() { return this->operator()(0, 0); }
	Scalar &xy() { return this->operator()(0, 1); }
	Scalar &xz() { return this->operator()(0, 2); }
	Scalar &yx() { return this->operator()(1, 0); }
	Scalar &yy() { return this->operator()(1, 1); }
	Scalar &yz() { return this->operator()(1, 2); }
	Scalar &zx() { return this->operator()(2, 0); }
	Scalar &zy() { return this->operator()(2, 1); }
	Scalar &zz() { return this->operator()(2, 2); }
};

template<typename T>
class Array1D : public std::vector<T>
{
public:
	Array1D(size_t n = 0) : std::vector<T>(n) {}
	Array1D(size_t n, const T &val) : std::vector<T>(n, val) {}
	~Array1D() = default;

	/* 1-based indexing */
	T &operator()(int i)
	{
		if (i >= 1)
			return std::vector<T>::at(i - 1);
		else if (i <= -1)
			return std::vector<T>::at(std::vector<T>::size() + i);
		else
			throw std::invalid_argument("Invalid index.");
	}
	const T &operator()(int i) const
	{
		if (i >= 1)
			return std::vector<T>::at(i - 1);
		else if (i <= -1)
			return std::vector<T>::at(std::vector<T>::size() + i);
		else
			throw std::invalid_argument("Invalid index.");
	}
};

struct Cell;

struct Point
{
	size_t index; // 1-based global index.	
	Vector coordinate; // 3D Location.

	Scalar rho;
	Vector U;
	Scalar p;
	Scalar T;

	Vector grad_rho;
	Tensor grad_U;
	Vector grad_T;

	Vector rhoU;
};

struct Face
{
	// 1-based global index.
	size_t index;

	bool atBdry;
	bool rho_isDirichletBC;
	bool U_isDirichletBC;
	bool p_isDirichletBC;
	bool T_isDirichletBC;
	Scalar rho_ghost;
	Vector U_ghost;
	Scalar p_ghost;
	Scalar T_ghost;

	Vector center;
	Scalar area;
	Array1D<Point*> vertex;
	Cell *c0, *c1;
	Vector r0, r1;

	Scalar rho;
	Vector U;
	Scalar p;
	Scalar T;

	Vector grad_rho;
	Tensor grad_U;
	Vector grad_T;

	Vector rhoU;

	Tensor tau;
};

const size_t NumOfTimeLevel = 3;
size_t curTimeLevel = 0;
size_t iter = 0;
const size_t MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

struct Cell
{
	size_t index;

	Vector center;
	Scalar volume;
	Array1D<Point*> vertex;
	Array1D<Face*> surface;
	Array1D<Vector> S;
	Array1D<Cell*> adjCell;

	Scalar rho[NumOfTimeLevel];
	Vector U[NumOfTimeLevel];
	Scalar p[NumOfTimeLevel];
	Scalar T[NumOfTimeLevel];

	Vector grad_rho;
	Tensor grad_U;
	Vector grad_p;
	Vector grad_T;

	Vector rhoU;
	Vector rhoU_star;

	/* Compute gradients using Least-squares approach */
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> rho_QR;
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> U_QR;
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> p_QR;
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> T_QR;
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

// Grid
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

// Physical variables located at geom entities
Array1D<Point> pnt;
Array1D<Face> face;
Array1D<Cell> cell;

// The group of boundary faces
Array1D<Patch> patch; 

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
			f_dst.r0 = f_dst.center - f_dst.c0->center;

		if (f_dst.c1)
			f_dst.r1 = f_dst.center - f_dst.c1->center;
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
		fout << '\t' << cell(i).rho[curTimeLevel];
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U[curTimeLevel].x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U[curTimeLevel].y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U[curTimeLevel].z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).p[curTimeLevel];
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).T[curTimeLevel];
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

/// Extract the solution only. Both size and 
/// connectivity should be consistent with existing mesh.
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
		fin >> cell(i).rho[curTimeLevel];

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U[curTimeLevel].x();

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U[curTimeLevel].y();

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U[curTimeLevel].z();

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).p[curTimeLevel];

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).T[curTimeLevel];

	/* Skip connectivity info */
	// Finalize
	fin.close();
}

void IC()
{
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		auto &n_dst = pnt(i);
		n_dst.rho = rho0;
		n_dst.U = { 0, 0, 0 };
		n_dst.p = P0;
		n_dst.T = T0;
	}
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);
		f_dst.rho = rho0;
		f_dst.U = { 0, 0, 0 };
		f_dst.p = P0;
		f_dst.T = T0;
	}
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c_dst = cell(i);
		c_dst.rho[curTimeLevel] = rho0;
		c_dst.U[curTimeLevel] = { 0, 0, 0 };
		c_dst.p[curTimeLevel] = P0;
		c_dst.T[curTimeLevel] = T0;
	}
}

void BC()
{
	for (auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
				f->U = { U0, V0, W0 };
		}
		else
		{
			for (auto f : e.surface)
				f->U = { 0.0, 0.0, 0.0 };
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
		const size_t nF = c.surface.size();
		Eigen::Matrix<Scalar, Eigen::Dynamic, 3> J_rho, J_U, J_p, J_T;
		J_rho.resize(nF, Eigen::NoChange);
		J_U.resize(nF, Eigen::NoChange);
		J_p.resize(nF, Eigen::NoChange);
		J_T.resize(nF, Eigen::NoChange);

		for (size_t j = 1; j <= nF; ++j)
		{
			auto curFace = c.surface(j);
			auto curCell = c.adjCell(j);

			if (curFace->atBdry)
			{
				const Scalar dx = curFace->center.x() - c.center.x();
				const Scalar dy = curFace->center.y() - c.center.y();
				const Scalar dz = curFace->center.z() - c.center.z();

				const Scalar dx2 = 2 * dx;
				const Scalar dy2 = 2 * dy;
				const Scalar dz2 = 2 * dz;

				if (curFace->rho_isDirichletBC)
					J_rho.row(j - 1) << dx, dy, dz;
				else
					J_rho.row(j - 1) << dx2, dy2, dz2;

				if (curFace->U_isDirichletBC)
					J_U.row(j - 1) << dx, dy, dz;
				else
					J_U.row(j - 1) << dx2, dy2, dz2;

				if (curFace->p_isDirichletBC)
					J_p.row(j - 1) << dx, dy, dz;
				else
					J_p.row(j - 1) << dx2, dy2, dz2;

				if (curFace->T_isDirichletBC)
					J_T.row(j - 1) << dx, dy, dz;
				else
					J_T.row(j - 1) << dx2, dy2, dz2;
			}
			else
			{
				const Scalar dx = curCell->center.x() - c.center.x();
				const Scalar dy = curCell->center.y() - c.center.y();
				const Scalar dz = curCell->center.z() - c.center.z();

				J_rho.row(j - 1) << dx, dy, dz;
				J_U.row(j - 1) << dx, dy, dz;
				J_p.row(j - 1) << dx, dy, dz;
				J_T.row(j - 1) << dx, dy, dz;
			}
		}

		c.rho_QR = J_rho.householderQr();
		c.U_QR = J_U.householderQr();
		c.p_QR = J_p.householderQr();
		c.T_QR = J_T.householderQr();
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

void calcCellGradients()
{
	for (auto &c : cell)
	{
		const size_t nF = c.surface.size();
		Eigen::VectorXd dphi(nF), gphi(nF);

		/* Gradient of density */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->rho_isDirichletBC)
					dphi(i - 1) = curFace->rho - c.rho[curTimeLevel];
				else
					dphi(i - 1) = curFace->rho_ghost - c.rho[curTimeLevel];
			}
			else
				dphi(i - 1) = curCell->rho[curTimeLevel] - c.rho[curTimeLevel];
		}
		gphi = c.rho_QR.solve(dphi);
		c.grad_rho.x() = gphi(0);
		c.grad_rho.y() = gphi(1);
		c.grad_rho.z() = gphi(2);

		/* Gradient of x-dim velocity */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->U_isDirichletBC)
					dphi(i - 1) = curFace->U.x() - c.U[curTimeLevel].x();
				else
					dphi(i - 1) = curFace->U_ghost.x() - c.U[curTimeLevel].x();
			}
			else
				dphi(i - 1) = curCell->U[curTimeLevel].x() - c.U[curTimeLevel].x();
		}
		gphi = c.U_QR.solve(dphi);
		c.grad_U.xx() = gphi(0);
		c.grad_U.yx() = gphi(1);
		c.grad_U.zx() = gphi(2);

		/* Gradient of y-dim velocity */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->U_isDirichletBC)
					dphi(i - 1) = curFace->U.y() - c.U[curTimeLevel].y();
				else
					dphi(i - 1) = curFace->U_ghost.y() - c.U[curTimeLevel].y();
			}
			else
				dphi(i - 1) = curCell->U[curTimeLevel].y() - c.U[curTimeLevel].y();
		}
		gphi = c.U_QR.solve(dphi);
		c.grad_U.xy() = gphi(0);
		c.grad_U.yy() = gphi(1);
		c.grad_U.zy() = gphi(2);

		/* Gradient of z-dim velocity */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->U_isDirichletBC)
					dphi(i - 1) = curFace->U.z() - c.U[curTimeLevel].z();
				else
					dphi(i - 1) = curFace->U_ghost.z() - c.U[curTimeLevel].z();
			}
			else
				dphi(i - 1) = curCell->U[curTimeLevel].z() - c.U[curTimeLevel].z();
		}
		gphi = c.U_QR.solve(dphi);
		c.grad_U.xz() = gphi(0);
		c.grad_U.yz() = gphi(1);
		c.grad_U.zz() = gphi(2);

		/* Gradient of pressure */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->p_isDirichletBC)
					dphi(i - 1) = curFace->p - c.p[curTimeLevel];
				else
					dphi(i - 1) = curFace->p_ghost - c.p[curTimeLevel];
			}
			else
				dphi(i - 1) = curCell->p[curTimeLevel] - c.p[curTimeLevel];
		}
		gphi = c.p_QR.solve(dphi);
		c.grad_p.x() = gphi(0);
		c.grad_p.y() = gphi(1);
		c.grad_p.z() = gphi(2);

		/* Gradient of temperature */
		for (size_t i = 1; i <= nF; ++i)
		{
			auto curFace = c.surface(i);
			auto curCell = c.adjCell(i);

			if (curFace->atBdry)
			{
				if (curFace->T_isDirichletBC)
					dphi(i - 1) = curFace->T - c.T[curTimeLevel];
				else
					dphi(i - 1) = curFace->T_ghost - c.T[curTimeLevel];
			}
			else
				dphi(i - 1) = curCell->T[curTimeLevel] - c.T[curTimeLevel];
		}
		gphi = c.T_QR.solve(dphi);
		c.grad_T.x() = gphi(0);
		c.grad_T.y() = gphi(1);
		c.grad_T.z() = gphi(2);
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

			convection_flux += curFace->rhoU * curFace->U.dot(c.S(j));

			pressure_flux -= c.S(j) * curFace->p;

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
		calcCellGradients();
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
