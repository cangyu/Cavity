#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "xf.h"
#include "Eigen/Dense"

/*************************************************Global types and variables******************************************/
/* 1-based array */
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

/* Math types */
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

/* BC types */
typedef enum { Dirichlet, Neumann, Robin } BC_Category;

struct unsupported_boundary_condition : public std::invalid_argument
{
	unsupported_boundary_condition(BC_Category x) : std::invalid_argument(std::to_string((int)x) + ".") {}
};

/* Geom elements */
struct Cell;

struct Point
{
	// 1-based global index
	size_t index;

	// 3D Location
	Vector coordinate;

	// Physical variables
	Scalar rho;
	Vector U;
	Scalar p;
	Scalar T;
};

struct Face
{
	// 1-based global index
	size_t index;

	// 3D location of face centroid
	Vector center;

	// Area of the face element
	Scalar area;

	/* Connectivity */
	Array1D<Point*> vertex;
	Cell *c0, *c1;

	// Displacement vector
	Vector r0, r1;

	// Boundary flags
	bool atBdry;
	BC_Category rho_BC = Dirichlet;
	std::array<BC_Category, 3> U_BC = { Dirichlet, Dirichlet, Dirichlet };
	BC_Category p_BC = Neumann;
	BC_Category T_BC = Neumann;

	// Ghost variables if needed
	Scalar rho_ghost;
	Vector U_ghost;
	Scalar p_ghost;
	Scalar T_ghost;

	// Physical variables
	Scalar rho;
	Vector U;
	Scalar p;
	Scalar T;

	// Averaged physical variables
	Scalar rho_av;
	Vector U_av;
	Scalar p_av;
	Scalar T_av;

	// Gradient of physical variables
	Vector grad_rho;
	Tensor grad_U;
	Vector grad_p;
	Vector grad_T;

	// Averaged gradient of physical variables.
	// In general, they function as prediction of true gradients.
	Vector grad_rho_av;
	Tensor grad_U_av;
	Vector grad_p_av;
	Vector grad_T_av;
};

struct Cell
{
	// 1-based global index
	size_t index;

	// 3D location of cell centroid
	Vector center;

	// Volume of the cell element
	Scalar volume;

	/* Connectivity */
	Array1D<Point*> vertex;
	Array1D<Face*> surface;
	Array1D<Cell*> adjCell;

	// Surface vector
	Array1D<Vector> S;

	/* Compute gradients using Least-squares approach */
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> rho_QR;
	std::array<Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>>, 3> U_QR;
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> p_QR;
	Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>> T_QR;

	/* Variables at current time-level */
	// Physical variables
	Scalar rho;
	Vector U;
	Scalar p;
	Scalar T;
	Vector rhoU;
	Scalar rhoE;
	Scalar rhoH;

	/* Runge-Kutta temporary variables */
	// Physical properties
	Scalar mu_rk;

	// Physical variables
	Scalar rho_rk;
	Vector U_rk;
	Scalar p_rk;
	Scalar T_rk;

	// Gradients
	Vector grad_rho_rk;
	Tensor grad_U_rk;
	Vector grad_p_rk;
	Vector grad_T_rk;

	// Equation residuals
	Scalar continuity_res;
	Vector momentum_res;
	Scalar energy_res;
};

struct Patch
{
	std::string name;
	int BC;
	Array1D<Face*> surface;
};

/* Iteration timing and counting */
size_t iter = 0;
const size_t MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

/* Constant field variables */
const Scalar rho0 = 1.225; //kg/m^3
const Vector U0 = { 1.0, 0.0, 0.0 }; // m/s
const Scalar P0 = 101325.0; // Pa
const Scalar T0 = 300.0; // K

/* Gravity */
const Scalar g = 9.80665; // m/s^2

/* Grid utilities */
using namespace GridTool;

struct empty_connectivity : public std::runtime_error
{
	empty_connectivity(size_t idx) : std::runtime_error("Both c0 and c1 are NULL on face " + std::to_string(idx) + ".") {}
};

size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

Array1D<Point> pnt; // Node objects
Array1D<Face> face; // Face objects
Array1D<Cell> cell; // Cell objects
Array1D<Patch> patch; // Group of boundary faces

/****************************************************Functions********************************************************/
/// Weighted averaged between 2 objects.
template<typename T>
static T relaxation(const T &a, const T &b, Scalar x)
{
	return (1.0 - x) * a + x * b;
}

/// Dynamic viscosity of ideal gas.
Scalar Sutherland(Scalar T)
{
	return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
}

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
		c_dst.rho = rho0;
		c_dst.U = { 0, 0, 0 };
		c_dst.p = P0;
		c_dst.T = T0;
	}
}

void BC()
{
	for (auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
				f->U = U0;
		}
		else
		{
			for (auto f : e.surface)
				f->U = { 0.0, 0.0, 0.0 };
		}
	}
}

// Compute J and its QR decomposition for each cell
void calcLeastSquareCoef()
{
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

				if (curFace->rho_BC == Dirichlet)
					J_rho.row(j - 1) << dx, dy, dz;
				else
					J_rho.row(j - 1) << dx2, dy2, dz2;

				if (curFace->U_BC == Dirichlet)
					J_U.row(j - 1) << dx, dy, dz;
				else
					J_U.row(j - 1) << dx2, dy2, dz2;

				if (curFace->p_BC == Dirichlet)
					J_p.row(j - 1) << dx, dy, dz;
				else
					J_p.row(j - 1) << dx2, dy2, dz2;

				if (curFace->T_BC == Dirichlet)
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

Vector interpGradientToFace(const Vector &grad_phi_f_av, Scalar phi_C, Scalar phi_F, const Vector &e_CF, Scalar d_CF)
{
	return grad_phi_f_av + ((phi_F - phi_C) / d_CF - grad_phi_f_av.dot(e_CF))*e_CF;
}

// Spatial discretization
void calcFaceValue()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{

		}
		else
		{
			// weighting coefficient
			const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

			// pressure
			const Scalar p_0 = f.c0->p[curTimeLevel] + f.c0->grad_p.dot(f.r0);
			const Scalar p_1 = f.c1->p[curTimeLevel] + f.c1->grad_p.dot(f.r1);
			f.p = relaxation(p_0, p_1, 0.5);

			// temperature
			const Scalar T_0 = f.c0->T[curTimeLevel] + f.c0->grad_T.dot(f.r0);
			const Scalar T_1 = f.c1->T[curTimeLevel] + f.c1->grad_T.dot(f.r1);
			f.T = relaxation(T_0, T_1, ksi);

			// velocity
			const Vector U_0 = f.c0->U[curTimeLevel] + f.r0.transpose()*f.c0->grad_U;
			const Vector U_1 = f.c1->U[curTimeLevel] + f.r1.transpose()*f.c1->grad_U;
			f.U_av = 0.5 * (U_0 + U_1);
			f.U = f.U_av;

			// density

		}
	}
}

void calcFaceGradient()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{
			/* Gradients at boundary face */
			if (f.c0)
			{
				const Vector &r_C = f.c0->center;
				const Vector &r_F = f.center;
				Vector e_CF = r_F - r_C;
				const Scalar d_CF = e_CF.norm();
				e_CF /= d_CF;

				// density
				if (f.rho_BC == Dirichlet)
				{
					const Scalar rho_C = f.c0->rho_rk;
					const Scalar rho_F = f.rho;
					f.grad_rho = interpGradientToFace(f.grad_rho_av, rho_C, rho_F, e_CF, d_CF);
				}

				// velocity-x
				if (f.U_BC[0] == Dirichlet)
				{
					const Scalar u_C = f.c0->U_rk.x();
					const Scalar u_F = f.U.x();
					f.grad_U.col(0) = interpGradientToFace(f.grad_U_av.col(0), u_C, u_F, e_CF, d_CF);
				}

				// velocity-y
				if (f.U_BC[1] == Dirichlet)
				{
					const Scalar v_C = f.c0->U_rk.y();
					const Scalar v_F = f.U.y();
					f.grad_U.col(1) = interpGradientToFace(f.grad_U_av.col(1), v_C, v_F, e_CF, d_CF);
				}

				// velocity-z
				if (f.U_BC[2] == Dirichlet)
				{
					const Scalar w_C = f.c0->U_rk.z();
					const Scalar w_F = f.U.z();
					f.grad_U.col(2) = interpGradientToFace(f.grad_U_av.col(2), w_C, w_F, e_CF, d_CF);
				}

				// pressure
				if (f.p_BC == Dirichlet)
				{
					const Scalar p_C = f.c0->p_rk;
					const Scalar p_F = f.p;
					f.grad_p = interpGradientToFace(f.grad_p_av, p_C, p_F, e_CF, d_CF);
				}

				// temperature
				if (f.T_BC == Dirichlet)
				{
					const Scalar T_C = f.c0->T_rk;
					const Scalar T_F = f.T;
					f.grad_T = interpGradientToFace(f.grad_T_av, T_C, T_F, e_CF, d_CF);
				}
			}
			else if (f.c1)
			{
				const Vector &r_C = f.c1->center;
				const Vector &r_F = f.center;
				Vector e_CF = r_F - r_C;
				const Scalar d_CF = e_CF.norm();
				e_CF /= d_CF;

				// density
				if (f.rho_BC == Dirichlet)
				{
					const Scalar rho_C = f.c1->rho_rk;
					const Scalar rho_F = f.rho;
					f.grad_rho = interpGradientToFace(f.grad_rho_av, rho_C, rho_F, e_CF, d_CF);
				}

				// velocity-x
				if (f.U_BC[0] == Dirichlet)
				{
					const Scalar u_C = f.c1->U_rk.x();
					const Scalar u_F = f.U.x();
					f.grad_U.col(0) = interpGradientToFace(f.grad_U_av.col(0), u_C, u_F, e_CF, d_CF);
				}

				// velocity-y
				if (f.U_BC[1] == Dirichlet)
				{
					const Scalar v_C = f.c1->U_rk.y();
					const Scalar v_F = f.U.y();
					f.grad_U.col(1) = interpGradientToFace(f.grad_U_av.col(1), v_C, v_F, e_CF, d_CF);
				}

				// velocity-z
				if (f.U_BC[2] == Dirichlet)
				{
					const Scalar w_C = f.c1->U_rk.z();
					const Scalar w_F = f.U.z();
					f.grad_U.col(2) = interpGradientToFace(f.grad_U_av.col(2), w_C, w_F, e_CF, d_CF);
				}

				// pressure
				if (f.p_BC == Dirichlet)
				{
					const Scalar p_C = f.c1->p_rk;
					const Scalar p_F = f.p;
					f.grad_p = interpGradientToFace(f.grad_p_av, p_C, p_F, e_CF, d_CF);
				}

				// temperature
				if (f.T_BC == Dirichlet)
				{
					const Scalar T_C = f.c1->T_rk;
					const Scalar T_F = f.T;
					f.grad_T = interpGradientToFace(f.grad_T_av, T_C, T_F, e_CF, d_CF);
				}
			}
			else
				throw empty_connectivity(f.index);
		}
		else
		{
			/* Gradients at internal face */
			const Vector &r_C = f.c0->center;
			const Vector &r_F = f.c1->center;
			Vector e_CF = r_F - r_C;
			const Scalar d_CF = e_CF.norm();
			e_CF /= d_CF;

			// density
			const Scalar rho_C = f.c0->rho_rk;
			const Scalar rho_F = f.c1->rho_rk;
			f.grad_rho = interpGradientToFace(f.grad_rho_av, rho_C, rho_F, e_CF, d_CF);

			// velocity-x
			const Scalar u_C = f.c0->U_rk.x();
			const Scalar u_F = f.c1->U_rk.x();
			f.grad_U.col(0) = interpGradientToFace(f.grad_U_av.col(0), u_C, u_F, e_CF, d_CF);

			// velocity-y
			const Scalar v_C = f.c0->U_rk.y();
			const Scalar v_F = f.c1->U_rk.y();
			f.grad_U.col(1) = interpGradientToFace(f.grad_U_av.col(1), v_C, v_F, e_CF, d_CF);

			// velocity-z
			const Scalar w_C = f.c0->U_rk.z();
			const Scalar w_F = f.c1->U_rk.z();
			f.grad_U.col(2) = interpGradientToFace(f.grad_U_av.col(2), w_C, w_F, e_CF, d_CF);

			// pressure
			const Scalar p_C = f.c0->p_rk;
			const Scalar p_F = f.c1->p_rk;
			f.grad_p = interpGradientToFace(f.grad_p_av, p_C, p_F, e_CF, d_CF);

			// temperature
			const Scalar T_C = f.c0->T_rk;
			const Scalar T_F = f.c1->T_rk;
			f.grad_T = interpGradientToFace(f.grad_T_av, T_C, T_F, e_CF, d_CF);
		}
	}
}

void calcFaceAveragedGradient()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{
			/* Averaged gradients at boundary face */
			if (f.c0)
			{
				// density
				if (f.rho_BC == Dirichlet)
					f.grad_rho_av = f.c0->grad_rho_rk;
				else if (f.rho_BC == Neumann)
					f.grad_rho_av = f.grad_rho;
				else
					throw unsupported_boundary_condition(f.rho_BC);

				// velocity-x
				if (f.U_BC[0] == Dirichlet)
					f.grad_U_av.col(0) = f.c0->grad_U_rk.col(0);
				else if (f.U_BC[0] == Neumann)
					f.grad_U_av.col(0) = f.grad_U.col(0);
				else
					throw unsupported_boundary_condition(f.U_BC[0]);

				// velocity-y
				if (f.U_BC[1] == Dirichlet)
					f.grad_U_av.col(1) = f.c0->grad_U_rk.col(1);

				// velocity-z
				if (f.U_BC[2] == Dirichlet)
					f.grad_U_av.col(2) = f.c0->grad_U_rk.col(2);

				// pressure
				if (f.p_BC == Dirichlet)
					f.grad_p_av = f.c0->grad_p_rk;

				// temperature
				if (f.T_BC == Dirichlet)
					f.grad_T_av = f.c0->grad_T_rk;
			}
			else if (f.c1)
			{
				// density
				if (f.rho_BC == Dirichlet)
					f.grad_rho_av = f.c1->grad_rho_rk;

				// velocity-x
				if (f.U_BC[0] == Dirichlet)
					f.grad_U_av.col(0) = f.c1->grad_U_rk.col(0);

				// velocity-y
				if (f.U_BC[1] == Dirichlet)
					f.grad_U_av.col(1) = f.c1->grad_U_rk.col(1);

				// velocity-z
				if (f.U_BC[0] == Dirichlet)
					f.grad_U_av.col(2) = f.c1->grad_U_rk.col(2);

				// pressure
				if (f.p_BC == Dirichlet)
					f.grad_p_av = f.c1->grad_p_rk;

				// temperature
				if (f.T_BC == Dirichlet)
					f.grad_T_av = f.c1->grad_T_rk;
			}
			else
				throw empty_connectivity(f.index);
		}
		else
		{
			/* Averaged gradients at internal face */
			const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

			// density
			f.grad_rho_av = relaxation(f.c0->grad_rho_rk, f.c1->grad_rho_rk, ksi);

			// velocity
			f.grad_U_av = relaxation(f.c0->grad_U_rk, f.c1->grad_U_rk, ksi);

			// pressure
			f.grad_p_av = relaxation(f.c0->grad_p_rk, f.c1->grad_p_rk, ksi);

			// temperature
			f.grad_T_av = relaxation(f.c0->grad_T_rk, f.c1->grad_T_rk, ksi);
		}
	}
}

void calcCellGradient()
{
	for (auto &c : cell)
	{
		const size_t nF = c.surface.size();
		Eigen::VectorXd dphi(nF);

		/* Gradient of density */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->rho_BC == Dirichlet)
					dphi(i) = curFace->rho - c.rho_rk;
				else
					dphi(i) = curFace->rho_ghost - c.rho_rk;
			}
			else
				dphi(i) = curAdjCell->rho_rk - c.rho_rk;
		}
		c.grad_rho_rk = c.rho_QR.solve(dphi);

		/* Gradient of x-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[0] == Dirichlet)
					dphi(i) = curFace->U.x() - c.U_rk.x();
				else
					dphi(i) = curFace->U_ghost.x() - c.U_rk.x();
			}
			else
				dphi(i) = curAdjCell->U_rk.x() - c.U_rk.x();
		}
		c.grad_U_rk.col(0) = c.U_QR[0].solve(dphi);

		/* Gradient of y-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[1] == Dirichlet)
					dphi(i) = curFace->U.y() - c.U_rk.y();
				else
					dphi(i) = curFace->U_ghost.y() - c.U_rk.y();
			}
			else
				dphi(i) = curAdjCell->U_rk.y() - c.U_rk.y();
		}
		c.grad_U_rk.col(1) = c.U_QR[1].solve(dphi);

		/* Gradient of z-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[2] == Dirichlet)
					dphi(i) = curFace->U.z() - c.U_rk.z();
				else
					dphi(i) = curFace->U_ghost.z() - c.U_rk.z();
			}
			else
				dphi(i) = curAdjCell->U_rk.z() - c.U_rk.z();
		}
		c.grad_U_rk.col(2) = c.U_QR[2].solve(dphi);

		/* Gradient of pressure */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->p_BC == Dirichlet)
					dphi(i) = curFace->p - c.p_rk;
				else
					dphi(i) = curFace->p_ghost - c.p_rk;
			}
			else
				dphi(i) = curAdjCell->p_rk - c.p_rk;
		}
		c.grad_p_rk = c.p_QR.solve(dphi);

		/* Gradient of temperature */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->T_BC == Dirichlet)
					dphi(i) = curFace->T - c.T_rk;
				else
					dphi(i) = curFace->T_ghost - c.T_rk;
			}
			else
				dphi(i) = curAdjCell->T_rk - c.T_rk;
		}
		c.grad_T_rk = c.T_QR.solve(dphi);
	}
}

void calcCellProperty()
{
	for (auto &c : cell)
	{
		c.mu_rk = Sutherland(c.T);
	}
}

void calcFaceGhostVariable()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{
			if (f.c0)
			{
				// density
				if (f.rho_BC == Neumann)
					f.rho_ghost = f.c0->rho_rk + 2 * f.grad_rho.dot(f.r0);

				// velocity-x
				if (f.U_BC[0] == Neumann)
					f.U_ghost.x() = f.c0->U_rk[0] + 2 * f.grad_U.col(0).dot(f.r0);

				// velocity-y
				if (f.U_BC[1] == Neumann)
					f.U_ghost.y() = f.c0->U_rk[1] + 2 * f.grad_U.col(1).dot(f.r0);

				// velocity-z
				if (f.U_BC[2] == Neumann)
					f.U_ghost.z() = f.c0->U_rk[2] + 2 * f.grad_U.col(2).dot(f.r0);

				// pressure
				if (f.p_BC == Neumann)
					f.p_ghost = f.c0->p_rk + 2 * f.grad_p.dot(f.r0);

				// temperature
				if (f.T_BC == Neumann)
					f.T_ghost = f.c0->T_rk + 2 * f.grad_T.dot(f.r0);
			}
			else if (f.c1)
			{
				// density
				if (f.rho_BC == Neumann)
					f.rho_ghost = f.c1->rho_rk + 2 * f.grad_rho.dot(f.r1);

				// velocity-x
				if (f.U_BC[0] == Neumann)
					f.U_ghost.x() = f.c1->U_rk[0] + 2 * f.grad_U.col(0).dot(f.r1);

				// velocity-y
				if (f.U_BC[1] == Neumann)
					f.U_ghost.y() = f.c1->U_rk[1] + 2 * f.grad_U.col(1).dot(f.r1);

				// velocity-z
				if (f.U_BC[2] == Neumann)
					f.U_ghost.z() = f.c1->U_rk[2] + 2 * f.grad_U.col(2).dot(f.r1);

				// pressure
				if (f.p_BC == Neumann)
					f.p_ghost = f.c1->p_rk + 2 * f.grad_p.dot(f.r1);

				// temperature
				if (f.T_BC == Neumann)
					f.T_ghost = f.c1->T_rk + 2 * f.grad_T.dot(f.r1);
			}
			else
				throw empty_connectivity(f.index);
		}
	}
}

/// Explicit Fractional-Step Method
void FSM(Scalar TimeStep)
{
	// Physical properties like dynamic viscosity
	calcCellProperty();

	// Boundary ghost values if any
	calcFaceGhostVariable();

	// Gradients at each cell's centroid
	calcCellGradient();

	// Averaged gradients at each face's centroid
	calcFaceAveragedGradient();

	// Gradients at each face's centroid 
	calcFaceGradient();

	/* Interpolate values on each face at current time level */
	calcFaceValue();

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

/// 4-step Runge-Kutta for time-marching
void RK4(Scalar TimeStep)
{
	/* Init */
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		c.rho_rk = c.rho;
		c.U_rk = c.U;
		c.p_rk = c.p;
		c.T_rk = c.T;
	}

	static const std::array<Scalar, 4> alpha = { 1.0 / 4, 1.0 / 3, 1.0 / 2, 1.0 };

	/* Step 1-4 */
	for (size_t m = 0; m < 4; ++m)
	{
		const Scalar cur_dt = alpha[m] * TimeStep;

		// Update pressure-velocity coupling, 
		// and compute residuals.
		FSM(cur_dt);

		// Update scalars
		for (size_t i = 1; i <= NumOfCell; ++i)
		{
			auto &c = cell(i);
			c.rho_rk = c.rho + cur_dt * c.continuity_res;
			c.T_rk = c.T + cur_dt * c.energy_res;
		}
	}

	/* Update */
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		c.rho = c.rho_rk;
		c.U = c.U_rk;
		c.p = c.p_rk;
		c.T = c.T_rk;
	}
}

bool diagnose()
{
	bool ret = false;

	return ret;
}

Scalar calcTimeStep()
{
	Scalar ret = 0.0;

	return ret;
}

void solve(std::ostream &fout = std::cout)
{
	static const size_t OUTPUT_GAP = 100;

	bool done = false;
	writeTECPLOT();
	while (!done)
	{
		fout << "Iter" << ++iter << ":" << std::endl;
		dt = calcTimeStep();
		fout << "\tt=" << t << "s, dt=" << dt << "s" << std::endl;
		RK4(dt);
		t += dt;
		done = diagnose();
		if (done || !(iter % OUTPUT_GAP))
			writeTECPLOT();
	}
	fout << "Finished!" << std::endl;
}

void init()
{
	readMSH();
	IC();
	BC();
	calcLeastSquareCoef();
}

int main(int argc, char *argv[])
{
	init();
	solve();
	return 0;
}
