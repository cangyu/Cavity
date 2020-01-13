#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "xf.h"
#include "Eigen/Dense"

/************************************************ Global types and variables *****************************************/
/* Math types */
typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 3, 3> Tensor;

/* BC types */
enum BC_Category { Dirichlet, Neumann, Robin };

/* 1-based array */
template<typename T>
class NaturalArray : public std::vector<T>
{
private:
	struct index_is_zero : public std::invalid_argument
	{
		index_is_zero() : std::invalid_argument("0 is invalid when using 1-based index.") {}
	};

public:
	NaturalArray() : std::vector<T>() {}
	explicit NaturalArray(size_t n) : std::vector<T>(n) {}
	NaturalArray(size_t n, const T &val) : std::vector<T>(n, val) {}
	~NaturalArray() = default;

	/* 1-based indexing */
	T &operator()(long long i)
	{
		if (i >= 1)
			return std::vector<T>::at(i - 1);
		else if (i <= -1)
			return std::vector<T>::at(std::vector<T>::size() + i);
		else
			throw index_is_zero();
	}
	const T &operator()(long long i) const
	{
		if (i >= 1)
			return std::vector<T>::at(i - 1);
		else if (i <= -1)
			return std::vector<T>::at(std::vector<T>::size() + i);
		else
			throw index_is_zero();
	}
};

/* Geom elements */
struct Cell;
struct Point
{
	// 1-based global index
	int index;

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
	int index;

	// 3D location of face centroid
	Vector center;

	// Area of the face element
	Scalar area;

	/* Connectivity */
	NaturalArray<Point*> vertex;
	Cell *c0 = nullptr, *c1 = nullptr;

	// Displacement vector
	Vector r0, r1;

	// Boundary flags
	bool atBdry = false;
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
	Vector rhoU;

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
	Tensor tau;

	// Averaged gradient of physical variables.
	// In general, they function as prediction of true gradients.
	Vector grad_rho_av;
	Tensor grad_U_av;
	Vector grad_p_av;
	Vector grad_T_av;

	// Physical properties
	Scalar mu;
};
struct Cell
{
	// 1-based global index
	int index;

	// 3D location of cell centroid
	Vector center;

	// Volume of the cell element
	Scalar volume;

	// Surface vector
	NaturalArray<Vector> S;

	// Connectivity
	NaturalArray<Point*> vertex;
	NaturalArray<Face*> surface;
	NaturalArray<Cell*> adjCell;

	// Least-squares method variables
	Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_rho;
	std::array<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3> J_INV_U;
	Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_p;
	Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_T;

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

	/* Fractional-Step Method temporary variables */
	// Momentum equation
	Vector pressure_flux;
	Vector convection_flux;
	Vector viscous_flux;
	Vector rhoU_star;
};
struct Patch
{
	std::string name;
	int BC;
	NaturalArray<Face*> surface;
};

/* Iteration timing and counting */
int iter = 0;
const int MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

/* Constant field variables */
const Scalar rho0 = 1.225; //kg/m^3
const Vector U0 = { 1.0, 0.0, 0.0 }; // m/s
const Scalar P0 = 101325.0; // Pa
const Scalar T0 = 300.0; // K

/* Grid utilities */
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

NaturalArray<Point> pnt; // Node objects
NaturalArray<Face> face; // Face objects
NaturalArray<Cell> cell; // Cell objects
NaturalArray<Patch> patch; // Group of boundary faces

/********************************************* Errors and Exceptions *************************************************/
struct failed_to_open_file : public std::runtime_error
{
	failed_to_open_file(const std::string &fn) : std::runtime_error("Failed to open target file: \"" + fn + "\".") {}
};

struct unsupported_boundary_condition : public std::invalid_argument
{
	unsupported_boundary_condition(BC_Category x) : std::invalid_argument(std::to_string((int)x) + ".") {}
};

struct empty_connectivity : public std::runtime_error
{
	empty_connectivity(int idx) : std::runtime_error("Both c0 and c1 are NULL on face " + std::to_string(idx) + ".") {}
};

/*************************************************** File I/O ********************************************************/
/// Load mesh.
void readMSH(const std::string &MESH_PATH)
{
	using namespace GridTool;

	// Load ANSYS Fluent mesh using external independent package.
	const XF::MESH mesh(MESH_PATH);

	// Update counting of geom elements.
	NumOfPnt = mesh.numOfNode();
	NumOfFace = mesh.numOfFace();
	NumOfCell = mesh.numOfCell();

	// Allocate memory for geom information and related physical variables.
	pnt.resize(NumOfPnt);
	face.resize(NumOfFace);
	cell.resize(NumOfCell);

	// Update point information.
	for (int i = 1; i <= NumOfPnt; ++i)
	{
		const auto &n_src = mesh.node(i);
		auto &n_dst = pnt(i);

		// Assign node index.
		n_dst.index = i;

		// Node location.
		const auto &c_src = n_src.coordinate;
		n_dst.coordinate = { c_src.x(), c_src.y(), c_src.z() };
	}

	// Update face information.
	for (int i = 1; i <= NumOfFace; ++i)
	{
		const auto &f_src = mesh.face(i);
		auto &f_dst = face(i);

		// Assign face index.
		f_dst.index = i;
		f_dst.atBdry = f_src.atBdry;

		// Face center location.
		const auto &c_src = f_src.center;
		f_dst.center = { c_src.x(), c_src.y(), c_src.z() };

		// Face area.
		f_dst.area = f_src.area;

		// Face included nodes.
		const auto N1 = f_src.includedNode.size();
		f_dst.vertex.resize(N1);
		for (int j = 1; j <= N1; ++j)
			f_dst.vertex(j) = &pnt(f_src.includedNode(j));

		// Face adjacent 2 cells.
		f_dst.c0 = f_src.leftCell ? &cell(f_src.leftCell) : nullptr;
		f_dst.c1 = f_src.rightCell ? &cell(f_src.rightCell) : nullptr;
	}

	// Update cell information.
	for (int i = 1; i <= NumOfCell; ++i)
	{
		const auto &c_src = mesh.cell(i);
		auto &c_dst = cell(i);

		// Assign cell index.
		c_dst.index = i;

		// Cell center location.
		const auto &centroid_src = c_src.center;
		c_dst.center = { centroid_src.x(), centroid_src.y(), centroid_src.z() };

		// Cell included nodes.
		const auto N1 = c_src.includedNode.size();
		c_dst.vertex.resize(N1);
		for (int j = 1; j <= N1; ++j)
			c_dst.vertex(j) = &pnt(c_src.includedNode(j));

		// Cell included faces.
		const auto N2 = c_src.includedFace.size();
		c_dst.surface.resize(N2);
		for (int j = 1; j <= N2; ++j)
			c_dst.surface(j) = &face(c_src.includedFace(j));

		// Cell adjacent cells.
		c_dst.adjCell.resize(N2);
		for (int j = 1; j <= N2; ++j)
		{
			const auto adjIdx = c_src.adjacentCell(j);
			c_dst.adjCell(j) = adjIdx ? &cell(adjIdx) : nullptr;
		}
	}

	// Cell center to face center vectors 
	for (int i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);

		if (f_dst.c0)
			f_dst.r0 = f_dst.center - f_dst.c0->center;
		else
			f_dst.r0 = { 0, 0, 0 };

		if (f_dst.c1)
			f_dst.r1 = f_dst.center - f_dst.c1->center;
		else
			f_dst.r1 = { 0, 0, 0 };
	}

	// Count valid patches.
	size_t NumOfPatch = 0;
	for (int i = 1; i <= mesh.numOfZone(); ++i)
	{
		const auto &z_src = mesh.zone(i);

		const auto curZoneIdx = z_src.ID;
		auto curZonePtr = z_src.obj;
		auto curFace = dynamic_cast<XF::FACE*>(curZonePtr);
		if (curFace)
		{
			if (curFace->identity() != XF::SECTION::FACE || curFace->zone() != curZoneIdx)
				throw std::runtime_error("Inconsistency detected.");

			if (XF::BC::str2idx(z_src.type) != XF::BC::INTERIOR)
				++NumOfPatch;
		}
	}

	// Update boundary patch information.
	patch.resize(NumOfPatch);
	for (int i = 1, cnt = 0; i <= mesh.numOfZone(); ++i)
	{
		const auto &curZone = mesh.zone(i);
		auto curFace = dynamic_cast<XF::FACE*>(curZone.obj);
		if (curFace == nullptr || XF::BC::str2idx(curZone.type) == XF::BC::INTERIOR)
			continue;

		auto &p_dst = patch[cnt];
		p_dst.name = curZone.name;
		p_dst.BC = XF::BC::str2idx(curZone.type);
		p_dst.surface.resize(curFace->num());
		const auto loc_first = curFace->first_index();
		const auto loc_last = curFace->last_index();
		for (auto j = loc_first; j <= loc_last; ++j)
			p_dst.surface.at(j - loc_first) = &face(j);

		cnt += 1;
	}
}

/// Output computation results.
/// Nodal values are exported, including boundary variables.
void writeTECPLOT_Nodal(const std::string &fn, const std::string &title)
{
	static const size_t RECORD_PER_LINE = 10;

	std::ofstream fout(fn);
	if (fout.fail())
		throw failed_to_open_file(fn);

	fout << R"(TITLE=")" << title << "\"" << std::endl;
	fout << "FILETYPE=FULL" << std::endl;
	fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;

	fout << R"(ZONE T="Nodal")" << std::endl;
	fout << "NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-9]=NODAL)" << std::endl;

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
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).rho;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-X
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).U.x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Y
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).U.y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Z
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).U.z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Pressure
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).p;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Temperature
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		fout << '\t' << pnt(i).T;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfPnt % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Connectivity Information
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		const auto v = cell(i).vertex;
		for (const auto &e : v)
			fout << '\t' << e->index;
		fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	fout.close();
}

/// Output computation results.
/// Cell-Centered values are exported, boundary variables are NOT included.
/// Only for continuation purpose.
void writeTECPLOT_CellCentered(const std::string &fn, const std::string &title)
{
	static const size_t RECORD_PER_LINE = 10;

	std::ofstream fout(fn);
	if (fout.fail())
		throw failed_to_open_file(fn);

	fout << R"(TITLE=")" << title << "\"" << std::endl;
	fout << "FILETYPE=FULL" << std::endl;
	fout << R"(VARIABLES="X", "Y", "Z", "rho", "U", "V", "W", "P", "T")" << std::endl;

	fout << R"(ZONE T="Cell-Centroid")" << std::endl;
	fout << "NODES=" << NumOfPnt << ", ELEMENTS=" << NumOfCell << ", ZONETYPE=FEBRICK, DATAPACKING=BLOCK, VARLOCATION=([1-3]=NODAL, [4-9]=CELLCENTERED)" << std::endl;

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
		fout << '\t' << cell(i).rho;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U.x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U.y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U.z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).p;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).T;
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
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	fout.close();
}

/// Extract NODAL solution variables only.
/// Should be consistent with existing mesh!!!
void readTECPLOT_Nodal(const std::string &fn)
{
	std::ifstream fin(fn);
	if (fin.fail())
		throw failed_to_open_file(fn);

	/* Skip header */
	for (int i = 0; i < 5; ++i)
	{
		std::string tmp;
		std::getline(fin, tmp);
	}

	/* Skip coordinates */
	for (int k = 0; k < 3; ++k)
	{
		Scalar var;
		for (size_t i = 1; i <= NumOfPnt; ++i)
			fin >> var;
	}

	/* Load data */
	// Density
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).rho;

	// Velocity-X
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).U.x();

	// Velocity-Y
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).U.y();

	// Velocity-Z
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).U.z();

	// Pressure
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).p;

	// Temperature
	for (size_t i = 1; i <= NumOfPnt; ++i)
		fin >> pnt(i).T;

	/* Finalize */
	fin.close();
}

/// Extract CELL-CENTERED solution variables only.
/// Should be consistent with existing mesh!!!
/// Only for continuation purpose.
void readTECPLOT_CellCentered(const std::string &fn)
{
	std::ifstream fin(fn);
	if (fin.fail())
		throw failed_to_open_file(fn);

	/* Skip header */
	for (int i = 0; i < 5; ++i)
	{
		std::string tmp;
		std::getline(fin, tmp);
	}

	/* Skip coordinates */
	for (int k = 0; k < 3; ++k)
	{
		Scalar var;
		for (size_t i = 1; i <= NumOfPnt; ++i)
			fin >> var;
	}

	/* Load data */
	// Density
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).rho;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U.x();

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U.y();

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U.z();

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).p;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).T;

	/* Finalize */
	fin.close();
}

/*************************************************** Functions *******************************************************/
/// QR decomposition matrix of each cell.
/// For boundary cells, ghost cells are used when the B.C. of related faces are set to Neumann type.
void calcLeastSquareCoef()
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> J_rho;
	std::array<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>, 3> J_U;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> J_p;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 3> J_T;

	for (auto &c : cell)
	{
		const size_t nF = c.surface.size();

		J_rho.resize(nF, Eigen::NoChange);
		J_U[0].resize(nF, Eigen::NoChange);
		J_U[1].resize(nF, Eigen::NoChange);
		J_U[2].resize(nF, Eigen::NoChange);
		J_p.resize(nF, Eigen::NoChange);
		J_T.resize(nF, Eigen::NoChange);

		for (size_t j = 0; j < nF; ++j)
		{
			auto curFace = c.surface.at(j);
			if (curFace->atBdry)
			{
				const auto dx = curFace->center.x() - c.center.x();
				const auto dy = curFace->center.y() - c.center.y();
				const auto dz = curFace->center.z() - c.center.z();

				const auto dx2 = 2 * dx;
				const auto dy2 = 2 * dy;
				const auto dz2 = 2 * dz;

				// Density
				switch (curFace->rho_BC)
				{
				case Dirichlet:
					J_rho.row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_rho.row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// Velocity-X
				switch (curFace->U_BC[0])
				{
				case Dirichlet:
					J_U[0].row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_U[0].row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// Velocity-Y
				switch (curFace->U_BC[1])
				{
				case Dirichlet:
					J_U[1].row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_U[1].row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// Velocity-Z
				switch (curFace->U_BC[2])
				{
				case Dirichlet:
					J_U[2].row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_U[2].row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// Pressure
				switch (curFace->p_BC)
				{
				case Dirichlet:
					J_p.row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_p.row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// Temperature
				switch (curFace->T_BC)
				{
				case Dirichlet:
					J_T.row(j) << dx, dy, dz;
					break;
				case Neumann:
					J_T.row(j) << dx2, dy2, dz2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}
			}
			else
			{
				auto curAdjCell = c.adjCell.at(j);

				const auto dx = curAdjCell->center.x() - c.center.x();
				const auto dy = curAdjCell->center.y() - c.center.y();
				const auto dz = curAdjCell->center.z() - c.center.z();

				// Density
				J_rho.row(j) << dx, dy, dz;

				// Velocity-X
				J_U[0].row(j) << dx, dy, dz;

				// Velocity-Y
				J_U[1].row(j) << dx, dy, dz;

				// Velocity-Z
				J_U[2].row(j) << dx, dy, dz;

				// Pressure
				J_p.row(j) << dx, dy, dz;

				// Temperature
				J_T.row(j) << dx, dy, dz;
			}
		}


		auto Q = J_rho.householderQr().matrixQR();

		/*
		c.rho_QR = J_rho.householderQr();
		c.U_QR[0] = J_U.householderQr();
		c.p_QR = J_p.householderQr();
		c.T_QR = J_T.householderQr();
		*/
	}
}

/// Dynamic viscosity of ideal gas.
Scalar Sutherland(Scalar T)
{
	return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
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

/// Initial conditions.
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

/// Boundary conditions.
void BC()
{
	for (auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
			{
				f->U = U0;
				for (auto v : f->vertex)
					v->U = U0;
			}
		}
		else
		{
			for (auto f : e.surface)
				f->U = { 0.0, 0.0, 0.0 };
		}
	}
}

void solve(std::ostream &fout = std::cout)
{
	static const size_t OUTPUT_GAP = 100;

	bool done = false;


	while (!done)
	{
		fout << "Iter" << ++iter << ":" << std::endl;
		dt = calcTimeStep();
		fout << "\tt=" << t << "s, dt=" << dt << "s" << std::endl;
		//RK4(dt);
		t += dt;
		done = diagnose();
		if (done || !(iter % OUTPUT_GAP))
			writeTECPLOT_Nodal("flow" + std::to_string(iter) + ".dat", "3D Cavity");
	}
	fout << "Finished!" << std::endl;
}

/// Initialize the compuation environment.
void init()
{
	readMSH("grid0.msh");
	calcLeastSquareCoef();


	IC();
	BC();
	writeTECPLOT_Nodal("flow0.dat", "3D Cavity");
	exit(-1);

}

int main(int argc, char *argv[])
{
	init();
	solve();

	return 0;
}
