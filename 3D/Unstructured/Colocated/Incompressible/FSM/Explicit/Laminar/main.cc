#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <map>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "xf.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

/***************************************************** Marcos ********************************************************/

#define ZERO_INDEX 0
#define ZERO_SCALAR 0.0
#define ZERO_VECTOR {0.0, 0.0, 0.0}

/****************************************************** Types ********************************************************/

/* Math types */
typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 3, 3> Tensor;

/* BC types */
enum BC_CATEGORY { Dirichlet = 0, Neumann, Robin };
static const std::array<std::string, 3> BC_CATEGORY_STR = { "Dirichlet", "Neumann", "Robin" };

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
	T &operator()(int i)
	{
		if (i >= 1)
			return std::vector<T>::at(i - 1);
		else if (i <= -1)
			return std::vector<T>::at(std::vector<T>::size() + i);
		else
			throw index_is_zero();
	}
	const T &operator()(int i) const
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
struct Patch;
struct Point
{
	// 1-based global index
	int index = ZERO_INDEX;

	// 3D Location
	Vector coordinate = ZERO_VECTOR;

	// Physical variables
	Scalar rho = ZERO_SCALAR;
	Vector U = ZERO_VECTOR;
	Scalar p = ZERO_SCALAR;
	Scalar T = ZERO_SCALAR;
};
struct Face
{
	// 1-based global index
	int index = ZERO_INDEX;

	// 3D location of face centroid
	Vector center = ZERO_VECTOR;

	// Area of the face element
	Scalar area = ZERO_SCALAR;
	Vector n01 = ZERO_VECTOR, n10 = ZERO_VECTOR;

	// Connectivity
	NaturalArray<Point*> vertex;
	Cell *c0 = nullptr, *c1 = nullptr;

	// Displacement vector
	Vector r0 = ZERO_VECTOR, r1 = ZERO_VECTOR;

	// Boundary flags
	bool atBdry = false;
	Patch *parent = nullptr;
	BC_CATEGORY rho_BC = Dirichlet;
	std::array<BC_CATEGORY, 3> U_BC = { Dirichlet, Dirichlet, Dirichlet };
	BC_CATEGORY p_BC = Neumann;
	BC_CATEGORY T_BC = Neumann;

	// Ghost variables if needed
	Scalar rho_ghost;
	Vector U_ghost = ZERO_VECTOR;
	Scalar p_ghost;
	Scalar T_ghost;

	// Physical properties
	Scalar mu = ZERO_SCALAR;

	// Physical variables
	Scalar rho = ZERO_SCALAR;
	Vector U = ZERO_VECTOR;
	Scalar p = ZERO_SCALAR;
	Scalar T = ZERO_SCALAR;
	Vector rhoU;

	// Gradient of physical variables
	Vector grad_rho;
	Tensor grad_U;
	Vector grad_p;
	Vector grad_T;
	Tensor tau;

	/* Fractional-Step Method temporary variables */
	Vector rhoU_star;
};
struct Cell
{
	// 1-based global index
	int index = ZERO_INDEX;

	// 3D location of cell centroid
	Vector center = ZERO_VECTOR;

	// Volume of the cell element
	Scalar volume = ZERO_SCALAR;

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
	Scalar rho0 = ZERO_SCALAR;
	Vector U0 = ZERO_VECTOR;
	Scalar p0 = ZERO_SCALAR;
	Scalar T0 = ZERO_SCALAR;
	Vector rhoU0 = ZERO_VECTOR;

	/* Runge-Kutta temporary variables */
	// Physical properties
	Scalar mu = ZERO_SCALAR;

	// Physical variables
	Scalar rho = ZERO_SCALAR;
	Vector U = ZERO_VECTOR;
	Scalar p = ZERO_SCALAR;
	Scalar T = ZERO_SCALAR;

	// Gradients
	Vector grad_rho;
	Tensor grad_U;
	Vector grad_p;
	Vector grad_T;

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

/********************************************* Errors and Exceptions *************************************************/

struct failed_to_open_file : public std::runtime_error
{
	explicit failed_to_open_file(const std::string &fn) : std::runtime_error("Failed to open target file: \"" + fn + "\".") {}
};

struct unsupported_boundary_condition : public std::invalid_argument
{
	explicit unsupported_boundary_condition(BC_CATEGORY x) : std::invalid_argument("\"" + BC_CATEGORY_STR[x] + "\" condition is not supported.") {}
};

struct empty_connectivity : public std::runtime_error
{
	explicit empty_connectivity(int idx) : std::runtime_error("Both c0 and c1 are NULL on face " + std::to_string(idx) + ".") {}
};

struct unexpected_patch : public std::runtime_error
{
	unexpected_patch(const std::string &name) : std::runtime_error("Patch \"" + name + "\" is not expected to be a boundary patch.") {}
};

/*************************************************** Global Variables ************************************************/

/* Iteration timing and counting */
const int MAX_ITER = 2000;
const Scalar MAX_TIME = 100.0; // s

/* Grid utilities */
size_t NumOfPnt = 0;
size_t NumOfFace = 0;
size_t NumOfCell = 0;

NaturalArray<Point> pnt; // Node objects
NaturalArray<Face> face; // Face objects
NaturalArray<Cell> cell; // Cell objects
NaturalArray<Patch> patch; // Group of boundary faces

/*************************************************** File I/O ********************************************************/

/**
 * Load mesh.
 * @param MESH_PATH
 */
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

		// Face unit normal.
		f_dst.n10 = { f_src.n_RL.x(), f_src.n_RL.y(), f_src.n_RL.z() };
		f_dst.n01 = { f_src.n_LR.x(), f_src.n_LR.y(), f_src.n_LR.z() };

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
		{
			p_dst.surface.at(j - loc_first) = &face(j);
			face(j).parent = &p_dst;
		}
		cnt += 1;
	}
}

/**
 * Output computation results.
 * Nodal values are exported, including boundary variables.
 * @param fn
 * @param title
 */
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

/**
 * Output computation results.
 * Cell-Centered values are exported, boundary variables are NOT included.
 * Only for continuation purpose.
 * @param fn
 * @param title
 */
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
		fout << '\t' << cell(i).rho0;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U0.x();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U0.y();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).U0.z();
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).p0;
		if (i % RECORD_PER_LINE == 0)
			fout << std::endl;
	}
	if (NumOfCell % RECORD_PER_LINE != 0)
		fout << std::endl;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		fout << '\t' << cell(i).T0;
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

/**
 * Extract NODAL solution variables only.
 * Should be consistent with existing mesh!!!
 * @param fn
 */
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

/**
 * Extract CELL-CENTERED solution variables only.
 * Should be consistent with existing mesh!!!
 * Only for continuation purpose.
 * @param fn
 */
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
		fin >> cell(i).rho0;

	// Velocity-X
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U0.x();

	// Velocity-Y
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U0.y();

	// Velocity-Z
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).U0.z();

	// Pressure
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).p0;

	// Temperature
	for (size_t i = 1; i <= NumOfCell; ++i)
		fin >> cell(i).T0;

	/* Finalize */
	fin.close();
}

/*********************************************** Least-Squares Method ************************************************/

/**
 * Convert Eigen's intrinsic QR decomposition matrix into R^-1 * Q^T
 * @param J
 * @param J_INV
 */
static void extractQRMatrix(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &J, Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &J_INV)
{
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mat;

	auto QR = J.householderQr();
	const Mat Q = QR.householderQ();
	const Mat R = QR.matrixQR().triangularView<Eigen::Upper>();

	const Mat Q0 = Q.block(0, 0, J.rows(), 3);
	const Mat R0 = R.block<3, 3>(0, 0);

	J_INV = R0.inverse() * Q0.transpose();
}

/**
 * QR decomposition matrix of each cell.
 * Ghost cells are used when the B.C. of boundary faces are set to Neumann type.
 */
void calcLeastSquareCoefficients()
{
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> Mat;

	Mat J_rho;
	std::array<Mat, 3> J_U;
	Mat J_p;
	Mat J_T;

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

		// Density
		extractQRMatrix(J_rho, c.J_INV_rho);

		// Velocity-X
		extractQRMatrix(J_U[0], c.J_INV_U[0]);

		// Velocity-Y
		extractQRMatrix(J_U[1], c.J_INV_U[1]);

		// Velocity-Z
		extractQRMatrix(J_U[2], c.J_INV_U[2]);

		// Pressure
		extractQRMatrix(J_p, c.J_INV_p);

		// Temperature
		extractQRMatrix(J_T, c.J_INV_T);
	}
}

/************************************************ Physical properties ************************************************/

/**
 * Dynamic viscosity of ideal gas.
 * @param T
 * @return
 */
Scalar Sutherland(Scalar T)
{
	return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
}

/***************************************************** I.C. & B.C. ***************************************************/

void BC_TABLE()
{
	for (const auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Dirichlet;
			}
		}
		else if (e.name == "DOWN")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Dirichlet;
			}
		}
		else if (e.name == "LEFT")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Neumann;
			}
		}
		else if (e.name == "RIGHT")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Neumann;
			}
		}
		else if (e.name == "FRONT")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Neumann;
			}
		}
		else if (e.name == "BACK")
		{
			for (auto f : e.surface)
			{
				f->rho_BC = Dirichlet;
				f->U_BC = { Dirichlet, Dirichlet, Dirichlet };
				f->p_BC = Neumann;
				f->T_BC = Neumann;
			}
		}
		else
			throw unexpected_patch(e.name);
	}
}

/**
 * Initial conditions on all nodes, faces and cells.
 * Boundary elements are also set same to interior, will be corrected in BC routine.
 */
void IC()
{
	const Scalar rho0 = 1.225; //kg/m^3	
	const Scalar P0 = 101325.0; // Pa
	const Scalar T0 = 300.0; // K

	// Node
	for (size_t i = 1; i <= NumOfPnt; ++i)
	{
		auto &n_dst = pnt(i);
		n_dst.rho = rho0;
		n_dst.U = ZERO_VECTOR;
		n_dst.p = P0;
		n_dst.T = T0;
	}

	// Face
	for (size_t i = 1; i <= NumOfFace; ++i)
	{
		auto &f_dst = face(i);
		f_dst.rho = rho0;
		f_dst.U = ZERO_VECTOR;
		f_dst.p = P0;
		f_dst.T = T0;
	}

	// Cell
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c_dst = cell(i);
		c_dst.rho0 = rho0;
		c_dst.U0 = ZERO_VECTOR;
		c_dst.p0 = P0;
		c_dst.T0 = T0;
	}
}

/**
 * Boundary conditions on all related nodes and faces for all variables.
 */
void BC()
{
	const Vector U0 = { 1.0, 0.0, 0.0 }; // m/s
	const Scalar T_L = 300.0, T_H = 1500.0; // K

	for (const auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
			{
				f->U = U0;
				f->T = T_H;
			}
		}
		else if (e.name == "DOWN")
		{
			for (auto f : e.surface)
			{
				f->U = ZERO_VECTOR;
				f->T = T_L;
			}
		}
		else if (e.name == "LEFT")
		{
			for (auto f : e.surface)
			{
				f->U = ZERO_VECTOR;
				f->grad_T = ZERO_VECTOR;
			}
		}
		else if (e.name == "RIGTH")
		{
			for (auto f : e.surface)
			{
				f->U = ZERO_VECTOR;
				f->grad_T = ZERO_VECTOR;
			}
		}
		else if (e.name == "FRONT")
		{
			for (auto f : e.surface)
			{
				f->U = ZERO_VECTOR;
				f->grad_T = ZERO_VECTOR;
			}
		}
		else if (e.name == "BACK")
		{
			for (auto f : e.surface)
			{
				f->U = ZERO_VECTOR;
				f->grad_T = ZERO_VECTOR;
			}
		}
		else
			throw unexpected_patch(e.name);
	}
}

void updateNodalValue()
{
	std::vector<bool> visited(NumOfPnt + 1, false);

	/* Velocity */
	const Vector U0 = { 1.0, 0.0, 0.0 }; // m/s
	for (auto &e : patch)
	{
		if (e.name == "UP")
		{
			for (auto f : e.surface)
				for (auto v : f->vertex)
					if (!visited[v->index])
					{
						v->U = U0;
						visited[v->index] = true;
					}
		}
	}
	for (auto & e : patch)
	{
		if (e.name != "UP")
		{
			for (auto f : e.surface)
				for (auto v : f->vertex)
					if (!visited[v->index])
					{
						v->U = { 0.0, 0.0, 0.0 };
						visited[v->index] = true;
					}
		}
	}
}

/*********************************************** Spatial Discretization **********************************************/

void calcCellProperty()
{
	for (auto &c : cell)
	{
		// Dynamic viscosity
		c.mu = Sutherland(c.T);
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
				switch (f.rho_BC)
				{
				case Neumann:
					f.rho_ghost = f.c0->rho + 2 * f.grad_rho.dot(f.r0);
					break;
				case Dirichlet:
					f.rho_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Neumann:
					f.U_ghost.x() = f.c0->U[0] + 2 * f.grad_U.col(0).dot(f.r0);
					break;
				case Dirichlet:
					f.U_ghost.x() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Neumann:
					f.U_ghost.y() = f.c0->U[1] + 2 * f.grad_U.col(1).dot(f.r0);
					break;
				case Dirichlet:
					f.U_ghost.y() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Neumann:
					f.U_ghost.z() = f.c0->U[2] + 2 * f.grad_U.col(2).dot(f.r0);
					break;
				case Dirichlet:
					f.U_ghost.z() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Neumann:
					f.p_ghost = f.c0->p + 2 * f.grad_p.dot(f.r0);
					break;
				case Dirichlet:
					f.p_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Neumann:
					f.T_ghost = f.c0->T + 2 * f.grad_T.dot(f.r0);
					break;
				case Dirichlet:
					f.p_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}
			}
			else if (f.c1)
			{
				// density
				switch (f.rho_BC)
				{
				case Neumann:
					f.rho_ghost = f.c1->rho + 2 * f.grad_rho.dot(f.r1);
					break;
				case Dirichlet:
					f.rho_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Neumann:
					f.U_ghost.x() = f.c1->U[0] + 2 * f.grad_U.col(0).dot(f.r1);
					break;
				case Dirichlet:
					f.U_ghost.x() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Neumann:
					f.U_ghost.y() = f.c1->U[1] + 2 * f.grad_U.col(1).dot(f.r1);
					break;
				case Dirichlet:
					f.U_ghost.y() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Neumann:
					f.U_ghost.z() = f.c1->U[2] + 2 * f.grad_U.col(2).dot(f.r1);
					break;
				case Dirichlet:
					f.U_ghost.z() = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Neumann:
					f.p_ghost = f.c1->p + 2 * f.grad_p.dot(f.r1);
					break;
				case Dirichlet:
					f.p_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Neumann:
					f.T_ghost = f.c1->T + 2 * f.grad_T.dot(f.r1);
					break;
				case Dirichlet:
					f.T_ghost = 0.0;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}
			}
			else
				throw empty_connectivity(f.index);
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
					dphi(i) = curFace->rho - c.rho;
				else
					dphi(i) = curFace->rho_ghost - c.rho;
			}
			else
				dphi(i) = curAdjCell->rho - c.rho;
		}
		c.grad_rho = c.J_INV_rho * dphi;

		/* Gradient of x-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[0] == Dirichlet)
					dphi(i) = curFace->U.x() - c.U.x();
				else
					dphi(i) = curFace->U_ghost.x() - c.U.x();
			}
			else
				dphi(i) = curAdjCell->U.x() - c.U.x();
		}
		c.grad_U.col(0) = c.J_INV_U[0] * dphi;

		/* Gradient of y-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[1] == Dirichlet)
					dphi(i) = curFace->U.y() - c.U.y();
				else
					dphi(i) = curFace->U_ghost.y() - c.U.y();
			}
			else
				dphi(i) = curAdjCell->U.y() - c.U.y();
		}
		c.grad_U.col(1) = c.J_INV_U[1] * dphi;

		/* Gradient of z-dim velocity */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->U_BC[2] == Dirichlet)
					dphi(i) = curFace->U.z() - c.U.z();
				else
					dphi(i) = curFace->U_ghost.z() - c.U.z();
			}
			else
				dphi(i) = curAdjCell->U.z() - c.U.z();
		}
		c.grad_U.col(2) = c.J_INV_U[2] * dphi;

		/* Gradient of pressure */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->p_BC == Dirichlet)
					dphi(i) = curFace->p - c.p;
				else
					dphi(i) = curFace->p_ghost - c.p;
			}
			else
				dphi(i) = curAdjCell->p - c.p;
		}
		c.grad_p = c.J_INV_p * dphi;

		/* Gradient of temperature */
		for (size_t i = 0; i < nF; ++i)
		{
			auto curFace = c.surface.at(i);
			auto curAdjCell = c.adjCell.at(i);

			if (curFace->atBdry)
			{
				if (curFace->T_BC == Dirichlet)
					dphi(i) = curFace->T - c.T;
				else
					dphi(i) = curFace->T_ghost - c.T;
			}
			else
				dphi(i) = curAdjCell->T - c.T;
		}
		c.grad_T = c.J_INV_T * dphi;
	}
}

inline Vector interpGradientToFace(const Vector &predicted_grad_phi_f, Scalar phi_C, Scalar phi_F, const Vector &e_CF, Scalar d_CF)
{
	return predicted_grad_phi_f + ((phi_F - phi_C) / d_CF - predicted_grad_phi_f.dot(e_CF))*e_CF;
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
				switch (f.rho_BC)
				{
				case Dirichlet:
					f.grad_rho = interpGradientToFace(f.c0->grad_rho, f.c0->rho, f.rho, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Dirichlet:
					f.grad_U.col(0) = interpGradientToFace(f.c0->grad_U.col(0), f.c0->U.x(), f.U.x(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Dirichlet:
					f.grad_U.col(1) = interpGradientToFace(f.c0->grad_U.col(1), f.c0->U.y(), f.U.y(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Dirichlet:
					f.grad_U.col(2) = interpGradientToFace(f.c0->grad_U.col(2), f.c0->U.z(), f.U.z(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Dirichlet:
					f.grad_p = interpGradientToFace(f.c0->grad_p, f.c0->p, f.p, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Dirichlet:
					f.grad_T = interpGradientToFace(f.c0->grad_T, f.c0->T, f.T, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
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
				switch (f.rho_BC)
				{
				case Dirichlet:
					f.grad_rho = interpGradientToFace(f.c1->grad_rho, f.c1->rho, f.rho, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Dirichlet:
					f.grad_U.col(0) = interpGradientToFace(f.c1->grad_U.col(0), f.c1->U.x(), f.U.x(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Dirichlet:
					f.grad_U.col(1) = interpGradientToFace(f.c1->grad_U.col(1), f.c1->U.y(), f.U.y(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Dirichlet:
					f.grad_U.col(2) = interpGradientToFace(f.c1->grad_U.col(2), f.c1->U.z(), f.U.z(), e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Dirichlet:
					f.grad_p = interpGradientToFace(f.c1->grad_p, f.c1->p, f.p, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Dirichlet:
					f.grad_T = interpGradientToFace(f.c1->grad_T, f.c1->T, f.T, e_CF, d_CF);
					break;
				case Neumann:
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
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

			const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

			// density
			const Vector predicted_grad_rho = ksi * f.c0->grad_rho + (1 - ksi) * f.c1->grad_rho;
			const Scalar rho_C = f.c0->rho;
			const Scalar rho_F = f.c1->rho;
			f.grad_rho = interpGradientToFace(predicted_grad_rho, rho_C, rho_F, e_CF, d_CF);

			// velocity-x
			const Vector predicted_grad_u = ksi * f.c0->grad_U.col(0) + (1 - ksi) * f.c1->grad_U.col(0);
			const Scalar u_C = f.c0->U.x();
			const Scalar u_F = f.c1->U.x();
			f.grad_U.col(0) = interpGradientToFace(predicted_grad_u, u_C, u_F, e_CF, d_CF);

			// velocity-y
			const Vector predicted_grad_v = ksi * f.c0->grad_U.col(1) + (1 - ksi) * f.c1->grad_U.col(1);
			const Scalar v_C = f.c0->U.y();
			const Scalar v_F = f.c1->U.y();
			f.grad_U.col(1) = interpGradientToFace(predicted_grad_v, v_C, v_F, e_CF, d_CF);

			// velocity-z
			const Vector predicted_grad_w = ksi * f.c0->grad_U.col(2) + (1 - ksi) * f.c1->grad_U.col(2);
			const Scalar w_C = f.c0->U.z();
			const Scalar w_F = f.c1->U.z();
			f.grad_U.col(2) = interpGradientToFace(predicted_grad_w, w_C, w_F, e_CF, d_CF);

			// pressure
			const Vector predicted_grad_p = ksi * f.c0->grad_p + (1 - ksi) * f.c1->grad_p;
			const Scalar p_C = f.c0->p;
			const Scalar p_F = f.c1->p;
			f.grad_p = interpGradientToFace(predicted_grad_p, p_C, p_F, e_CF, d_CF);

			// temperature
			const Vector predicted_grad_T = ksi * f.c0->grad_T + (1 - ksi) * f.c1->grad_T;
			const Scalar T_C = f.c0->T;
			const Scalar T_F = f.c1->T;
			f.grad_T = interpGradientToFace(predicted_grad_T, T_C, T_F, e_CF, d_CF);
		}
	}
}

void calcFaceValue()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{
			if (f.c0)
			{
				// density
				switch (f.rho_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.rho = f.c0->rho + (f.grad_rho + f.c0->grad_rho).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.x() = f.c0->U.x() + (f.grad_U.col(0) + f.c0->grad_U.col(0)).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.y() = f.c0->U.y() + (f.grad_U.col(1) + f.c0->grad_U.col(1)).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.z() = f.c0->U.z() + (f.grad_U.col(2) + f.c0->grad_U.col(2)).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.p = f.c0->p + (f.grad_p + f.c0->grad_p).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.T = f.c0->T + (f.grad_T + f.c0->grad_T).dot(f.r0) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}
			}
			else if (f.c1)
			{
				// density
				switch (f.rho_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.rho = f.c1->rho + (f.grad_rho + f.c1->grad_rho).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-x
				switch (f.U_BC[0])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.x() = f.c1->U.x() + (f.grad_U.col(0) + f.c1->grad_U.col(0)).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-y
				switch (f.U_BC[1])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.y() = f.c1->U.y() + (f.grad_U.col(1) + f.c1->grad_U.col(1)).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// velocity-z
				switch (f.U_BC[2])
				{
				case Dirichlet:
					break;
				case Neumann:
					f.U.z() = f.c1->U.z() + (f.grad_U.col(2) + f.c1->grad_U.col(2)).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// pressure
				switch (f.p_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.p = f.c1->p + (f.grad_p + f.c1->grad_p).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}

				// temperature
				switch (f.T_BC)
				{
				case Dirichlet:
					break;
				case Neumann:
					f.T = f.c1->T + (f.grad_T + f.c1->grad_T).dot(f.r1) / 2;
					break;
				case Robin:
					throw unsupported_boundary_condition(Robin);
				default:
					break;
				}
			}
			else
				throw empty_connectivity(f.index);
		}
		else
		{
			// weighting coefficient
			const Scalar ksi = f.r1.norm() / (f.r0.norm() + f.r1.norm());

			// pressure
			const Scalar p_0 = f.c0->p + f.c0->grad_p.dot(f.r0);
			const Scalar p_1 = f.c1->p + f.c1->grad_p.dot(f.r1);
			f.p = 0.5 * (p_0 + p_1);

			// temperature
			const Scalar T_0 = f.c0->T + f.c0->grad_T.dot(f.r0);
			const Scalar T_1 = f.c1->T + f.c1->grad_T.dot(f.r1);
			f.T = ksi * T_0 + (1.0 - ksi) * T_1;

			// velocity
			if (f.U.dot(f.n01) > 0)
			{
				const Scalar u_0 = f.c0->U.x() + f.c0->grad_U.col(0).dot(f.r0);
				const Scalar v_0 = f.c0->U.y() + f.c0->grad_U.col(1).dot(f.r0);
				const Scalar w_0 = f.c0->U.z() + f.c0->grad_U.col(2).dot(f.r0);

				f.U = { u_0, v_0, w_0 };
			}
			else
			{
				const Scalar u_1 = f.c1->U.x() + f.c1->grad_U.col(0).dot(f.r1);
				const Scalar v_1 = f.c1->U.y() + f.c1->grad_U.col(1).dot(f.r1);
				const Scalar w_1 = f.c1->U.z() + f.c1->grad_U.col(2).dot(f.r1);

				f.U = { u_1, v_1, w_1 };
			}

			// density
			if (f.U.dot(f.n01) > 0)
				f.rho = f.c0->rho + f.c0->grad_rho.dot(f.r0);
			else
				f.rho = f.c1->rho + f.c1->grad_rho.dot(f.r1);
		}
	}
}

void calcFaceProperty()
{
	for (auto &f : face)
	{
		// Dynamic viscosity
		f.mu = Sutherland(f.T);
	}
}

void calcCellFlux()
{
	// Continuity equation
	// TODO

	// Momentum equation
	for (auto &c : cell)
	{
		c.pressure_flux.setZero();
		c.convection_flux.setZero();
		c.viscous_flux.setZero();

		for (size_t j = 0; j < c.S.size(); ++j)
		{
			auto f = c.surface.at(j);
			const auto &Sf = c.S.at(j);

			// pressure term
			const Vector cur_pressure_flux = f->p * Sf;
			c.pressure_flux += cur_pressure_flux;

			// convection term
			const Vector cur_convection_flux = f->rhoU * f->U.dot(Sf);
			c.convection_flux += cur_convection_flux;

			// viscous term
			const Vector cur_viscous_flux = { Sf.dot(f->tau.col(0)), Sf.dot(f->tau.col(1)), Sf.dot(f->tau.col(2)) };
			c.viscous_flux += cur_viscous_flux;
		}
	}

	// Energy equation
	// TODO
}

void calcFace_rhoU_star()
{
	for (auto &f : face)
	{
		if (f.atBdry)
		{
			f.rhoU_star = f.rhoU;
		}
		else
		{
			f.rhoU_star = (f.c0->rhoU_star + f.c1->rhoU0) / 2;
		}
	}
}

void calcPoissonEquationCoefficient(Eigen::SparseMatrix<Scalar> &A, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs)
{
	rhs.setZero();
	std::vector<Eigen::Triplet<Scalar>> coef;

	for (const auto &C : cell)
	{
		// Initialize coefficient baseline.
		std::map<int, double> cur_coef;
		cur_coef[C.index] = 0.0;
		for (auto F : C.adjCell)
		{
			if (F)
			{
				cur_coef[F->index] = 0.0;
				for (auto FF : F->adjCell)
					cur_coef[FF->index] = 0.0;
			}
		}

		// Compute coefficient contributions.
		const auto N_C = C.surface.size();
		for (auto f = 1; f <= N_C; ++f)
		{
			const auto &S_f = C.S(f);
			auto curFace = C.surface(f);

			// Neumann B.C. for 'dp' by default.
			// No need to handle boundary case as zero-gradient is assumed.

			if (curFace->atBdry)
			{
				auto F = C.adjCell(f);
				const auto N_F = F->surface.size();

				const Vector r_C = curFace->center - C.center;
				const Vector r_F = curFace->center - F->center;
				const Scalar d_f = (F->center - C.center).norm();
				const Vector e_f = (r_C - r_F) / d_f;
				const Scalar ksi_f = 1.0 / (1.0 + r_F.norm() / r_C.norm());
				const Scalar x_f = e_f.dot(S_f);
				const Vector y_f = S_f - x_f * e_f;

				const Eigen::VectorXd J_C = ksi_f * y_f.transpose() * C.J_INV_p;
				const Eigen::VectorXd J_F = (1.0 - ksi_f) * y_f.transpose() * F->J_INV_p;

				// Part1
				cur_coef[F->index] += x_f / d_f;
				cur_coef[C.index] -= x_f / d_f;

				// Part2
				for (auto i = 0; i < N_C; ++i)
				{
					auto C_i = C.adjCell.at(i);

					// No need to handle boundary case as zero-gradient is assumed.

					if (C_i)
					{
						cur_coef[C_i->index] += J_C(i);
						cur_coef[C.index] -= J_C(i);
					}
				}

				// Part3
				for (auto i = 0; i < N_F; ++i)
				{
					auto F_i = F->adjCell.at(i);

					// No need to handle boundary case as zero-gradient is assumed.

					if (F_i)
					{
						cur_coef[F_i->index] += J_F(i);
						cur_coef[F->index] -= J_F(i);
					}
				}
			}
		}

		// Record current line
		for (auto it = cur_coef.begin(); it != cur_coef.end(); ++it)
			coef.emplace_back(C.index - 1, it->first - 1, it->second);

		// RHS term
		for (auto f = 0; f < N_C; ++f)
		{
			const auto &S_f = C.S(f);
			auto curFace = C.surface(f);

			rhs(C.index - 1) += curFace->rhoU_star.dot(S_f);
		}
	}

	A.setFromTriplets(coef.begin(), coef.end());
}

/*********************************************** Temporal Discretization *********************************************/

/**
 * Explicit Fractional-Step Method.
 * @param TimeStep
 */
void FSM(Scalar TimeStep)
{
	BC();

	// Physical properties at centroid of each cell
	calcCellProperty();

	// Boundary ghost values if any
	calcFaceGhostVariable();

	// Gradients at centroid of each cell
	calcCellGradient();

	// Gradients at centroid of each face
	calcFaceGradient();

	// Interpolate values on each face
	calcFaceValue();

	// Physical properties at centroid of each face
	calcFaceProperty();

	calcCellFlux();

	// Prediction
	for (auto &c : cell)
		c.rhoU_star = c.rhoU0 + TimeStep / c.volume * (c.pressure_flux + c.viscous_flux - c.convection_flux);

	// rhoU_star at each face
	calcFace_rhoU_star();

	// Correction
	Eigen::SparseMatrix<Scalar> A(NumOfCell, NumOfCell);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(NumOfCell);
	calcPoissonEquationCoefficient(A, b);
	b /= TimeStep;
	A.makeCompressed();
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
	solver.analyzePattern(A);
	solver.factorize(A);
	Eigen::VectorXd dp = solver.solve(b);

	// Update
	// TODO
}

/**
 * 4-step Runge-Kutta time-marching.
 * @param TimeStep
 */
void RK4(Scalar TimeStep)
{
	/* Init */
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		c.rho = c.rho0;
		c.U = c.U0;
		c.p = c.p0;
		c.T = c.T0;
	}

	static const std::array<Scalar, 4> alpha = { 1.0 / 4, 1.0 / 3, 1.0 / 2, 1.0 };

	/* Step 1-4 */
	for (size_t m = 0; m < 4; ++m)
	{
		const Scalar cur_dt = alpha[m] * TimeStep;

		// Update pressure-velocity coupling, and compute residuals.
		FSM(cur_dt);

		// Update scalars
		for (size_t i = 1; i <= NumOfCell; ++i)
		{
			auto &c = cell(i);
			c.rho = c.rho0 + cur_dt * c.continuity_res;
			c.T = c.T0 + cur_dt * c.energy_res;
		}
	}

	/* Update */
	for (size_t i = 1; i <= NumOfCell; ++i)
	{
		auto &c = cell(i);
		c.rho0 = c.rho;
		c.U0 = c.U;
		c.p0 = c.p;
		c.T0 = c.T;
		c.rhoU0 = c.rho0 * c.U0;
	}
}

/***************************************************** Solution Control **********************************************/

bool diagnose()
{
	bool ret = false;

	return ret;
}

Scalar calcTimeStep()
{
	Scalar ret = 1e-5;

	return ret;
}

void solve(std::ostream &fout = std::cout)
{
	static const size_t OUTPUT_GAP = 100;

	int iter = 0;
	Scalar dt = 0.0; // s
	Scalar t = 0.0; // s
	bool done = false;
	while (!done)
	{
		fout << "Iter" << ++iter << ":" << std::endl;
		dt = calcTimeStep();
		fout << "\tt=" << t << "s, dt=" << dt << "s" << std::endl;
		RK4(dt);
		t += dt;
		done = diagnose();
		if (done || !(iter % OUTPUT_GAP))
		{
			updateNodalValue();
			writeTECPLOT_Nodal("flow" + std::to_string(iter) + "_NODAL.dat", "3D Cavity");
			writeTECPLOT_CellCentered("flow" + std::to_string(iter) + "_CELL.dat", "3D Cavity");
		}
	}
	fout << "Finished!" << std::endl;
}

/**
 * Initialize the computation environment.
 */
void init()
{
	readMSH("grid0.msh");
	BC_TABLE();
	calcLeastSquareCoefficients();
	IC();
}

/**
 * Solver entrance.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
	init();
	solve();

	return 0;
}

/********************************************************* END *******************************************************/
