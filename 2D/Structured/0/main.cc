#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <Eigen/Sparse>

using namespace std;

class Array1D
{
private:
	vector<double> m_data;

public:
	Array1D(size_t nx, double val = 0.0) :
		m_data(nx, val)
	{
		if (nx == 0)
			throw runtime_error("Invalid size: nx=" + to_string(nx));
	}

	~Array1D() = default;

	// 0-based indexing
	double &at(size_t i)
	{
		return m_data[i];
	}

	double at(size_t i) const
	{
		return m_data[i];
	}

	// 1-based indexing
	double &operator()(size_t i)
	{
		return at(i - 1);
	}

	double operator()(size_t i) const
	{
		return at(i - 1);
	}
};

class Array2D
{
private:
	vector<double> m_data;
	size_t m_Nx, m_Ny;

public:
	Array2D(size_t nx, size_t ny, double val = 0.0) :
		m_Nx(nx),
		m_Ny(ny),
		m_data(nx * ny, val)
	{
		if (nx == 0 || ny == 0)
			throw runtime_error("Invalid size: nx=" + to_string(nx) + ", ny=" + to_string(ny));
	}

	~Array2D() = default;

	// 0-based indexing
	double &at(size_t i, size_t j)
	{
		return m_data[idx(i, j)];
	}

	double at(size_t i, size_t j) const
	{
		return m_data[idx(i, j)];
	}

	// 1-based indexing
	double &operator()(size_t i, size_t j)
	{
		return at(i - 1, j - 1);
	}

	double operator()(size_t i, size_t j) const
	{
		return at(i - 1, j - 1);
	}

private:
	// Internal 0-based indexing interface.
	size_t idx(size_t i, size_t j) const
	{
		return i + m_Nx * j;
	}
};

// Geom
const double Lx = 0.1, Ly = 0.08; // m
const size_t Nx = 51, Ny = 41;
const double xLeft = -Lx / 2, xRight = Lx / 2;
const double yBottom = -Ly / 2, yTop = Ly / 2;
const double dx = Lx / (Nx - 1), dy = Ly / (Ny - 1);

// Flow param
const double Re = 100.0;
const double rho = 1.0; // kg/m^3
const double p0 = 101325.0; // Pa
const double u0 = 0.1; // m/s
const double v0 = 0.0; // m/s
const double nu = u0 * max(Lx, Ly) / Re; // m^2 / s
const double mu = rho * nu; // Pa*s

// Timing
double dt = 0.0; // s
double t = 0.0; // s
size_t iter = 0;
const size_t MAX_ITER = 5000;

// Coordinate
Array1D x(Nx, 0.0), y(Ny, 0.0); // m
Array1D xP(Nx + 1, 0.0), yP(Ny + 1, 0.0);
Array1D xU(Nx, 0.0), yU(Ny + 1, 0.0);
Array1D xV(Nx + 1, 0.0), yV(Ny, 0.0);

// Variables
Array2D p(Nx + 1, Ny + 1, 0.0); // Pa
Array2D u(Nx, Ny + 1, 0.0); // m/s
Array2D v(Nx + 1, Ny, 0.0); // m/s

Array2D p_star(Nx + 1, Ny + 1, 0.0);
Array2D u_star(Nx, Ny + 1, 0.0);
Array2D v_star(Nx + 1, Ny, 0.0);

Array2D p_prime(Nx + 1, Ny + 1, 0.0);
Array2D u_prime(Nx, Ny + 1, 0.0);
Array2D v_prime(Nx + 1, Ny, 0.0);

inline double distance(double x_src, double y_src, double x_dst, double y_dst)
{
	return sqrt(pow(x_dst - x_src, 2) + pow(y_dst - y_src, 2));
}

template<typename T>
inline T relaxation(const T &a, const T &b, double alpha)
{
	return (1 - alpha) *a + alpha * b;
}

inline double df(double fl, double fc, double fr, double pl, double pc, double pr)
{
	return ((fr - fc) / (pr - pc) * (pc - pl) + (fl - fc) / (pl - pc) * (pr - pc)) / (pr - pl);
}

inline double ddf(double fl, double fc, double fr, double pl, double pc, double pr)
{
	return 2.0 / (pr - pl) * ((fr - fc) / (pr - pc) - (fl - fc) / (pl - pc));
}

// Shepard Interpolation
inline double interp_f(
	double f_nw, double x_nw, double y_nw,
	double f_ne, double x_ne, double y_ne,
	double f_se, double x_se, double y_se,
	double f_sw, double x_sw, double y_sw,
	double x0, double y0
)
{
	const double h[4] = {
		distance(x_nw, y_nw, x0, y0),
		distance(x_ne, y_ne, x0, y0),
		distance(x_se, y_se, x0, y0),
		distance(x_sw, y_sw, x0, y0)
	};

	double hp[4], hp_sum = 0.0;
	for (auto i = 0; i < 4; ++i)
	{
		hp[i] = 1.0 / pow(h[i], 2);
		hp_sum += hp[i];
	}

	double w[4];
	for (auto i = 0; i < 4; ++i)
		w[i] = hp[i] / hp_sum;

	const double f = w[0] * f_nw + w[1] * f_ne + w[2] * f_se + w[3] * f_sw;

	return f;
}

void set_velocity_bc(Array2D &u_, Array2D &v_)
{
	// u
	for (size_t i = 1; i <= Nx; ++i)
	{
		u_(i, 1) = 0.0;
		u_(i, Ny + 1) = u0;
	}
	for (size_t j = 2; j <= Ny; ++j)
	{
		u_(1, j) = 0.0;
		u_(Nx, j) = 0.0;
	}

	// v
	for (size_t i = 1; i <= Nx + 1; ++i)
	{
		v_(i, 1) = 0.0;
		v_(i, Ny) = 0.0;
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
	{
		v_(1, j) = 0.0;
		v_(Nx + 1, j) = 0.0;
	}
}

double TimeStep()
{
	// TODO
	return 1e-3;
}

void init()
{
	cout << "Re=" << Re << endl;
	cout << "rho=" << rho << endl;
	cout << "u0=" << u0 << endl;
	cout << "mu=" << mu << endl;

	// Grid of geom
	for (size_t i = 1; i <= Nx; ++i)
		x(i) = relaxation(xLeft, xRight, 1.0 * (i - 1) / (Nx - 1));
	for (size_t j = 1; j <= Ny; ++j)
		y(j) = relaxation(yBottom, yTop, 1.0 * (j - 1) / (Ny - 1));

	// Grid of p
	xP(1) = xLeft;
	for (size_t i = 2; i <= Nx; ++i)
		xP(i) = relaxation(x(i - 1), x(i), 0.5);
	xP(Nx + 1) = xRight;

	yP(1) = yBottom;
	for (size_t j = 2; j <= Ny; ++j)
		yP(j) = relaxation(y(j - 1), y(j), 0.5);
	yP(Ny + 1) = yTop;

	// Grid of u
	for (size_t i = 1; i <= Nx; ++i)
		xU(i) = x(i);

	yU(1) = yBottom;
	for (size_t j = 2; j <= Ny; ++j)
		yU(j) = relaxation(y(j - 1), y(j), 0.5);
	yU(Ny + 1) = yTop;

	// Grid of v
	xV(1) = xLeft;
	for (size_t i = 2; i <= Nx; ++i)
		xV(i) = relaxation(x(i - 1), x(i), 0.5);
	xV(Nx + 1) = xRight;

	for (size_t j = 1; j <= Ny; ++j)
		yV(j) = y(j);

	// I.C.
	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Ny + 1; ++i)
			p(i, j) = p0;

	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
			u(i, j) = u0;

	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
			v(i, j) = v0;

	// B.C.
	set_velocity_bc(u, v);
}

void write_user(size_t n)
{
	// TODO
}

void write_tecplot(size_t n)
{
	// Output format params
	static const size_t WIDTH = 18;
	static const size_t DIGITS = 9;

	/******************************* Interpolate ******************************/
	Array2D p_interp(Nx, Ny, 0.0);
	Array2D u_interp(Nx, Ny, 0.0);
	Array2D v_interp(Nx, Ny, 0.0);

	// p
	p_interp(1, 1) = p(1, 1);
	p_interp(Nx, 1) = p(Nx + 1, 1);
	p_interp(1, Ny) = p(1, Ny + 1);
	p_interp(Nx, Ny) = p(Nx + 1, Ny + 1);
	for (size_t i = 2; i <= Nx - 1; ++i)
	{
		p_interp(i, 1) = relaxation(p(i, 1), p(i + 1, 1), 0.5);
		p_interp(i, Ny) = relaxation(p(i, Ny + 1), p(i + 1, Ny + 1), 0.5);
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
	{
		p_interp(1, j) = relaxation(p(1, j), p(1, j + 1), 0.5);
		p_interp(Nx, j) = relaxation(p(Nx + 1, j), p(Nx + 1, j + 1), 0.5);
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			p_interp(i, j) = 0.25*(p(i, j) + p(i + 1, j) + p(i, j + 1) + p(i + 1, j + 1));

	// u
	for (size_t i = 1; i <= Nx; ++i)
	{
		u_interp(i, 1) = u(i, 1);
		u_interp(i, Ny) = u(i, Ny + 1);
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
			u_interp(i, j) = relaxation(u(i, j), u(i, j + 1), 0.5);

	// v
	for (size_t j = 1; j <= Ny; ++j)
	{
		v_interp(1, j) = v(1, j);
		v_interp(Nx, j) = v(Nx, j);
	}
	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			v_interp(i, j) = relaxation(v(i, j), v(i + 1, j), 0.5);

	/********************************* Output *********************************/
	// Create output file
	ofstream fout("flow" + to_string(n) + ".dat");
	if (!fout)
		throw runtime_error("Failed to create data file!");

	// Header
	fout << R"(TITLE = "2D Lid-Driven Cavity Flow at t=)" << t << R"(s")" << endl;
	fout << R"(VARIABLES = "X", "Y", "rho", "U", "V", "P")" << endl;
	fout << "ZONE I=" << Nx << ", J=" << Ny << ", F=POINT" << endl;

	// Flow-field data
	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx; ++i)
		{
			fout << setw(WIDTH) << setprecision(DIGITS) << x(i);
			fout << setw(WIDTH) << setprecision(DIGITS) << y(j);
			fout << setw(WIDTH) << setprecision(DIGITS) << rho;
			fout << setw(WIDTH) << setprecision(DIGITS) << u(i, j);
			fout << setw(WIDTH) << setprecision(DIGITS) << v(i, j);
			fout << setw(WIDTH) << setprecision(DIGITS) << p(i, j);
			fout << endl;
		}

	// Finalize
	fout.close();
}

void output()
{
	if (!(iter % 10))
		write_tecplot(iter);

	write_user(iter);
}

// 0-based indexing of stencil pts
inline void poisson_stencil(int i, int j, int &id, int &id_w, int &id_e, int &id_n, int &id_s)
{
	// Convert i, j from 1-based to 0-based indexing.
	const int i0 = i - 1;
	const int j0 = j - 1;

	// Set the band-width
	const int bw = Nx + 1;

	// Calculating 0-based index of stencil pts
	id = j0 * bw + i0;
	id_w = id - 1;
	id_e = id + 1;
	id_n = id + bw;
	id_s = id - bw;
}

void solvePoissonEquation()
{
	typedef Eigen::SparseMatrix<double> SpMat;
	typedef Eigen::Triplet<double> T;

	const size_t m = (Nx + 1) * (Ny + 1);
	vector<T> coef;
	Eigen::VectorXd rhs(m);
	SpMat A(m, m);

	int id, id_w, id_e, id_n, id_s;

	/***************************** Equations at inner *************************/
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			const double dxL = xP(i) - xP(i - 1);
			const double dxR = xP(i + 1) - xP(i);
			const double dxC = xP(i + 1) - xP(i - 1);
			const double dyL = yP(j) - yP(j - 1);
			const double dyR = yP(j + 1) - yP(j);
			const double dyC = yP(j + 1) - yP(j - 1);

			const double dusdx = (u_star(i, j) - u_star(i - 1, j)) / (xU(i) - xU(i - 1));
			const double dvsdy = (v_star(i, j) - v_star(i, j - 1)) / (yV(j) - yV(j - 1));
			const double divergence = dusdx + dvsdy;
			const double p_rhs = rho / dt * divergence;

			poisson_stencil(i, j, id, id_w, id_e, id_n, id_s);

			const double a = -2.0 * (1.0 / (dxR * dxL) + 1.0 / (dyR * dyL));
			const double b_w = 2.0 / (dxC * dxL);
			const double b_e = 2.0 / (dxC * dxR);
			const double c_n = 2.0 / (dyC * dyR);
			const double c_s = 2.0 / (dyC * dyL);

			coef.emplace_back(id, id, a);
			coef.emplace_back(id, id_w, b_w);
			coef.emplace_back(id, id_e, b_e);
			coef.emplace_back(id, id_n, c_n);
			coef.emplace_back(id, id_s, c_s);
			rhs(id) = p_rhs;
		}

	/*************************** Equations at boundary ************************/
	// Left
	for (size_t j = 2; j <= Ny; ++j)
	{
		poisson_stencil(1, j, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_e, -1.0);
		rhs(id) = 0.0;
	}
	// Right
	for (size_t j = 2; j <= Ny; ++j)
	{
		poisson_stencil(Nx + 1, j, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_w, -1.0);
		rhs(id) = 0.0;
	}
	// Bottom
	for (size_t i = 2; i <= Nx; ++i)
	{
		poisson_stencil(i, 1, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_n, -1.0);
		rhs(id) = 0.0;
	}
	// Top
	for (size_t i = 2; i <= Nx; ++i)
	{
		poisson_stencil(i, Ny + 1, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_s, -1.0);
		rhs(id) = 0.0;
	}

	/************************** Equations at 4 corners ************************/
	// Left-Bottom
	poisson_stencil(1, 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_n, -0.5);
	coef.emplace_back(id, id_e, -0.5);
	rhs(id) = 0.0;
	// Right-Bottom
	poisson_stencil(Nx + 1, 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_n, -0.5);
	coef.emplace_back(id, id_w, -0.5);
	rhs(id) = 0.0;
	// Left-Top
	poisson_stencil(1, Ny + 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_s, -0.5);
	coef.emplace_back(id, id_e, -0.5);
	rhs(id) = 0.0;
	// Right-Top
	poisson_stencil(Nx + 1, Ny + 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_s, -0.5);
	coef.emplace_back(id, id_w, -0.5);
	rhs(id) = 0.0;

	/*************************** Solve all equations **************************/
	// Construct sparse matrix
	A.setFromTriplets(coef.begin(), coef.end());

	// Solve the linear system: Ax = rhs
	Eigen::SimplicialCholesky<SpMat> chl(A);
	Eigen::VectorXd res = chl.solve(rhs);

	// Update p
	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
		{
			poisson_stencil(i, j, id, id_w, id_e, id_n, id_s);
			p(i, j) = res(id);
		}
}

// Explicit time-marching
void ProjectionMethod()
{
	/******************************* Prediction ******************************/
	// Derivateives at inner
	Array2D dudx(Nx, Ny + 1, 0.0), dduddx(Nx, Ny + 1, 0.0);
	Array2D dudy(Nx, Ny + 1, 0.0), dduddy(Nx, Ny + 1, 0.0);
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			dudx(i, j) = df(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));
			dduddx(i, j) = ddf(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));
			dudy(i, j) = df(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
			dduddy(i, j) = ddf(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
		}

	Array2D dvdx(Nx + 1, Ny, 0.0), ddvddx(Nx + 1, Ny, 0.0);
	Array2D dvdy(Nx + 1, Ny, 0.0), ddvddy(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			dvdx(i, j) = df(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));
			ddvddx(i, j) = ddf(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));
			dvdy(i, j) = df(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
			ddvddy(i, j) = ddf(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
		}

	// Approximated values at inner
	Array2D v_bar(Nx, Ny + 1, 0.0);
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			v_bar(i, j) = interp_f(
				v(i, j), xV(i), yV(j),
				v(i + 1, j), xV(i + 1), yV(j),
				v(i + 1, j - 1), xV(i + 1), yV(j - 1),
				v(i, j - 1), xV(i), yV(j - 1),
				xU(i), yU(j));

	Array2D u_bar(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
			u_bar(i, j) = interp_f(
				u(i - 1, j + 1), xU(i - 1), yU(j + 1),
				u(i, j + 1), xU(i), yU(j + 1),
				u(i, j), xU(i), yU(j),
				u(i - 1, j), xU(i - 1), yU(j),
				xV(i), yV(j));

	// F at inner
	Array2D F2(Nx, Ny + 1, 0.0);
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			F2(i, j) = -(u(i, j)*dudx(i, j) + v_bar(i, j) * dudy(i, j)) + nu * (dduddx(i, j) + dduddy(i, j));

	Array2D F3(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
			F3(i, j) = -(u_bar(i, j) * dvdx(i, j) + v(i, j) * dvdy(i, j)) + nu * (ddvddx(i, j) + ddvddy(i, j));

	// Star values at inner
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			u_star(i, j) = u(i, j) + dt * F2(i, j);

	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
			v_star(i, j) = v(i, j) + dt * F3(i, j);

	// Star values at boundary
	set_velocity_bc(u_star, v_star);

	/******************************* Poisson ******************************/
	solvePoissonEquation();

	/******************************* Correction ******************************/
	// U, V at inner
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			u_prime(i, j) = -dt / rho * (p(i + 1, j) - p(i, j)) / (xP(i + 1) - xP(i));
			u(i, j) = u_star(i, j) + u_prime(i, j);
		}

	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			v_prime(i, j) = -dt / rho * (p(i, j + 1) - p(i, j)) / (yP(j + 1) - yP(j));
			v(i, j) = v_star(i, j) + v_prime(i, j);
		}

	// U, V at boundary
	set_velocity_bc(u, v);
}

bool checkConvergence()
{
	double rsd = numeric_limits<double>::max();

	return rsd < 1e-4 || iter > MAX_ITER;
}

void solve()
{
	bool ok = false;
	while (!ok)
	{
		cout << "Iter" << ++iter << ":" << endl;
		dt = TimeStep();
		cout << "\tt=" << t << "s, dt=" << dt << "s" << endl;
		ProjectionMethod();
		t += dt;
		output();
		ok = checkConvergence();
	}
	cout << "Converged!" << endl;
	write_tecplot(iter);
}

int main(int argc, char *argv[])
{
	init();
	output();
	solve();

	return 0;
}
