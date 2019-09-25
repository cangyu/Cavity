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
const double Lx = 0.1, Ly = 0.1; // m
const size_t Nx = 257, Ny = 257;
const double xLeft = -Lx / 2, xRight = Lx / 2;
const double yBottom = -Ly / 2, yTop = Ly / 2;
const double dx = Lx / (Nx - 1), dy = Ly / (Ny - 1);

// Flow param
const double Re = 400.0;
const double rho = 1.225; // kg/m^3
const double p0 = 101325.0; // Operating pressure, Pa
const double u0 = 0.1; // m/s
const double v0 = 0.0; // m/s
const double nu = u0 * max(Lx, Ly) / Re; // m^2 / s
const double mu = rho * nu; // Pa*s

// Timing
const double CFL = 0.5;
double dt = 0.0; // s
double t = 0.0; // s
size_t iter = 0;
const size_t MAX_ITER = 100000;

// Coordinate
Array1D x(Nx, 0.0), y(Ny, 0.0); // m
Array1D xP(Nx + 1, 0.0), yP(Ny + 1, 0.0); // m
Array1D xU(Nx, 0.0), yU(Ny + 1, 0.0); // m
Array1D xV(Nx + 1, 0.0), yV(Ny, 0.0); // m

// Variables
Array2D p(Nx + 1, Ny + 1, 0.0); // Pa
Array2D u(Nx, Ny + 1, 0.0); // m/s
Array2D v(Nx + 1, Ny, 0.0); // m/s

Array2D u_star(Nx, Ny + 1, 0.0);
Array2D v_star(Nx + 1, Ny, 0.0);

Array2D p_prime(Nx + 1, Ny + 1, 0.0);
Array2D u_prime(Nx, Ny + 1, 0.0);
Array2D v_prime(Nx + 1, Ny, 0.0);

// Poisson equation
const size_t m = (Nx + 1) * (Ny + 1);
Eigen::SparseMatrix<double> A(m, m);
Eigen::VectorXd rhs(m);
Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
const int bw = Nx + 1; // Band-width

// 0-based indexing of stencil pts
inline int stencil_center_idx(int i, int j)
{
	// Convert i, j from 1-based to 0-based indexing.
	const int i0 = i - 1;
	const int j0 = j - 1;

	return(i0 + j0 * bw);
}

inline void poisson_stencil(int i, int j, int &id, int &id_w, int &id_e, int &id_n, int &id_s)
{
	// Calculating 0-based index of stencil pts
	id = stencil_center_idx(i, j);
	id_w = id - 1;
	id_e = id + 1;
	id_n = id + bw;
	id_s = id - bw;
}

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

inline double df_upwind(double fl, double fc, double fr, double pl, double pc, double pr, double dir)
{
	if (dir > 0)
		return (fc - fl) / (pc - pl);
	else
		return (fr - fc) / (pr - pc);
}

inline double df_upwind2(
	double fl2, double fl1, double fc, double fr1, double fr2,
	double pl2, double pl1, double pc, double pr1, double pr2, double dir)
{
	if (dir > 0)
	{
		const double df2 = fl2 - fc;
		const double df1 = fl1 - fc;
		const double d2 = 1.0 / (pl2 - pc);
		const double d1 = 1.0 / (pl1 - pc);

		return (df1 * pow(d1, 2) - df2 * pow(d2, 2)) / (d1 - d2);
	}
	else
	{
		const double df2 = fr2 - fc;
		const double df1 = fr1 - fc;
		const double d2 = 1.0 / (pr2 - pc);
		const double d1 = 1.0 / (pr1 - pc);

		return (df1 * pow(d1, 2) - df2 * pow(d2, 2)) / (d1 - d2);
	}
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
	double x0, double y0)
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
	for (size_t i = 2; i <= Nx - 1; ++i)
	{
		u_(i, 1) = 0.0;
		u_(i, Ny + 1) = u0;
	}
	for (size_t j = 1; j <= Ny + 1; ++j)
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
	double dt = numeric_limits<double>::max();

	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
		{
			double loc_dt = dx / (abs(u(i, j)) + numeric_limits<double>::epsilon());
			if (loc_dt < dt)
				dt = loc_dt;
		}

	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
		{
			double loc_dt = dy / (abs(v(i, j)) + numeric_limits<double>::epsilon());
			if (loc_dt < dt)
				dt = loc_dt;
		}

	dt *= CFL;
	return dt;
}

void init()
{
	cout << "Re=" << Re << endl;
	cout << "rho=" << rho << endl;
	cout << "u0=" << u0 << endl;
	cout << "mu=" << mu << endl;

	/********************************** Grid **********************************/
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

	/*********************************** I.C. *********************************/
	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Ny + 1; ++i)
			p(i, j) = p0;

	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
			u(i, j) = 0.0;

	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
			v(i, j) = 0.0;
	/*********************************** B.C. *********************************/
	set_velocity_bc(u, v);

	/***************************** Poisson equation ***************************/
	vector<Eigen::Triplet<double>> coef; // Coefficient matrix
	int id, id_w, id_e, id_n, id_s;

	// Inner
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			const double dxL = xP(i) - xP(i - 1);
			const double dxR = xP(i + 1) - xP(i);
			const double dxC = xP(i + 1) - xP(i - 1);
			const double dyL = yP(j) - yP(j - 1);
			const double dyR = yP(j + 1) - yP(j);
			const double dyC = yP(j + 1) - yP(j - 1);

			const double a = -2.0 * (1.0 / (dxR * dxL) + 1.0 / (dyR * dyL));
			const double b_w = 2.0 / (dxC * dxL);
			const double b_e = 2.0 / (dxC * dxR);
			const double c_n = 2.0 / (dyC * dyR);
			const double c_s = 2.0 / (dyC * dyL);

			poisson_stencil(i, j, id, id_w, id_e, id_n, id_s);

			coef.emplace_back(id, id, a);
			coef.emplace_back(id, id_w, b_w);
			coef.emplace_back(id, id_e, b_e);
			coef.emplace_back(id, id_n, c_n);
			coef.emplace_back(id, id_s, c_s);
		}
	// Left
	for (size_t j = 2; j <= Ny; ++j)
	{
		poisson_stencil(1, j, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_e, -1.0);
	}
	// Right
	for (size_t j = 2; j <= Ny; ++j)
	{
		poisson_stencil(Nx + 1, j, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_w, -1.0);
	}
	// Bottom
	for (size_t i = 2; i <= Nx; ++i)
	{
		poisson_stencil(i, 1, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_n, -1.0);
	}
	// Top
	for (size_t i = 2; i <= Nx; ++i)
	{
		poisson_stencil(i, Ny + 1, id, id_w, id_e, id_n, id_s);
		coef.emplace_back(id, id, 1.0);
		coef.emplace_back(id, id_s, -1.0);
	}
	// Left-Bottom
	poisson_stencil(1, 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_n, -0.5);
	coef.emplace_back(id, id_e, -0.5);
	// Right-Bottom
	poisson_stencil(Nx + 1, 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_n, -0.5);
	coef.emplace_back(id, id_w, -0.5);
	// Left-Top
	poisson_stencil(1, Ny + 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_s, -0.5);
	coef.emplace_back(id, id_e, -0.5);
	// Right-Top
	poisson_stencil(Nx + 1, Ny + 1, id, id_w, id_e, id_n, id_s);
	coef.emplace_back(id, id, 1.0);
	coef.emplace_back(id, id_s, -0.5);
	coef.emplace_back(id, id_w, -0.5);

	// Construct sparse matrix
	A.setFromTriplets(coef.begin(), coef.end());
	solver.analyzePattern(A);
	solver.factorize(A);
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
			p_interp(i, j) = interp_f(
				p(i, j + 1), xP(i), yP(j + 1),
				p(i + 1, j + 1), xP(i + 1), yP(j + 1),
				p(i + 1, j), xP(i + 1), yP(j),
				p(i, j), xP(i), yP(j),
				x(i), y(j));

	// u
	for (size_t i = 1; i <= Nx; ++i)
	{
		u_interp(i, 1) = u(i, 1); // Bottom
		u_interp(i, Ny) = u(i, Ny + 1); // Top
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
			u_interp(i, j) = relaxation(u(i, j), u(i, j + 1), 0.5);

	// v
	for (size_t j = 1; j <= Ny; ++j)
	{
		v_interp(1, j) = v(1, j); // Left
		v_interp(Nx, j) = v(Nx + 1, j); // Right
	}
	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			v_interp(i, j) = relaxation(v(i, j), v(i + 1, j), 0.5);

	/********************************* Output *********************************/
	// Create output file
	ofstream fout("flow" + to_string(n) + ".dat");
	if (fout.fail())
		throw runtime_error("Failed to create data file!");

	// Header
	fout << R"(TITLE = "2D lid-driven cavity flow at t=)" << t << R"(s")" << endl;
	fout << R"(VARIABLES = "X", "Y", "P", "U", "V")" << endl;
	fout << "ZONE I=" << Nx << ", J=" << Ny << ", F=POINT" << endl;

	// Flow-field data
	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx; ++i)
		{
			fout << setw(WIDTH) << setprecision(DIGITS) << x(i);
			fout << setw(WIDTH) << setprecision(DIGITS) << y(j);
			fout << setw(WIDTH) << setprecision(DIGITS) << p_interp(i, j);
			fout << setw(WIDTH) << setprecision(DIGITS) << u_interp(i, j);
			fout << setw(WIDTH) << setprecision(DIGITS) << v_interp(i, j);
			fout << endl;
		}

	// Finalize
	fout.close();
}

void output()
{
	if (!(iter % 100))
		write_tecplot(iter);

	write_user(iter);
}

void compute_source()
{
	/******************************* RHS at inner *****************************/
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			const double dusdx = (u_star(i, j) - u_star(i - 1, j)) / (xU(i) - xU(i - 1));
			const double dvsdy = (v_star(i, j) - v_star(i, j - 1)) / (yV(j) - yV(j - 1));
			const double divergence = dusdx + dvsdy;
			const double p_rhs = rho / dt * divergence;
			rhs(stencil_center_idx(i, j)) = p_rhs;
		}

	/***************************** RHS at boundary ****************************/
	// Left
	for (size_t j = 2; j <= Ny; ++j)
		rhs(stencil_center_idx(1, j)) = 0.0;

	// Right
	for (size_t j = 2; j <= Ny; ++j)
		rhs(stencil_center_idx(Nx + 1, j)) = 0.0;

	// Bottom
	for (size_t i = 2; i <= Nx; ++i)
		rhs(stencil_center_idx(i, 1)) = 0.0;

	// Top
	for (size_t i = 2; i <= Nx; ++i)
		rhs(stencil_center_idx(i, Ny + 1)) = 0.0;

	/**************************** RHS at 4 corners ****************************/
	// Left-Bottom
	rhs(stencil_center_idx(1, 1)) = 0.0;
	// Right-Bottom
	rhs(stencil_center_idx(Nx + 1, 1)) = 0.0;
	// Left-Top
	rhs(stencil_center_idx(1, Ny + 1)) = 0.0;
	// Right-Top
	rhs(stencil_center_idx(Nx + 1, Ny + 1)) = 0.0;
}

// Explicit time-marching
void ProjectionMethod()
{
	/******************************** Prediction ******************************/
	// Approximated values at inner
	Array2D v_bar(Nx, Ny + 1, 0.0);
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
			v_bar(i, j) = interp_f(v(i, j), xV(i), yV(j), v(i + 1, j), xV(i + 1), yV(j), v(i + 1, j - 1), xV(i + 1), yV(j - 1), v(i, j - 1), xV(i), yV(j - 1), xU(i), yU(j));

	Array2D u_bar(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
			u_bar(i, j) = interp_f(u(i - 1, j + 1), xU(i - 1), yU(j + 1), u(i, j + 1), xU(i), yU(j + 1), u(i, j), xU(i), yU(j), u(i - 1, j), xU(i - 1), yU(j), xV(i), yV(j));

	// Derivateives at inner
	Array2D dudx(Nx, Ny + 1, 0.0), dduddx(Nx, Ny + 1, 0.0);
	Array2D dudy(Nx, Ny + 1, 0.0), dduddy(Nx, Ny + 1, 0.0);
	Array2D dpdx(Nx, Ny + 1, 0.0);
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			if (i == 2 || i == Nx - 1)
				dudx(i, j) = df(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));
			else
				//dudx(i, j) = df_upwind(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1), u(i, j));
				dudx(i, j) = df_upwind2(u(i - 2, j), u(i - 1, j), u(i, j), u(i + 1, j), u(i + 2, j), xU(i - 2), xU(i - 1), xU(i), xU(i + 1), xU(i + 2), u(i, j));

			dduddx(i, j) = ddf(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));

			if (j == 2 || j == Ny)
				dudy(i, j) = df(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
			else
				//dudy(i, j) = df_upwind(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1), v_bar(i, j));
				dudy(i, j) = df_upwind2(u(i, j - 2), u(i, j - 1), u(i, j), u(i, j + 1), u(i, j + 2), yU(j - 2), yU(j - 1), yU(j), yU(j + 1), yU(j + 2), v_bar(i, j));

			dduddy(i, j) = ddf(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));

			dpdx(i, j) = (p(i + 1, j) - p(i, j)) / (xP(i + 1) - xP(i));
		}

	Array2D dvdx(Nx + 1, Ny, 0.0), ddvddx(Nx + 1, Ny, 0.0);
	Array2D dvdy(Nx + 1, Ny, 0.0), ddvddy(Nx + 1, Ny, 0.0);
	Array2D dpdy(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			if (i == 2 || i == Nx)
				dvdx(i, j) = df(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));
			else
				//dvdx(i, j) = df_upwind(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1), u_bar(i, j));
				dvdx(i, j) = df_upwind2(v(i - 2, j), v(i - 1, j), v(i, j), v(i + 1, j), v(i + 2, j), xV(i - 2), xV(i - 1), xV(i), xV(i + 1), xV(i + 2), u_bar(i, j));

			ddvddx(i, j) = ddf(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));

			if (j == 2 || j == Ny - 1)
				dvdy(i, j) = df(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
			else
				//dvdy(i, j) = df_upwind(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1), v(i, j));
				dvdy(i, j) = df_upwind2(v(i, j - 2), v(i, j - 1), v(i, j), v(i, j + 1), v(i, j + 2), yV(j - 2), yV(j - 1), yV(j), yV(j + 1), yV(j + 2), v(i, j));

			ddvddy(i, j) = ddf(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));

			dpdx(i, j) = (p(i, j + 1) - p(i, j)) / (yP(j + 1) - yP(i));
		}

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

	/***************************** Poisson Equation ***************************/
	// RHS
	compute_source();

	// Solve the linear system: Ax = rhs
	Eigen::VectorXd res = solver.solve(rhs);

	// Update p
	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
			p(i, j) = res(stencil_center_idx(i, j));

	/******************************** Correction ******************************/
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
	// inf-norm
	double u_res = 0.0;
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			auto du = abs(u_prime(i, j));
			if (du > u_res)
				u_res = du;
		}
	u_res = log10(u_res);

	double v_res = 0.0;
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			auto dv = abs(v_prime(i, j));
			if (dv > v_res)
				v_res = dv;
		}
	v_res = log10(v_res);

	cout << "\tlog10(|u'|)=" << u_res << ", log10(|v'|)=" << v_res << endl;

	// divergence
	double div_max = 0.0;
	size_t i_max, j_max;
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			double loc_div = (u(i, j) - u(i - 1, j)) / (xU(i) - xU(i - 1)) + (v(i, j) - v(i, j - 1)) / (yV(j) - yV(j - 1));
			loc_div = abs(loc_div);
			if (loc_div > div_max)
			{
				div_max = loc_div;
				i_max = i;
				j_max = j;
			}
		}

	cout << "\tMax divergence: " << div_max << " at: (" << i_max << ", " << j_max << ")" << endl;

	return max(u_res, v_res) < -3 || iter > MAX_ITER;
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
