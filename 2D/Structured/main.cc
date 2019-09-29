#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <Eigen/Sparse>

using namespace std;

class Array1D : public vector<double>
{
public:
	Array1D(size_t nx, double val = 0.0) : vector<double>(nx, val) {}

	// 1-based indexing
	double &operator()(size_t i) { return vector<double>::at(i - 1); }

	double operator()(size_t i) const { return vector<double>::at(i - 1); }
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
const double Lx = 1.0, Ly = 1.0; // m
const size_t Nx = 129, Ny = 129;
const double xLeft = 0.0, xRight = Lx;
const double yBottom = 0.0, yTop = Ly;

// Flow param
const double Re = 400.0;
const double rho = 1.225; // kg/m^3
const double p0 = 101325.0; // Operating pressure, Pa
const double u0 = 1.0; // m/s
const double v0 = 0.0; // m/s
const double nu = u0 * max(Lx, Ly) / Re; // m^2 / s
const double mu = rho * nu; // Pa*s

// Timing
const double CFL = 0.5;
double dt = 0.0; // s
double t = 0.0; // s
size_t iter = 0;

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
const size_t m = (Nx - 1) * (Ny - 1);
Eigen::SparseMatrix<double> A(m, m);
Eigen::VectorXd rhs(m);
Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
const int bw = Nx - 1; // Band-width

// Statistics
const size_t TECPLOT_GAP = 500;
const string f_hist = "history.txt";
double max_divergence = 0.0;

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

inline double df_left2(double fl2, double fl1, double fc, double pl2, double pl1, double pc)
{
	const double df2 = fl2 - fc;
	const double df1 = fl1 - fc;
	const double d2 = 1.0 / (pl2 - pc);
	const double d1 = 1.0 / (pl1 - pc);
	return (df1 * pow(d1, 2) - df2 * pow(d2, 2)) / (d1 - d2);
}

inline double df_right2(double fc, double fr1, double fr2, double pc, double pr1, double pr2)
{
	const double df2 = fr2 - fc;
	const double df1 = fr1 - fc;
	const double d2 = 1.0 / (pr2 - pc);
	const double d1 = 1.0 / (pr1 - pc);
	return (df1 * pow(d1, 2) - df2 * pow(d2, 2)) / (d1 - d2);
}

inline double df_upwind2(double fl2, double fl1, double fc, double fr1, double fr2, double pl2, double pl1, double pc, double pr1, double pr2, double dir)
{
	if (dir > 0)
		return df_left2(fl2, fl1, fc, pl2, pl1, pc);
	else
		return df_right2(fc, fr1, fr2, pc, pr1, pr2);
}

inline double ddf(double fl, double fc, double fr, double pl, double pc, double pr)
{
	return 2.0 / (pr - pl) * ((fr - fc) / (pr - pc) - (fl - fc) / (pl - pc));
}

// Shepard Interpolation
inline double interp_f(double f_nw, double x_nw, double y_nw, double f_ne, double x_ne, double y_ne, double f_se, double x_se, double y_se, double f_sw, double x_sw, double y_sw, double x0, double y0)
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

double TimeStep()
{
	double dt = numeric_limits<double>::max();

	for (size_t j = 1; j <= Ny + 1; ++j)
		for (size_t i = 1; i <= Nx; ++i)
		{
			double loc_dt = (xP(i + 1) - xP(i)) / (abs(u(i, j)) + numeric_limits<double>::epsilon());
			if (loc_dt < dt)
				dt = loc_dt;
		}

	for (size_t j = 1; j <= Ny; ++j)
		for (size_t i = 1; i <= Nx + 1; ++i)
		{
			double loc_dt = (yP(j + 1) - yP(j)) / (abs(v(i, j)) + numeric_limits<double>::epsilon());
			if (loc_dt < dt)
				dt = loc_dt;
		}

	dt *= CFL;
	return dt;
}

// Index of stencil pts.
// Input is 1-based while output is 0-based.
inline int stencil_center_idx(int i, int j)
{
	// Convert i, j from 1-based to 0-based indexing.
	// Only inner pts are unknown, outside pressure are extropolated using boundary conditon.
	// This is the common practice to convert 2nd-class B.C. to 1st-class B.C. so that the discrete linear system has unique solution.
	const int i0 = i - 2;
	const int j0 = j - 2;

	return(i0 + j0 * bw);
}

// Index of both center and surrounding stencil pts.
// Input is 1-based while output is 0-based.
inline void poisson_stencil(int i, int j, int &id, int &id_w, int &id_e, int &id_n, int &id_s)
{
	id = stencil_center_idx(i, j);
	id_w = id - 1;
	id_e = id + 1;
	id_n = id + bw;
	id_s = id - bw;
}

void init()
{
	cout << "Re=" << Re << endl;
	cout << "rho=" << rho << endl;
	cout << "u0=" << u0 << endl;
	cout << "mu=" << mu << endl;

	ofstream fout(f_hist);
	if (fout.fail())
		throw runtime_error("Failed to create user-defined output file.");
	fout.close();

	/********************************** Grid **********************************/
	// Grid of geom
	for (size_t i = 1; i <= Nx; ++i)
		x(i) = relaxation(xLeft, xRight, 1.0 * (i - 1) / (Nx - 1));
	for (size_t j = 1; j <= Ny; ++j)
		y(j) = relaxation(yBottom, yTop, 1.0 * (j - 1) / (Ny - 1));

	// Grid of p
	xP(1) = xLeft - 0.5 * (x(2) - x(1));
	for (size_t i = 2; i <= Nx; ++i)
		xP(i) = relaxation(x(i - 1), x(i), 0.5);
	xP(Nx + 1) = xRight + 0.5 * (x(Nx) - x(Nx - 1));

	yP(1) = yBottom - 0.5 * (y(2) - y(1));
	for (size_t j = 2; j <= Ny; ++j)
		yP(j) = relaxation(y(j - 1), y(j), 0.5);
	yP(Ny + 1) = yTop + 0.5 * (y(Ny) - y(Ny - 1));

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
	// u
	for (size_t i = 1; i <= Nx; ++i)
	{
		u(i, 1) = 0.0;
		u(i, Ny + 1) = u0;
	}
	for (size_t j = 2; j <= Ny; ++j)
	{
		u(1, j) = 0.0;
		u(Nx, j) = 0.0;
	}

	// v
	for (size_t i = 1; i <= Nx + 1; ++i)
	{
		v(i, 1) = 0.0;
		v(i, Ny) = 0.0;
	}
	for (size_t j = 2; j <= Ny - 1; ++j)
	{
		v(1, j) = 0.0;
		v(Nx + 1, j) = 0.0;
	}

	/***************************** Poisson equation ***************************/
	// Coefficient matrix
	vector<Eigen::Triplet<double>> coef;

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

			int id, id_w, id_e, id_n, id_s;
			double rdx = 0.0, rdy = 0.0;
			poisson_stencil(i, j, id, id_w, id_e, id_n, id_s);

			if (i == 2 && j == 2) // Left-Bottom
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_n, c_n);
			}
			else if (i == Nx && j == 2) // Right-Bottom
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_n, c_n);
			}
			else if (i == 2 && j == Ny) // Left-Top
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_s, c_s);
			}
			else if (i == Nx && j == Ny) // Right-Top
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_s, c_s);
			}
			else if (i == 2) // Left
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_n, c_n);
				coef.emplace_back(id, id_s, c_s);
			}
			else if (i == Nx) // Right
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_n, c_n);
				coef.emplace_back(id, id_s, c_s);
			}
			else if (j == 2)	// Bottom
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_n, c_n);
			}
			else if (j == Ny) // Top
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_s, c_s);
			}
			else // Inner
			{
				coef.emplace_back(id, id, a);
				coef.emplace_back(id, id_w, b_w);
				coef.emplace_back(id, id_e, b_e);
				coef.emplace_back(id, id_n, c_n);
				coef.emplace_back(id, id_s, c_s);
			}
		}

	// Construct sparse matrix
	A.setFromTriplets(coef.begin(), coef.end());
	solver.analyzePattern(A);
	solver.factorize(A);
}

void write_user(size_t n)
{
	ofstream fout(f_hist, ios::app);
	if (fout.fail())
		throw runtime_error("Failed to open output file.");

	if (n > 0)
		fout << t << '\t' << max_divergence << endl;

	fout.close();
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
			p_interp(i, j) = interp_f(p(i, j + 1), xP(i), yP(j + 1), p(i + 1, j + 1), xP(i + 1), yP(j + 1), p(i + 1, j), xP(i + 1), yP(j), p(i, j), xP(i), yP(j), x(i), y(j));

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
			fout << setw(WIDTH) << setprecision(DIGITS) << p_interp(i, j) - p0;
			fout << setw(WIDTH) << setprecision(DIGITS) << u_interp(i, j);
			fout << setw(WIDTH) << setprecision(DIGITS) << v_interp(i, j);
			fout << endl;
		}

	// Finalize
	fout.close();
}

void output()
{
	if (!(iter % TECPLOT_GAP))
		write_tecplot(iter);

	write_user(iter);
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
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			if (i == 2 || i == Nx - 1)
				dudx(i, j) = df(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));
			else
			{
				dudx(i, j) = df(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));
				//dudx(i, j) = df_upwind(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1), u(i, j));
				//dudx(i, j) = df_upwind2(u(i - 2, j), u(i - 1, j), u(i, j), u(i + 1, j), u(i + 2, j), xU(i - 2), xU(i - 1), xU(i), xU(i + 1), xU(i + 2), u(i, j));
			}

			dduddx(i, j) = ddf(u(i - 1, j), u(i, j), u(i + 1, j), xU(i - 1), xU(i), xU(i + 1));

			if (j == 2 || j == Ny)
				dudy(i, j) = df(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
			else
			{
				dudy(i, j) = df(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
				//dudy(i, j) = df_upwind(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1), v_bar(i, j));
				//dudy(i, j) = df_upwind2(u(i, j - 2), u(i, j - 1), u(i, j), u(i, j + 1), u(i, j + 2), yU(j - 2), yU(j - 1), yU(j), yU(j + 1), yU(j + 2), v_bar(i, j));
			}

			dduddy(i, j) = ddf(u(i, j - 1), u(i, j), u(i, j + 1), yU(j - 1), yU(j), yU(j + 1));
		}

	Array2D dvdx(Nx + 1, Ny, 0.0), ddvddx(Nx + 1, Ny, 0.0);
	Array2D dvdy(Nx + 1, Ny, 0.0), ddvddy(Nx + 1, Ny, 0.0);
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			if (i == 2 || i == Nx)
				dvdx(i, j) = df(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));
			else
			{
				dvdx(i, j) = df(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));
				//dvdx(i, j) = df_upwind(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1), u_bar(i, j));
				//dvdx(i, j) = df_upwind2(v(i - 2, j), v(i - 1, j), v(i, j), v(i + 1, j), v(i + 2, j), xV(i - 2), xV(i - 1), xV(i), xV(i + 1), xV(i + 2), u_bar(i, j));
			}

			ddvddx(i, j) = ddf(v(i - 1, j), v(i, j), v(i + 1, j), xV(i - 1), xV(i), xV(i + 1));

			if (j == 2 || j == Ny - 1)
				dvdy(i, j) = df(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
			else
			{
				dvdy(i, j) = df(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
				//dvdy(i, j) = df_upwind(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1), v(i, j));
				//dvdy(i, j) = df_upwind2(v(i, j - 2), v(i, j - 1), v(i, j), v(i, j + 1), v(i, j + 2), yV(j - 2), yV(j - 1), yV(j), yV(j + 1), yV(j + 2), v(i, j));
			}

			ddvddy(i, j) = ddf(v(i, j - 1), v(i, j), v(i, j + 1), yV(j - 1), yV(j), yV(j + 1));
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

	/***************************** Poisson Equation ***************************/
	// RHS
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			int id, id_w, id_e, id_n, id_s;
			poisson_stencil(i, j, id, id_w, id_e, id_n, id_s);

			const double dusdx = (u_star(i, j) - u_star(i - 1, j)) / (xU(i) - xU(i - 1));
			const double dvsdy = (v_star(i, j) - v_star(i, j - 1)) / (yV(j) - yV(j - 1));
			const double divergence = dusdx + dvsdy;
			const double p_rhs = rho / dt * divergence;
			rhs(id) = p_rhs;

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

			if (i == 2)
				rhs(id) -= b_w * p(i - 1, j);
			else if (i == Nx)
				rhs(id) -= b_e * p(i + 1, j);
			else
				rhs(id) -= 0.0;

			if (j == 2)
				rhs(id) -= c_s * p(i, j - 1);
			else if (j == Ny)
				rhs(id) -= c_n * p(i, j + 1);
			else
				rhs(id) -= 0.0;
		}

	// Solve the linear system: Ax = rhs
	Eigen::VectorXd res = solver.solve(rhs);

	// Update p
	// Inner
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			auto id = stencil_center_idx(i, j);
			p(i, j) = res(id);
		}

	// Left
	double loc_dx = xP(2) - xP(1);
	for (size_t j = 2; j <= Ny; ++j)
	{
		const auto loc_F2 = nu * df_right2(u(1, j), u(2, j), u(3, j), xU(1), xU(2), xU(3));
		const auto dpdn = rho * loc_F2;
		p(1, j) = p(2, j) - loc_dx * dpdn;
	}

	// Right
	loc_dx = xP(Nx + 1) - xP(Nx);
	for (size_t j = 2; j <= Ny; ++j)
	{
		const auto loc_F2 = nu * df_left2(u(Nx - 2, j), u(Nx - 1, j), u(Nx, j), xU(Nx - 2), xU(Nx - 1), xU(Nx));
		const auto dpdn = rho * loc_F2 * -1.0;
		p(Nx + 1, j) = p(Nx, j) + loc_dx * dpdn;
	}

	// Bottom
	double loc_dy = yP(2) - yP(1);
	for (size_t i = 2; i <= Nx; ++i)
	{
		const auto loc_F3 = nu * df_right2(v(i, 1), v(i, 2), v(i, 3), yV(1), yV(2), yV(3));
		const auto dpdn = rho * loc_F3;
		p(i, 1) = p(i, 2) - loc_dy * dpdn;
	}

	// Top
	for (size_t i = 2; i <= Nx; ++i)
	{
		const auto loc_F3 = nu * df_left2(v(i, Ny - 2), v(i, Ny - 1), v(i, Ny), yV(Ny - 2), yV(Ny - 1), yV(Ny));
		const auto dpdn = rho * loc_F3 * -1.0;
		p(i, Ny + 1) = p(i, Ny) + loc_dy * dpdn;
	}

	// 4 corners
	p(1, 1) = relaxation(p(1, 2), p(2, 1), 0.5);
	p(Nx + 1, 1) = relaxation(p(Nx + 1, 2), p(Nx, 1), 0.5);
	p(1, Ny + 1) = relaxation(p(1, Ny), p(2, Ny + 1), 0.5);
	p(Nx + 1, Ny + 1) = relaxation(p(Nx + 1, Ny), p(Nx, Ny + 1), 0.5);

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
}

bool checkConvergence()
{
	size_t i_max, j_max;

	// inf-norm
	double u_res = 0.0;
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx - 1; ++i)
		{
			auto du = abs(u_prime(i, j));
			if (du > u_res)
			{
				u_res = du;
				i_max = i;
				j_max = j;
			}
		}
	cout << "\tMax |u'|=" << u_res << ", at (" << i_max << ", " << j_max << ")" << endl;

	double v_res = 0.0;
	for (size_t j = 2; j <= Ny - 1; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			auto dv = abs(v_prime(i, j));
			if (dv > v_res)
			{
				v_res = dv;
				i_max = i;
				j_max = j;
			}
		}
	cout << "\tMax |v'|=" << v_res << ", at (" << i_max << ", " << j_max << ")" << endl;

	// divergence
	for (size_t j = 2; j <= Ny; ++j)
		for (size_t i = 2; i <= Nx; ++i)
		{
			double loc_div = (u(i, j) - u(i - 1, j)) / (xU(i) - xU(i - 1)) + (v(i, j) - v(i, j - 1)) / (yV(j) - yV(j - 1));
			loc_div = abs(loc_div);
			if (loc_div > max_divergence)
			{
				max_divergence = loc_div;
				i_max = i;
				j_max = j;
			}
		}
	cout << "\tMax divergence=" << max_divergence << " at: (" << i_max << ", " << j_max << ")" << endl;

	return max_divergence < 1e-3;
}

void solve()
{
	bool ok = false;
	while (!ok)
	{
		cout << "Iter" << ++iter << ":" << endl;
		dt = TimeStep();
		cout << "\tt=" << t << "s, dt=" << dt << "s" << endl;
		max_divergence = 0.0;
		ProjectionMethod();
		t += dt;
		ok = checkConvergence();
		output();
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
