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
const double dxdx = pow(dx, 2), dydy = pow(dy, 2);
const double dx2 = 2 * dx, dy2 = 2 * dy;

// Flow param
const double Re = 100.0;
const double rho = 1.0; // kg/m^3
const double p0 = 101325.0; // Pa
const double u0 = 0.1; // m/s
const double v0 = 0.0; // m/s
const double nu = u0 * max(Lx, Ly) / Re; // m^2 / s
const double mu = rho * nu; // Pa*s

// Timing
double dt = 0.01; // s
double t = 0.0; // s
size_t iter = 0;
const size_t MAX_ITER = 2000;

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


template<typename T>
inline T relaxation(const T &a, const T &b, double alpha)
{
	return (1 - alpha) *a + alpha * b;
}

void BC()
{
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
	BC();
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

static void l2g(int loc_i, int loc_j, int &glb_i, int &glb_j)
{
	glb_i = loc_i + 1;
	glb_j = loc_j + 1;
}

static void g2l(int glb_i, int glb_j, int &loc_i, int &loc_j)
{
	loc_i = glb_i - 1;
	loc_j = glb_j - 1;
}

static int loc_idx(int loc_i, int loc_j)
{
	return loc_j * (Nx - 2) + loc_i;
}

static bool onBdy(int glb_i, int glb_j)
{
	return glb_i == 0 || glb_i == Nx - 1 || glb_j == 0 || glb_j == Ny - 1;
}

static void idx_around(int glb_i, int glb_j, int *ax, int *ay, int *ai, bool *bdyFlag)
{
	ax[0] = glb_i;
	ay[0] = glb_j;

	ax[1] = glb_i + 1;
	ay[1] = glb_j;

	ax[2] = glb_i;
	ay[2] = glb_j + 1;

	ax[3] = glb_i - 1;
	ay[3] = glb_j;

	ax[4] = glb_i;
	ay[4] = glb_j - 1;

	for (auto i = 0; i < 5; ++i)
	{
		int loc_i, loc_j;
		g2l(ax[i], ay[i], loc_i, loc_j);
		ai[i] = loc_idx(loc_i, loc_j);
		bdyFlag[i] = onBdy(ax[i], ay[i]);
	}
}

void solvePoissonEquation()
{
	cout << "\tSolve the Possion equation..." << endl;
	const int NumOfUnknown = (Nx - 2) * (Ny - 2);

	// Compute d
	VectorXd b(NumOfUnknown);
	b.setZero();
	int cnt = 0;
	for (int j = 1; j < Ny - 1; ++j)
		for (int i = 1; i < Nx - 1; ++i)
		{
			double drusdx = (rho*u_star[i][j] - rho * u_star[i - 1][j]) / dx;
			double drvsdy = (rho*v_star[i][j] - rho * v_star[i][j - 1]) / dy;
			p_prime[i][j] = drusdx + drvsdy;
			b[cnt++] = -p_prime[i][j];
		}

	// Compute coefficient matrix
	vector<Triplet<double>> elem;
	for (int j = 1; j < Ny - 1; ++j)
		for (int i = 1; i < Nx - 1; ++i)
		{
			int idx[5], ii[5], jj[5];
			bool flag[5];
			idx_around(i, j, ii, jj, idx, flag);

			double v[5] = { a, 0.0, 0.0, 0.0, 0.0 };

			// E
			if (flag[1])
				v[0] += b;
			else
				v[1] += b;

			// N
			if (!flag[2])
				v[2] += c;

			// W
			if (flag[3])
				v[0] += b;
			else
				v[3] += b;

			// S
			if (flag[4])
				v[0] += c;
			else
				v[4] += c;

			for (auto c = 0; c < 5; ++c)
				if (!flag[c])
					elem.push_back(Triplet<double>(idx[c], idx[c], v[c]));
		}

	SparseMatrix<double> A(NumOfUnknown, NumOfUnknown);
	A.setZero();
	A.setFromTriplets(elem.begin(), elem.end());

	// Take Cholesky decomposition of A
	SimplicialCholesky<SparseMatrix<double>> chol(A);

	// Solve
	VectorXd x = chol.solve(b);

	ofstream fout("A.txt");
	fout << A;
	fout.close();
	fout.open("b.txt");
	fout << b;
	fout.close();
	fout.open("x.txt");
	fout << x;
	fout.close();

	cnt = 0;
	for (int i = 1; i < Nx - 1; ++i)
		for (int j = 1; j < Ny - 1; ++j)
			p_prime[i][j] = x[cnt++];

	for (int j = 1; j < Ny - 1; ++j)
		for (int i = 0; i < Nx - 1; ++i)
			u_prime[i][j] = -dt / dx * (p_prime[i + 1][j] - p_prime[i][j]) / rho;

	for (int i = 1; i < Nx - 1; ++i)
		for (int j = 0; j < Ny - 1; ++j)
			v_prime[i][j] = -dt / dy * (p_prime[i][j + 1] - p_prime[i][j]) / rho;

	output("Cavity_prime.dat", u_prime, v_prime, p_prime);
}

// Explicit time-marching
void ProjectionMethod()
{
	cout << "\tInit star value from previous calculation..." << endl;
	for (int i = 0; i < Nx; ++i)
		for (int j = 0; j < Ny; ++j)
			p_star[i][j] = p[i][j];

	for (int i = 0; i < Nx - 1; ++i)
		for (int j = 0; j < Ny; ++j)
			u_star[i][j] = u[i][j];

	for (int i = 0; i < Nx; ++i)
		for (int j = 0; j < Ny - 1; ++j)
			v_star[i][j] = v[i][j];

	cout << "\tCalculate new star value..." << endl;
	double v_a, v_b, dru2dx, druvdy, dduddx, dduddy, A_star, dpdx;
	for (int j = 1; j < Ny - 1; ++j)
		for (int i = 0; i < Nx - 1; ++i)
		{
			v_a = relaxation(v[i][j], v[i + 1][j], 0.5);
			v_b = relaxation(v[i][j - 1], v[i + 1][j - 1], 0.5);
			if (i == 0)
			{
				dru2dx = (rho * pow(u[i + 1][j], 2) + 3 * rho * pow(u[i][j], 2)) / (3 * dx);
				dduddx = (u[i + 1][j] - 3 * u[i][j]) / (0.75 * dx2);
			}
			else if (i == Nx - 2)
			{
				dru2dx = (-rho * pow(u[i - 1][j], 2) - 3 * rho * pow(u[i][j], 2)) / (3 * dx);
				dduddx = (u[i - 1][j] - 3 * u[i][j]) / (0.75 * dx2);
			}
			else
			{
				dru2dx = (rho * pow(u[i + 1][j], 2) - rho * pow(u[i - 1][j], 2)) / (2 * dx);
				dduddx = (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / dx2;
			}
			druvdy = 0.5*(rho * u[i][j + 1] * v_a - rho * u[i][j - 1] * v_b) / dy;
			dduddy = (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / dy2;
			A_star = -(dru2dx + druvdy) + mu * (dduddx + dduddy);
			dpdx = (p[i + 1][j] - p[i][j]) / dx;
			u_star[i][j] += dt * (A_star - dpdx) / rho;
		}

	double u_c, u_d, drvudx, drv2dy, ddvddx, ddvddy, B_star, dpdy;
	for (int i = 1; i < Nx - 1; ++i)
		for (int j = 0; j < Ny - 1; ++j)
		{
			u_c = relaxation(u[i - 1][j], u[i - 1][j + 1], 0.5);
			u_d = relaxation(u[i][j], u[i][j + 1], 0.5);
			if (j == 0)
			{
				drv2dy = (rho * pow(v[i][j + 1], 2) + 3 * rho * pow(v[i][j], 2)) / (3 * dy);
				ddvddy = (v[i][j + 1] - 3 * v[i][j]) / (0.75 * dy2);
			}
			else if (j == Ny - 2)
			{
				drv2dy = (-rho * pow(v[i][j - 1], 2) - 3 * rho * pow(v[i][j], 2)) / (3 * dy);
				ddvddy = (v[i][j - 1] - 3 * v[i][j]) / (0.75 * dy2);
			}
			else
			{
				drv2dy = 0.5*(rho * pow(v[i][j + 1], 2) - rho * pow(v[i][j - 1], 2)) / dy;
				ddvddy = (v[i][j + 1] - 2 * v[i][j] + v[i][j - 1]) / dy2;
			}
			drvudx = 0.5*(rho * v[i + 1][j] * u_d - rho * v[i - 1][j] * u_c) / dx;
			ddvddx = (v[i + 1][j] - 2 * v[i][j] + v[i - 1][j]) / dx2;
			B_star = -(drvudx + drv2dy) + mu * (ddvddx + ddvddy);
			dpdy = (p[i][j + 1] - p[i][j]) / dy;
			v_star[i][j] += dt * (B_star - dpdy) / rho;
		}

	solvePoissonEquation();

	for (int i = 1; i < Nx - 1; ++i)
		for (int j = 1; j < Ny - 1; ++j)
			p[i][j] += alpha_p * p_prime[i][j];

	for (int j = 1; j < Ny - 1; ++j)
		for (int i = 0; i < Nx - 1; ++i)
			u[i][j] += u_prime[i][j];

	for (int i = 1; i < Nx - 1; ++i)
		for (int j = 0; j < Ny - 1; ++j)
			v[i][j] += v_prime[i][j];
}

bool checkConvergence()
{
	double rsd = 0.0;
	for (int p = 0; p < NumOfUnknown; ++p)
	{
		double tmp = fabs(b[p]);
		if (tmp > rsd)
			rsd = tmp;
	}
	cout << "\trsd=" << rsd << endl;

	return rsd < 1e-4 || iter > MAX_ITER;
}

void solve(void)
{
	bool ok = false;
	int iter = 0;
	while (!ok)
	{
		cout << "Iter" << ++iter << ":" << endl;
		dt = TimeStep();
		cout << "\tt=" << t << "s, dt=" << dt << "s" << endl;
		ProjectionMethod();
		t += dt;
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
