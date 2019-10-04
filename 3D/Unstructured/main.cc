#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf_msh.h"

using namespace std;

inline double fder1(double fl, double fm, double fr, double dxl, double dxr)
{
	const double dxrl = dxr / dxl;
	const double dxlr = dxl / dxr;
	return (dxlr*fr - dxrl * fl - (dxlr - dxrl)*fm) / (dxl + dxr);
}

inline double fder2(double fl, double fm, double fr, double dxl, double dxr)
{
	const double inv_dxl = 1.0 / dxl;
	const double inv_dxr = 1.0 / dxr;
	return 2.0 / (dxl + dxr) * (fl * inv_dxl + fr * inv_dxr - fm * (inv_dxl + inv_dxr));
}

const int NDIM = 2;
const int X_DIM = 0;
const int Y_DIM = 1;

const double g = 9.80665; // m/s^2
const double T0 = 300.0; // K
const double P0 = 101325.0; // Pa
const double rho_h2o = 0.996873e3; //kg/m^3
const double U0 = 5.0; // m/s
const double nu = 1e-5; //m^2/s
const vector<double> f{ 0.0, -g };

const int NumOfIter = 10;

class Point
{
private:
	vector<double> c;
	vector<double> vel;
	double p, rho;

public:
	vector<double> V_star, N, F;
	vector<double> grad_p, grad_u, grad_v, vel_laplace;
	double div_V_star, dp;
	vector<double> grad_dp;

public:
	Point(double x = 0.0, double y = 0.0) :
		c{ x, y },
		vel(NDIM, 0.0),
		p(101325.0),
		rho(1.0),
		V_star(NDIM, 0.0),
		N(NDIM, 0.0),
		F(NDIM, 0.0),
		grad_p(NDIM, 0.0),
		grad_u(NDIM, 0.0),
		grad_v(NDIM, 0.0),
		vel_laplace(NDIM, 0.0),
		div_V_star(0.0),
		dp(0.0),
		grad_dp(NDIM, 0.0)
	{
	}

	~Point() = default;

	double x(void) const
	{
		return c[X_DIM];
	}

	double& x(void)
	{
		return c[X_DIM];
	}

	double y(void) const
	{
		return c[Y_DIM];
	}

	double& y(void)
	{
		return c[Y_DIM];
	}

	double U(void) const
	{
		return vel[X_DIM];
	}

	double& U(void)
	{
		return vel[X_DIM];
	}

	double V(void) const
	{
		return vel[Y_DIM];
	}

	double& V(void)
	{
		return vel[Y_DIM];
	}

	double pressure(void) const
	{
		return p;
	}

	double& pressure(void)
	{
		return p;
	}

	double density(void) const
	{
		return rho;
	}

	double& density(void)
	{
		return rho;
	}
};

class Block
{
private:
	vector<int> dim;
	vector<vector<Point>> all_pnt;
	vector<vector<double>> coef;
	vector<double> rhs;

public:
	Block(int nx = 0, int ny = 0) :
		dim{ nx, ny },
		all_pnt(ny, vector<Point>(nx)),
		coef(nx*ny - 4, vector<double>(nx*ny - 4, 0.0)),
		rhs(nx*ny - 4, 0.0)
	{
	}

	~Block() = default;

	int x_pnt_num(void) const
	{
		return dim[X_DIM];
	}

	int y_pnt_num(void) const
	{
		return dim[Y_DIM];
	}

	Point& pnt(int i, int j)
	{
		while (i < 0)
			i += x_pnt_num();
		while (i > x_pnt_num())
			i -= x_pnt_num();

		while (j < 0)
			j += y_pnt_num();
		while (j > y_pnt_num())
			j -= y_pnt_num();

		return all_pnt[j][i];
	}

	void do_predict_step(double dt)
	{
		Point *p = nullptr;

		calc_gradients();
		calc_convection_term();

		//Calculate the combined term: F
		for (int j = 0; j < y_pnt_num(); ++j)
			for (int i = 0; i < x_pnt_num(); ++i)
			{
				p = &pnt(i, j);
				p->F[X_DIM] = -p->N[X_DIM] + nu * p->vel_laplace[X_DIM] + f[X_DIM];
				p->F[Y_DIM] = -p->N[Y_DIM] + nu * p->vel_laplace[Y_DIM] + f[Y_DIM];
			}

		//Calculate the predicted velocity
		for (int j = 0; j < y_pnt_num(); ++j)
			for (int i = 0; i < x_pnt_num(); ++i)
			{
				p = &pnt(i, j);
				p->V_star[X_DIM] = p->U() + dt * p->F[X_DIM] - dt / rho_h2o * p->grad_p[X_DIM];
				p->V_star[Y_DIM] = p->V() + dt * p->F[Y_DIM] - dt / rho_h2o * p->grad_p[Y_DIM];
			}
	}

	void solve_pressure_poisson_equation(double dt)
	{
		Point *pl = nullptr, *pm = nullptr, *pr = nullptr;
		double dl = 0.0, dr = 0.0;

		/*Calculate divergence*/
		//X-direction
		for (int j = 0; j < y_pnt_num(); ++j)
		{
			pm = &pnt(0, j);
			pr = &pnt(1, j);
			dr = pr->x() - pm->x();
			pm->div_V_star = (pr->V_star[X_DIM] - pm->V_star[X_DIM]) / dr;

			for (int i = 1; i < x_pnt_num() - 1; ++i)
			{
				pl = &pnt(i - 1, j);
				pm = &pnt(i, j);
				pr = &pnt(i + 1, j);

				//Upwind biased
				if (pm->U() > 0)
				{
					dl = pm->x() - pl->x();
					pm->div_V_star = (pm->V_star[X_DIM] - pl->V_star[X_DIM]) / dl;
				}
				else
				{
					dr = pr->x() - pm->x();
					pm->div_V_star = (pr->V_star[X_DIM] - pm->V_star[X_DIM]) / dr;
				}
			}

			pl = &pnt(-2, j);
			pm = &pnt(-1, j);
			dl = pm->x() - pl->x();
			pm->div_V_star = (pm->V_star[X_DIM] - pl->V_star[X_DIM]) / dl;
		}

		//Y-direction
		for (int i = 0; i < x_pnt_num(); ++i)
		{
			pm = &pnt(i, 0);
			pr = &pnt(i, 1);
			dr = pr->y() - pm->y();
			pm->div_V_star += (pr->V_star[Y_DIM] - pm->V_star[Y_DIM]) / dr;

			for (int j = 1; j < y_pnt_num() - 1; ++j)
			{
				pl = &pnt(i, j - 1);
				pm = &pnt(i, j);
				pr = &pnt(i, j + 1);

				//Upwind biased
				if (pm->V() > 0)
				{
					dl = pm->y() - pl->y();
					pm->div_V_star += (pm->V_star[Y_DIM] - pl->V_star[Y_DIM]) / dl;
				}
				else
				{
					dr = pr->y() - pm->y();
					pm->div_V_star += (pr->V_star[Y_DIM] - pm->V_star[Y_DIM]) / dr;
				}
			}

			pl = &pnt(i, -2);
			pm = &pnt(i, -1);
			dl = pm->y() - pl->y();
			pm->div_V_star += (pm->V_star[Y_DIM] - pl->V_star[Y_DIM]) / dl;
		}

		/*Calculate coefficient matrix*/
		//TODO

		/*Solve pressure correction*/
		//TODO
	}

	void do_correct_step(double dt)
	{
		calc_dp_gradient();

		Point *p = nullptr;
		for (int j = 0; j < y_pnt_num(); ++j)
			for (int i = 0; i < x_pnt_num(); ++i)
			{
				p = &pnt(i, j);
				p->U() = p->V_star[X_DIM] - dt / rho_h2o * p->grad_dp[X_DIM];
				p->V() = p->V_star[Y_DIM] - dt / rho_h2o * p->grad_dp[Y_DIM];
			}
	}

private:
	void calc_upwind_der1()
	{
		Point *pl = nullptr, *pm = nullptr, *pr = nullptr;
		double dl, dr;

		/*x-direction*/
		for (int j = 0; j < y_pnt_num(); ++j)
		{
			pm = &pnt(0, j);
			pr = &pnt(1, j);
			dr = pr->x() - pm->x();
			pm->grad_p[X_DIM] = (pr->pressure() - pm->pressure()) / dr;
			pm->grad_u[X_DIM] = (pr->U() - pm->U()) / dr;
			pm->grad_v[X_DIM] = (pr->V() - pm->V()) / dr;

			for (int i = 1; i < x_pnt_num() - 1; ++i)
			{
				pl = &pnt(i - 1, j);
				pm = &pnt(i, j);
				pr = &pnt(i + 1, j);

				//Upwind biased
				if (pm->U() > 0)
				{
					dl = pm->x() - pl->x();
					pm->grad_p[X_DIM] = (pm->pressure() - pl->pressure()) / dl;
					pm->grad_u[X_DIM] = (pm->U() - pl->U()) / dl;
					pm->grad_v[X_DIM] = (pm->V() - pl->V()) / dl;
				}
				else
				{
					dr = pr->x() - pm->x();
					pm->grad_p[X_DIM] = (pr->pressure() - pm->pressure()) / dr;
					pm->grad_u[X_DIM] = (pr->U() - pm->U()) / dr;
					pm->grad_v[X_DIM] = (pr->V() - pm->V()) / dr;
				}
			}

			pl = &pnt(-2, j);
			pm = &pnt(-1, j);
			dl = pm->x() - pr->x();
			pm->grad_p[X_DIM] = (pm->pressure() - pl->pressure()) / dl;
			pm->grad_u[X_DIM] = (pm->U() - pl->U()) / dl;
			pm->grad_v[X_DIM] = (pm->V() - pl->V()) / dl;
		}

		/*y-direction*/
		for (int i = 0; i < x_pnt_num(); ++i)
		{
			pm = &pnt(i, 0);
			pr = &pnt(i, 1);
			dr = pr->y() - pm->y();
			pm->grad_p[Y_DIM] = (pr->pressure() - pm->pressure()) / dr;
			pm->grad_u[Y_DIM] = (pr->U() - pm->U()) / dr;
			pm->grad_v[Y_DIM] = (pr->V() - pm->V()) / dr;

			for (int j = 1; j < y_pnt_num() - 1; ++j)
			{
				pl = &pnt(i, j - 1);
				pm = &pnt(i, j);
				pr = &pnt(i, j + 1);

				//Upwind biased
				if (pm->V() > 0)
				{
					dl = pm->y() - pl->y();
					pm->grad_p[Y_DIM] = (pm->pressure() - pl->pressure()) / dl;
					pm->grad_u[Y_DIM] = (pm->U() - pl->U()) / dl;
					pm->grad_v[Y_DIM] = (pm->V() - pl->V()) / dl;
				}
				else
				{
					dr = pr->y() - pm->y();
					pm->grad_p[Y_DIM] = (pr->pressure() - pm->pressure()) / dr;
					pm->grad_u[Y_DIM] = (pr->U() - pm->U()) / dr;
					pm->grad_v[Y_DIM] = (pr->V() - pm->V()) / dr;
				}
			}

			pl = &pnt(i, -2);
			pm = &pnt(i, -1);
			dl = pm->x() - pl->x();
			pm->grad_p[Y_DIM] = (pm->pressure() - pl->pressure()) / dl;
			pm->grad_u[Y_DIM] = (pm->U() - pl->U()) / dl;
			pm->grad_v[Y_DIM] = (pm->V() - pl->V()) / dl;
		}
	}

	void calc_vel_laplace()
	{
		Point *pl = nullptr, *pm = nullptr, *pr = nullptr;
		double dl, dr;

		/*x-direction*/
		for (int j = 0; j < y_pnt_num(); ++j)
		{
			pm = &pnt(0, j);
			pr = &pnt(1, j);
			dr = pr->x() - pm->x();
			pm->vel_laplace[X_DIM] = (pr->grad_u[X_DIM] - pm->grad_u[X_DIM]) / dr;
			pm->vel_laplace[Y_DIM] = (pr->grad_v[X_DIM] - pm->grad_v[X_DIM]) / dr;

			for (int i = 1; i < x_pnt_num() - 1; ++i)
			{
				pl = &pnt(i - 1, j);
				pm = &pnt(i, j);
				pr = &pnt(i + 1, j);

				//Central difference
				dr = pr->x() - pm->x();
				dl = pm->x() - pl->x();
				pm->vel_laplace[X_DIM] = fder2(pl->U(), pm->U(), pr->U(), dl, dr);
				pm->vel_laplace[Y_DIM] = fder2(pl->V(), pm->V(), pr->V(), dl, dr);
			}

			pl = &pnt(-2, j);
			pm = &pnt(-1, j);
			dl = pm->x() - pr->x();
			pm->vel_laplace[X_DIM] = (pm->grad_u[X_DIM] - pl->grad_u[X_DIM]) / dl;
			pm->vel_laplace[Y_DIM] = (pm->grad_v[X_DIM] - pl->grad_v[X_DIM]) / dl;
		}

		/*y-direction*/
		for (int i = 0; i < x_pnt_num(); ++i)
		{
			pm = &pnt(i, 0);
			pr = &pnt(i, 1);
			dr = pr->y() - pm->y();
			pm->vel_laplace[X_DIM] += (pr->grad_u[Y_DIM] - pm->grad_u[Y_DIM]) / dr;
			pm->vel_laplace[Y_DIM] += (pr->grad_v[Y_DIM] - pm->grad_v[Y_DIM]) / dr;

			for (int j = 1; j < y_pnt_num() - 1; ++j)
			{
				pl = &pnt(i, j - 1);
				pm = &pnt(i, j);
				pr = &pnt(i, j + 1);
				dr = pr->y() - pm->y();
				dl = pm->y() - pl->y();
				pm->vel_laplace[X_DIM] += fder2(pl->U(), pm->U(), pr->U(), dl, dr);
				pm->vel_laplace[Y_DIM] += fder2(pl->V(), pm->V(), pr->V(), dl, dr);
			}

			pl = &pnt(i, -2);
			pm = &pnt(i, -1);
			dl = pm->x() - pl->x();
			pm->vel_laplace[X_DIM] += (pm->grad_u[Y_DIM] - pl->grad_u[Y_DIM]) / dl;
			pm->vel_laplace[Y_DIM] += (pm->grad_v[Y_DIM] - pl->grad_v[Y_DIM]) / dl;
		}
	}

	void calc_gradients()
	{
		calc_upwind_der1();
		calc_vel_laplace();
	}

	void calc_convection_term()
	{
		Point *p = nullptr;
		double u = 0.0, v = 0.0;

		for (int j = 0; j < y_pnt_num(); ++j)
			for (int i = 0; i < x_pnt_num(); ++i)
			{
				p = &pnt(i, j);
				u = p->U();
				v = p->V();
				p->N[X_DIM] = u * p->grad_u[X_DIM] + v * p->grad_u[Y_DIM];
				p->N[Y_DIM] = u * p->grad_v[X_DIM] + v * p->grad_v[Y_DIM];
			}
	}

	void calc_dp_gradient()
	{
		Point *pl = nullptr, *pm = nullptr, *pr = nullptr;
		double dl = 0.0, dr = 0.0;

		//X-direction
		for (int j = 0; j < y_pnt_num(); ++j)
		{
			pm = &pnt(0, j);
			pr = &pnt(1, j);
			dr = pr->x() - pm->x();
			pm->grad_dp[X_DIM] = (pr->dp - pm->dp) / dr;

			for (int i = 1; i < x_pnt_num() - 1; ++i)
			{
				pl = &pnt(i - 1, j);
				pm = &pnt(i, j);
				pr = &pnt(i + 1, j);

				//Upwind
				if (pm->U() > 0)
				{
					dl = pm->x() - pl->x();
					pm->grad_dp[X_DIM] = (pm->dp - pl->dp) / dl;
				}
				else
				{
					dr = pr->x() - pm->x();
					pm->grad_dp[X_DIM] = (pr->dp - pm->dp) / dr;
				}
			}

			pl = &pnt(-2, j);
			pm = &pnt(-1, j);
			dl = pm->x() - pl->x();
			pm->grad_dp[X_DIM] = (pm->dp - pl->dp) / dl;
		}

		//Y-direction
		for (int i = 0; i < x_pnt_num(); ++i)
		{
			pm = &pnt(i, 0);
			pr = &pnt(i, 1);
			dr = pr->y() - pm->y();
			pm->grad_dp[Y_DIM] = (pr->dp - pm->dp) / dr;

			for (int j = 1; j < y_pnt_num() - 1; ++j)
			{
				pl = &pnt(i, j - 1);
				pm = &pnt(i, j);
				pr = &pnt(i, j + 1);

				//Upwind
				if (pm->V() > 0)
				{
					dl = pm->y() - pl->y();
					pm->grad_dp[Y_DIM] = (pm->dp - pl->dp) / dl;
				}
				else
				{
					dr = pr->y() - pm->y();
					pm->grad_dp[Y_DIM] = (pr->dp - pm->dp) / dr;
				}
			}

			pl = &pnt(i, -2);
			pm = &pnt(i, -1);
			dl = pm->y() - pl->y();
			pm->grad_dp[Y_DIM] = (pm->dp - pl->dp) / dl;
		}
	}

	int cord2idx(int i, int j)
	{
		return 0;
	}

	void idx2cord(int idx, int &i, int &j)
	{

	}
};

int main(int argc, char *argv[])
{
	//Read grid
	XF_MSH mesh;
	mesh.readFromFile("grid/fluent.msh");
	mesh.writeToFile("grid/blessed.msh");

	return 0;
}
