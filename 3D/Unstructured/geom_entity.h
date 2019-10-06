#ifndef __GEOM_ENTITY_H__
#define __GEOM_ENTITY_H__

#include "natural_array.h"
#include "math_type.h"

class Point
{
public:
	Vector V_star, N, F, grad_p, grad_u, grad_v, vel_laplace, grad_dp;
	Scalar div_V_star, dp;

public:
	Point(Scalar x = 0.0, Scalar y = 0.0, Scalar z = 0.0) : c(x, y, z), vel(0.0), p(101325.0), rho(1.0), T(300.0) {}

	~Point() = default;

	Vector &position() { return c; };

	Scalar &density() { return rho; }

	Vector &velocity() { return vel; };

	Scalar &pressure() { return p; }

	Scalar &temperature() { return T; }

private:
	Vector c, vel;
	Scalar p, rho, T;
};

class Face
{
public:
	Face() {}

	~Face() = default;

	int shape() const { return vertex.size(); }

	Vector &center() { return c; };

	Scalar &density() { return rho; }

	Vector &velocity() { return vel; };

	Scalar &pressure() { return p; }

	Scalar &temperature() { return T; }

private:
	Array1D<size_t> vertex; // Contents are 1-based pnt index.
	Vector c, vel;
	Scalar p, rho, T;
};

class Cell
{
public:
	Cell() {}

	~Cell() = default;

	int shape() const { return vertex.size(); }

	Vector &center() { return c; };

	Scalar &density() { return rho; }

	Vector &velocity() { return vel; };

	Scalar &pressure() { return p; }

	Scalar &temperature() { return T; }

private:
	Array1D<size_t> vertex; // Contents are 1-based pnt index.
	Vector c, vel;
	Scalar p, rho, T;
};

#endif
