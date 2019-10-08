#ifndef __GEOM_ENTITY_HPP__
#define __GEOM_ENTITY_HPP__

#include <utility>
#include "natural_array.hpp"
#include "math_type.hpp"

class GeneralPhysicalVar
{
private:
	Vector c, vel;
	Scalar p, rho, T;
	Vector drho, dp, dT;
	Tensor dvel;

	void exchange(GeneralPhysicalVar &rhs)
	{
		std::swap(c, rhs.c);
		std::swap(vel, rhs.vel);
		std::swap(p, rhs.p);
		std::swap(rho, rhs.rho);
		std::swap(T, rhs.T);
		std::swap(drho, rhs.drho);
		std::swap(dp, rhs.dp);
		std::swap(dT, rhs.dT);
		std::swap(dvel, rhs.dvel);
	}

public:
	GeneralPhysicalVar() = default;

	GeneralPhysicalVar(const GeneralPhysicalVar &rhs) = default;

	~GeneralPhysicalVar() = default;

	GeneralPhysicalVar &operator=(GeneralPhysicalVar rhs)
	{
		exchange(rhs);
		return *this;
	}

	Vector &position() { return c; };

	Scalar &density() { return rho; }
	Vector &velocity() { return vel; };
	Scalar &pressure() { return p; }
	Scalar &temperature() { return T; }

	Vector &density_gradient() { return drho; }
	Tensor &velocity_gradient() { return dvel; }
	Vector &pressure_gradient() { return dp; }
	Vector &temperature_gradient() { return dT; }
};

class Point : public GeneralPhysicalVar
{
public:
	Point() : GeneralPhysicalVar() {}

	~Point() = default;
};

class Face : public GeneralPhysicalVar
{
public:
	Face() : GeneralPhysicalVar() {}

	~Face() = default;

	int shape() const { return vertex.size(); }

private:
	Array1D<size_t> vertex; // Contents are 1-based pnt index.
};

class Cell : public GeneralPhysicalVar
{
public:
	Cell() : GeneralPhysicalVar() {}

	~Cell() = default;

	int shape() const { return vertex.size(); }

private:
	Array1D<size_t> vertex; // Contents are 1-based pnt index.
};

#endif
