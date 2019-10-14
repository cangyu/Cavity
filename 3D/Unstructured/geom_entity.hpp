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

protected:
	Vector &position() { return c; };

public:
	GeneralPhysicalVar() = default;

	GeneralPhysicalVar(const GeneralPhysicalVar &rhs) = default;

	~GeneralPhysicalVar() = default;

	GeneralPhysicalVar &operator=(GeneralPhysicalVar rhs)
	{
		exchange(rhs);
		return *this;
	}

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

	Vector &coordinate() { return position(); }
};

class Face : public GeneralPhysicalVar
{
public:
	Face() : GeneralPhysicalVar() {}

	~Face() = default;

	Vector &center() { return position(); }

	size_t shape() const { return m_vertex.size(); }

	Scalar &area() { return m_area; }
 
private:
	Array1D<size_t> m_vertex; // Contents are 1-based pnt index.
	Scalar m_area;

};

class Cell : public GeneralPhysicalVar
{
public:
	Cell() : GeneralPhysicalVar() {}

	~Cell() = default;

	Vector &center() { return position(); }

	size_t shape() const { return vertex.size(); }

private:
	Array1D<size_t> vertex; // Contents are 1-based pnt index.
};

class Patch
{
private:
	std::string m_name;
	int m_bc;
	Array1D<size_t> m_faceIncluded;

public:
	Patch() : m_bc(-1) {}

	std::string &name() { return m_name; }

	int &BC() { return m_bc; }

	size_t &face(size_t idx) { return m_faceIncluded(idx); }

	void resize(size_t n) { m_faceIncluded.resize(n); }
};

#endif
