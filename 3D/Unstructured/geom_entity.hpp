#ifndef __GEOM_ENTITY_HPP__
#define __GEOM_ENTITY_HPP__

#include <string>
#include "natural_array.hpp"
#include "math_type.hpp"

struct Point
{
	Vector coordinate;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Face
{
	Vector center;
	Scalar area;
	Array1D<Point*> vertex;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Cell
{
	Vector center;
	Array1D<Point*> vertex;
	Array1D<Face*> surface;

	Scalar density;
	Vector velocity;
	Scalar pressure;
	Scalar temperature;

	Vector density_gradient;
	Tensor velocity_gradient;
	Vector pressure_gradient;
	Vector temperature_gradient;
};

struct Patch
{
	std::string name;
	int BC;
	Array1D<Face*> surface;
};

#endif
