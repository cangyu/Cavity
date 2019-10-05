#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "xf_msh.h"

// Arrays
template<typename T>
class Array1D : public std::vector<T>
{
public:
	Array1D(size_t n = 0) : std::vector<T>(n) {}

	Array1D(size_t n, const T &val) : std::vector<T>(n, val) {}

	~Array1D() = default;

	// 1-based indexing
	T &operator()(size_t i)
	{
		return std::vector<T>::at(i - 1);
	}

	T operator()(size_t i) const
	{
		return std::vector<T>::at(i - 1);
	}
};

template<typename T>
class Array2D
{
private:
	std::vector<T> m_data;
	size_t m_Nx, m_Ny;

public:
	Array2D(size_t nx = 0, size_t ny = 0) : m_Nx(nx), m_Ny(ny), m_data(nx * ny) {}

	Array2D(size_t nx, size_t ny, const T &val) : m_Nx(nx), m_Ny(ny), m_data(nx * ny, val) {}

	~Array2D() = default;

	void resize(size_t nI, size_t nJ)
	{
		m_Nx = nI;
		m_Ny = nJ;
		m_data.resize(nI * nJ);
	}

	// 0-based indexing
	T &at(size_t i, size_t j)
	{
		return m_data[idx(i, j)];
	}

	T at(size_t i, size_t j) const
	{
		return m_data[idx(i, j)];
	}

	// 1-based indexing
	T &operator()(size_t i, size_t j)
	{
		return at(i - 1, j - 1);
	}

	T operator()(size_t i, size_t j) const
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

template<typename T>
class Array3D
{
private:
	std::vector<T> m_data;
	size_t m_Nx, m_Ny, m_Nz;

public:
	Array3D(size_t nx = 0, size_t ny = 0, size_t nz = 0) : m_Nx(nx), m_Ny(ny), m_Nz(nz), m_data(nx * ny * nz) {}

	Array3D(size_t nx, size_t ny, size_t nz, const T &val) : m_Nx(nx), m_Ny(ny), m_Nz(nz), m_data(nx * ny * nz, val) {}

	~Array3D() = default;

	void resize(size_t nI, size_t nJ, size_t nK)
	{
		m_Nx = nI;
		m_Ny = nJ;
		m_Nz = nK;
		m_data.resize(nI * nJ * nK);
	}

	// 0-based indexing
	T &at(size_t i, size_t j, size_t k)
	{
		return m_data[idx(i, j, k)];
	}

	T at(size_t i, size_t j, size_t k) const
	{
		return m_data[idx(i, j, k)];
	}

	// 1-based indexing
	T &operator()(size_t i, size_t j, size_t k)
	{
		return at(i - 1, j - 1, k - 1);
	}

	T operator()(size_t i, size_t j, size_t k) const
	{
		return at(i - 1, j - 1, k - 1);
	}

private:
	// Internal 0-based indexing interface.
	size_t idx(size_t i, size_t j, size_t k) const
	{
		return i + m_Nx * (j + m_Ny * k);
	}
};

// Basic mathematical types
typedef double Scalar;

class Vector
{
private:
	Scalar m_data[3];

public:
	Vector() : m_data{ 0.0, 0.0, 0.0 } {}

	Vector(Scalar val) : m_data{ val, val, val } {}

	Vector(Scalar v1, Scalar v2, Scalar v3) : m_data{ v1, v2, v3 } {}

	~Vector() = default;

	Scalar operator()(size_t idx) const // 1-based indexing
	{
		return m_data[idx - 1];
	}

	Scalar &operator()(size_t idx) // 1-based indexing
	{
		return m_data[idx - 1];
	}

	Scalar x() const { return this->operator()(1); }
	Scalar y() const { return this->operator()(2); }
	Scalar z() const { return this->operator()(3); }

	Scalar &x() { return this->operator()(1); }
	Scalar &y() { return this->operator()(2); }
	Scalar &z() { return this->operator()(3); }
};

class Tensor
{
private:
	Scalar m_data[3][3];

public:
	Tensor() : m_data{ {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} } {}

	~Tensor() = default;

	Scalar operator()(size_t i, size_t j) const // 1-based indexing
	{
		return m_data[i - 1][j - 1];
	}

	Scalar &operator()(size_t i, size_t j) // 1-based indexing
	{
		return m_data[i - 1][j - 1];
	}

	Scalar xx() const { return this->operator()(1, 1); }
	Scalar xy() const { return this->operator()(1, 2); }
	Scalar xz() const { return this->operator()(1, 3); }
	Scalar yx() const { return this->operator()(2, 1); }
	Scalar yy() const { return this->operator()(2, 2); }
	Scalar yz() const { return this->operator()(2, 3); }
	Scalar zx() const { return this->operator()(3, 1); }
	Scalar zy() const { return this->operator()(3, 2); }
	Scalar zz() const { return this->operator()(3, 3); }

	Scalar &xx() { return this->operator()(1, 1); }
	Scalar &xy() { return this->operator()(1, 2); }
	Scalar &xz() { return this->operator()(1, 3); }
	Scalar &yx() { return this->operator()(2, 1); }
	Scalar &yy() { return this->operator()(2, 2); }
	Scalar &yz() { return this->operator()(2, 3); }
	Scalar &zx() { return this->operator()(3, 1); }
	Scalar &zy() { return this->operator()(3, 2); }
	Scalar &zz() { return this->operator()(3, 3); }
};

// 1st-order derivative using central difference.
inline double fder1(double fl, double fm, double fr, double dxl, double dxr)
{
	const double dxrl = dxr / dxl;
	const double dxlr = dxl / dxr;
	return (dxlr*fr - dxrl * fl - (dxlr - dxrl)*fm) / (dxl + dxr);
}

// 2nd-order derivative using central difference.
inline double fder2(double fl, double fm, double fr, double dxl, double dxr)
{
	const double inv_dxl = 1.0 / dxl;
	const double inv_dxr = 1.0 / dxr;
	return 2.0 / (dxl + dxr) * (fl * inv_dxl + fr * inv_dxr - fm * (inv_dxl + inv_dxr));
}

const Scalar T0 = 300.0; // K
const Scalar P0 = 101325.0; // Pa
const Scalar rho = 1.225; //kg/m^3
const Scalar U0 = 1.0; // m/s
const Scalar nu = 1e-5; //m^2/s

const Scalar g = 9.80665; // m/s^2
const Vector f(0.0, 0.0, -g);

size_t iter = 10;
const size_t MAX_ITER = 2000;
Scalar dt = 0.0, t = 0.0; // s
const Scalar MAX_TIME = 100.0; // s

class Point
{
public:
	Vector V_star, N, F, grad_p, grad_u, grad_v, vel_laplace, grad_dp;
	Scalar div_V_star, dp;

public:
	Point(double x = 0.0, double y = 0.0, double z = 0.0) :
		c(x, y, z),
		vel(0.0),
		p(101325.0),
		rho(1.0),
		V_star(0.0),
		N(0.0),
		F(0.0),
		grad_p(0.0),
		grad_u(0.0),
		grad_v(0.0),
		vel_laplace(0.0),
		div_V_star(0.0),
		dp(0.0),
		grad_dp(0.0)
	{
	}

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

int main(int argc, char *argv[])
{
	//Read grid
	XF_MSH mesh;
	mesh.readFromFile("grid/fluent.msh");

	return 0;
}
