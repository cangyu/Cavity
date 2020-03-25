#ifndef __NATURAL_ARRAY_H__
#define __NATURAL_ARRAY_H__

#include <vector>
#include <string>
#include <cstddef>
#include <stdexcept>

class Array1D : public std::vector<double>
{
public:
	Array1D(size_t nx, double val = 0.0) : std::vector<double>(nx, val) {}

	// 1-based indexing
	double &operator()(size_t i) { return at(i - 1); }

	double operator()(size_t i) const { return at(i - 1); }
};

class Array2D
{
private:
	std::vector<double> m_data;
	size_t m_Nx, m_Ny;

public:
	Array2D(size_t nx, size_t ny, double val = 0.0) :
		m_Nx(nx),
		m_Ny(ny),
		m_data(nx * ny, val)
	{
		if (nx == 0 || ny == 0)
			throw std::runtime_error("Invalid size: nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny));
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

#endif
