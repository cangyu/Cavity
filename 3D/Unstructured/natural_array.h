#ifndef __NATURAL_ARRAY_H__
#define __NATURAL_ARRAY_H__

#include <vector>
#include <cstddef>

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

#endif
