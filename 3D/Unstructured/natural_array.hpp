#ifndef __NATURAL_ARRAY_HPP__
#define __NATURAL_ARRAY_HPP__

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
	T &operator()(size_t i) { return std::vector<T>::at(i - 1); }
	const T &operator()(size_t i) const { return std::vector<T>::at(i - 1); }
};

template<typename T>
class Array2D : public std::vector<T>
{
private:
	size_t m_Nx, m_Ny;

public:
	Array2D(size_t nx = 0, size_t ny = 0) : std::vector<T>(nx * ny), m_Nx(nx), m_Ny(ny) {}
	Array2D(size_t nx, size_t ny, const T &val) : std::vector<T>(nx * ny, val), m_Nx(nx), m_Ny(ny) {}
	~Array2D() = default;

	void resize(size_t nI, size_t nJ)
	{
		m_Nx = nI;
		m_Ny = nJ;
		resize(nI * nJ);
	}

	size_t IDIM() const { return m_Nx; }
	size_t JDIM() const { return m_Ny; }

	// 0-based indexing
	T &at(size_t i, size_t j) { return at(idx(i, j)); }
	const T &at(size_t i, size_t j) const { return at(idx(i, j)); }

	// 1-based indexing
	T &operator()(size_t i, size_t j) { return at(i - 1, j - 1); }
	const T &operator()(size_t i, size_t j) const { return at(i - 1, j - 1); }

private:
	// Internal 0-based indexing interface.
	size_t idx(size_t i, size_t j) const { return i + m_Nx * j; }
};

template<typename T>
class Array3D : public std::vector<T>
{
private:
	size_t m_Nx, m_Ny, m_Nz;

public:
	Array3D(size_t nx = 0, size_t ny = 0, size_t nz = 0) : std::vector<T>(nx * ny * nz), m_Nx(nx), m_Ny(ny), m_Nz(nz) {}
	Array3D(size_t nx, size_t ny, size_t nz, const T &val) : std::vector<T>(nx * ny * nz, val), m_Nx(nx), m_Ny(ny), m_Nz(nz) {}
	~Array3D() = default;

	void resize(size_t nI, size_t nJ, size_t nK)
	{
		m_Nx = nI;
		m_Ny = nJ;
		m_Nz = nK;
		resize(nI * nJ * nK);
	}

	size_t IDIM() const { return m_Nx; }
	size_t JDIM() const { return m_Ny; }
	size_t KDIM() const { return m_Nz; }

	// 0-based indexing
	T &at(size_t i, size_t j, size_t k) { return at(idx(i, j, k)); }
	const T &at(size_t i, size_t j, size_t k) const { return at(idx(i, j, k)); }

	// 1-based indexing
	T &operator()(size_t i, size_t j, size_t k) { return at(i - 1, j - 1, k - 1); }
	const T &operator()(size_t i, size_t j, size_t k) const { return at(i - 1, j - 1, k - 1); }

private:
	// Internal 0-based indexing interface.
	size_t idx(size_t i, size_t j, size_t k) const { return i + m_Nx * (j + m_Ny * k); }
};

#endif
