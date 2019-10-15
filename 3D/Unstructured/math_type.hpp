#ifndef __MATH_TYPE_HPP__
#define __MATH_TYPE_HPP__

#include <array>
#include <utility>

typedef double Scalar;

class Vector : public std::array<Scalar, 3>
{
public:
	Vector() : std::array<Scalar, 3>{ 0.0, 0.0, 0.0 } {}
	Vector(Scalar val) : std::array<Scalar, 3>{ val, val, val } {}
	Vector(Scalar v1, Scalar v2, Scalar v3) : std::array<Scalar, 3>{ v1, v2, v3 } {}

	// 1-based indexing
	const Scalar &operator()(int idx) const { return at(idx - 1); }
	Scalar &operator()(int idx) { return at(idx - 1); }

	// Access through component
	Scalar x() const { return at(0); }
	Scalar y() const { return at(1); }
	Scalar z() const { return at(2); }

	Scalar &x() { return at(0); }
	Scalar &y() { return at(1); }
	Scalar &z() { return at(2); }
};

class Tensor
{
private:
	Scalar m_data[3][3];

	void exchange(Tensor &rhs)
	{
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				std::swap(m_data[i][j], rhs.m_data[i][j]);
	}

public:
	Tensor() : m_data{ {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} } {}

	Tensor(const Tensor &rhs) = default;

	~Tensor() = default;

	Tensor &operator=(Tensor rhs)
	{
		exchange(rhs);
		return *this;
	}

	// 0-based indexing
	Scalar at(size_t i, size_t j) const { return m_data[i][j]; }
	Scalar &at(size_t i, size_t j) { return m_data[i][j]; }

	// 1-based indexing
	Scalar operator()(int i, int j) const { return at(i - 1, j - 1); }
	Scalar &operator()(int i, int j) { return at(i - 1, j - 1); }

	// Access through component
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

inline Scalar dot_product(const Vector &a, const Vector &b)
{
	return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline void dot_product(const Vector &a, const Vector &b, Scalar &dst)
{
	dst = a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline Vector cross_product(const Vector &a, const Vector &b)
{
	return Vector(a.y()*b.z() - a.z()*b.y(), a.z()*b.x() - a.x()*b.z(), a.x()*b.y() - a.y()*b.x());
}

inline void cross_product(const Vector &a, const Vector &b, Vector &dst)
{
	dst.x() = a.y()*b.z() - a.z()*b.y();
	dst.y() = a.z()*b.x() - a.x()*b.z();
	dst.z() = a.x()*b.y() - a.y()*b.x();
}


#endif
