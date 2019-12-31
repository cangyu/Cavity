#ifndef __MATH_TYPE_HPP__
#define __MATH_TYPE_HPP__

#include <array>
#include <utility>
#include <cmath>

inline int kronecker(int i, int j) { return i == j ? 1 : 0; }

inline int eddington(int r, int s, int t)
{
	const int e = (r - s) * (s - t) * (t - r);

	if (e > 0)
		return 1;
	else if (e < 0)
		return -1;
	else
		return 0;
}

typedef double Scalar;

class Vector : public std::array<Scalar, 3>
{
public:
	Vector() : std::array<Scalar, 3>{ 0.0, 0.0, 0.0 } {}
	Vector(Scalar val) : std::array<Scalar, 3>{ val, val, val } {}
	Vector(Scalar v1, Scalar v2, Scalar v3) : std::array<Scalar, 3>{ v1, v2, v3 } {}
	Vector(const Vector &rhs) : std::array<Scalar, 3>{ rhs.x(), rhs.y(), rhs.z() } {}

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

	// Operator
	Vector &operator=(const Vector &rhs)
	{
		this->x() = rhs.x();
		this->y() = rhs.y();
		this->z() = rhs.z();
		return *this;
	}
	Vector &operator+=(const Vector &rhs)
	{
		this->x() += rhs.x();
		this->y() += rhs.y();
		this->z() += rhs.z();
		return *this;
	}
	Vector &operator-=(const Vector &rhs)
	{
		this->x() -= rhs.x();
		this->y() -= rhs.y();
		this->z() -= rhs.z();
		return *this;
	}
	Vector &operator*=(const Scalar &rhs)
	{
		this->x() *= rhs;
		this->y() *= rhs;
		this->z() *= rhs;
		return *this;
	}
	Vector &operator/=(const Scalar &rhs)
	{
		this->x() /= rhs;
		this->y() /= rhs;
		this->z() /= rhs;
		return *this;
	}
	Vector operator+(const Vector &rhs)
	{
		return Vector(this->x() + rhs.x(), this->y() + rhs.y(), this->z() + rhs.z());
	}
	Vector operator-(const Vector &rhs)
	{
		return Vector(this->x() - rhs.x(), this->y() - rhs.y(), this->z() - rhs.z());
	}
	Vector operator*(const Scalar &rhs)
	{
		return Vector(this->x() * rhs, this->y() * rhs, this->z() * rhs);
	}
	Vector operator/(const Scalar &rhs)
	{
		return Vector(this->x() / rhs, this->y() / rhs, this->z() / rhs);
	}
};

class Tensor
{
private:
	Vector m_f1;
	Vector m_f2;
	Vector m_f3;

public:
	Tensor() : m_f1{ 0.0, 0.0, 0.0 }, m_f2{ 0.0, 0.0, 0.0 }, m_f3{ 0.0, 0.0, 0.0 } {}

	// 1-based indexing
	const Scalar &operator()(int i, int j) const
	{
		switch (i)
		{
		case 1:
			return m_f1(j);
		case 2:
			return m_f2(j);
		case 3:
			return m_f3(j);
		default:
			throw std::runtime_error("Invalid index.");
		}
	}
	Scalar &operator()(int i, int j)
	{
		switch (i)
		{
		case 1:
			return m_f1(j);
		case 2:
			return m_f2(j);
		case 3:
			return m_f3(j);
		default:
			throw std::runtime_error("Invalid index.");
		}
	}

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

	// Access row vector
	const Vector &row(int i) const
	{
		switch (i)
		{
		case 1:
			return m_f1;
		case 2:
			return m_f2;
		case 3:
			return m_f3;
		default:
			throw std::runtime_error("Invalid row index.");
		}
	}
	Vector &row(int i)
	{
		switch (i)
		{
		case 1:
			return m_f1;
		case 2:
			return m_f2;
		case 3:
			return m_f3;
		default:
			throw std::runtime_error("Invalid row index.");
		}
	}

	Scalar I1() const
	{
		return xx() + yy() + zz();
	}

	Scalar I2() const
	{
		Scalar ret = std::pow(I1(), 2);
		for (int i = 1; i <= 3; ++i)
			for (int j = 1; j <= 3; ++j)
				ret -= this->operator()(i, j) * this->operator()(j, i);
		return 0.5*ret;
	}

	Scalar I3() const
	{
		// TODO

		return 0;
	}
};

// Trace
inline Scalar tr(const Tensor &T)
{
	return T.xx() + T.yy() + T.zz();
}

// Determinant
inline Scalar det(const Tensor &T)
{
	Scalar ret = 0.0;
	for (int i = 1; i <= 3; ++i)
		for (int j = 1; j <= 3; ++j)
			for (int k = 1; k <= 3; ++k)
				ret += eddington(i, j, k) * T(1, i) * T(2, j) * T(3, k);
	return ret;
}

inline void dyad(const Vector &a, const Vector &b, Tensor &ret)
{
	ret(1, 1) = a(1) * b(1);
	ret(1, 2) = a(1) * b(2);
	ret(1, 3) = a(1) * b(3);
	ret(2, 1) = a(2) * b(1);
	ret(2, 2) = a(2) * b(2);
	ret(2, 3) = a(2) * b(3);
	ret(3, 1) = a(3) * b(1);
	ret(3, 2) = a(3) * b(2);
	ret(3, 3) = a(3) * b(3);
}

inline void dot_product(const Vector &a, const Vector &b, Scalar &dst)
{
	dst = a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline void dot_product(const Vector &a, const Tensor &T, Vector &ret)
{
	for (int j = 1; j <= 3; ++j)
		ret(j) = a(1) * T(1, j) + a(2) * T(2, j) + a(3) * T(3, j);
}

inline void dot_product(const Tensor &T, const Vector &a, Vector &ret)
{
	for (int i = 1; i <= 3; ++i)
		ret(i) = a(1) * T(i, 1) + a(2) * T(i, 2) + a(3) * T(i, 3);
}

inline void dot_product(const Tensor &A, const Tensor &B, Tensor &P)
{
	for (int i = 1; i <= 3; ++i)
		for (int j = 1; j <= 3; ++j)
			P(i, j) = A(i, 1) * B(1, j) + A(i, 2) * B(2, j) + A(i, 3) * B(3, j);
}

inline void cross_product(const Vector &a, const Vector &b, Vector &dst)
{
	dst.x() = a.y()*b.z() - a.z()*b.y();
	dst.y() = a.z()*b.x() - a.x()*b.z();
	dst.z() = a.x()*b.y() - a.y()*b.x();
}

inline Tensor dyad(const Vector &a, const Vector &b)
{
	Tensor ret;
	dyad(a, b, ret);
	return ret;
}

inline Scalar dot_product(const Vector &a, const Vector &b)
{
	Scalar ret = 0;
	dot_product(a, b, ret);
	return ret;
}

inline Vector cross_product(const Vector &a, const Vector &b)
{
	Vector ret;
	cross_product(a, b, ret);
	return ret;
}

#endif
