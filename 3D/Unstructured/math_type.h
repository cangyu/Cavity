#ifndef __MATH_TYPE_H__
#define __MATH_TYPE_H__

#include <array>

typedef double Scalar;

class Vector : public std::array<Scalar, 3>
{
public:
	Vector() : std::array<Scalar, 3>{ 0.0, 0.0, 0.0 } {}

	Vector(Scalar val) : std::array<Scalar, 3>{ val, val, val } {}

	Vector(Scalar v1, Scalar v2, Scalar v3) : std::array<Scalar, 3>{ v1, v2, v3 } {}

	~Vector() = default;

	Scalar operator()(size_t idx) const // 1-based indexing
	{
		return at(idx - 1);
	}

	Scalar &operator()(size_t idx) // 1-based indexing
	{
		return at(idx - 1);
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

inline Scalar dot_product(const Vector &a, const Vector &b)
{
	Scalar ret = 0.0;
	for(size_t i = 0; i < 3; ++i)
		ret += a.at(i) * b.at(i);
	return ret;
}

#endif
