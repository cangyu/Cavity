#ifndef __CUSTOM_TYPE_H__
#define __CUSTOM_TYPE_H__

#include <vector>
#include <string>
#include <array>
#include <cstddef>
#include <stdexcept>
#include "Eigen/Dense"
#include "Eigen/Sparse"

/***************************************************** Marcos ********************************************************/

#define ZERO_INDEX 0
#define ZERO_SCALAR 0.0
#define ZERO_VECTOR {0.0, 0.0, 0.0}

/****************************************************** Types ********************************************************/

/* Math types */
typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 3, 3> Tensor;

/* BC types */
enum BC_CATEGORY { Dirichlet = 0, Neumann, Robin };
static const std::array<std::string, 3> BC_CATEGORY_STR = { "Dirichlet", "Neumann", "Robin" };

/* 1-based array */
template<typename T>
class NaturalArray : public std::vector<T>
{
private:
    struct index_is_zero : public std::invalid_argument
    {
        index_is_zero() : std::invalid_argument("0 is invalid when using 1-based index.") {}
    };

public:
    NaturalArray() : std::vector<T>() {}
    explicit NaturalArray(size_t n) : std::vector<T>(n) {}
    NaturalArray(size_t n, const T &val) : std::vector<T>(n, val) {}
    ~NaturalArray() = default;

    /* 1-based indexing */
    T &operator()(long long i)
    {
        if (i >= 1)
            return std::vector<T>::at(i - 1);
        else if (i <= -1)
            return std::vector<T>::at(std::vector<T>::size() + i);
        else
            throw index_is_zero();
    }
    const T &operator()(long long i) const
    {
        if (i >= 1)
            return std::vector<T>::at(i - 1);
        else if (i <= -1)
            return std::vector<T>::at(std::vector<T>::size() + i);
        else
            throw index_is_zero();
    }
};

/* Geom elements */
struct Cell;
struct Patch;
struct Point
{
    // 1-based global index
    int index = ZERO_INDEX;

    // 3D Location
    Vector coordinate = ZERO_VECTOR;

    // Boundary flag
    bool atBdry = false;

    // Connectivity
    NaturalArray<Cell*> dependentCell;
    NaturalArray<Scalar> cellWeightingCoef;

    // Physical variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;
};
struct Face
{
    // 1-based global index
    int index = ZERO_INDEX;

    // 3D location of face centroid
    Vector center = ZERO_VECTOR;

    // Area of the face element
    Scalar area = ZERO_SCALAR;
    Vector n01 = ZERO_VECTOR, n10 = ZERO_VECTOR;

    // Connectivity
    NaturalArray<Point*> vertex;
    Cell *c0 = nullptr, *c1 = nullptr;

    // Displacement vector
    Vector r0 = ZERO_VECTOR, r1 = ZERO_VECTOR;

    // Boundary flags
    bool atBdry = false;
    Patch *parent = nullptr;
    BC_CATEGORY rho_BC = Dirichlet;
    std::array<BC_CATEGORY, 3> U_BC = { Dirichlet, Dirichlet, Dirichlet };
    BC_CATEGORY p_BC = Neumann;
    BC_CATEGORY T_BC = Neumann;

    // Ghost variables if needed
    Scalar rho_ghost = ZERO_SCALAR;
    Vector U_ghost = ZERO_VECTOR;
    Scalar p_ghost = ZERO_SCALAR;
    Scalar T_ghost = ZERO_SCALAR;

    // Physical properties
    Scalar mu = ZERO_SCALAR;

    // Physical variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;
    Vector rhoU = ZERO_VECTOR;

    // Gradient of physical variables
    Vector grad_rho = ZERO_VECTOR;
    Tensor grad_U;
    Vector grad_p = ZERO_VECTOR;
    Vector grad_T = ZERO_VECTOR;
    Tensor tau;

    /* Fractional-Step Method temporary variables */
    Vector rhoU_star = ZERO_VECTOR;
};
struct Cell
{
    // 1-based global index
    int index = ZERO_INDEX;

    // 3D location of cell centroid
    Vector center = ZERO_VECTOR;

    // Volume of the cell element
    Scalar volume = ZERO_SCALAR;

    // Surface vector
    NaturalArray<Vector> S;

    // Connectivity
    NaturalArray<Point*> vertex;
    NaturalArray<Face*> surface;
    NaturalArray<Cell*> adjCell;

    // Least-squares method variables
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_rho;
    std::array<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3> J_INV_U;
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_p, J_INV_p_prime;
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_T;

    /* Variables at current time-level */
    // Physical variables
    Scalar rho0 = ZERO_SCALAR;
    Vector U0 = ZERO_VECTOR;
    Scalar p0 = ZERO_SCALAR;
    Scalar T0 = ZERO_SCALAR;
    Vector rhoU0 = ZERO_VECTOR;

    /* Runge-Kutta temporary variables */
    // Physical properties
    Scalar mu = ZERO_SCALAR;

    // Physical variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;

    // Gradients
    Vector grad_rho;
    Tensor grad_U;
    Vector grad_p;
    Vector grad_T;

    // Equation residuals
    Scalar continuity_res;
    Vector momentum_res;
    Scalar energy_res;

    /* Fractional-Step Method temporary variables */
    // Momentum equation
    Vector pressure_flux;
    Vector convection_flux;
    Vector viscous_flux;
    Vector rhoU_star;
    Scalar p_prime;
    Vector grad_p_prime;

};
struct Patch
{
    std::string name;
    int BC;
    NaturalArray<Face*> surface;
};

/********************************************* Errors and Exceptions *************************************************/

struct failed_to_open_file : public std::runtime_error
{
    explicit failed_to_open_file(const std::string &fn) : std::runtime_error("Failed to open target file: \"" + fn + "\".") {}
};
struct unsupported_boundary_condition : public std::invalid_argument
{
    explicit unsupported_boundary_condition(BC_CATEGORY x) : std::invalid_argument("\"" + BC_CATEGORY_STR[x] + "\" condition is not supported.") {}
};
struct empty_connectivity : public std::runtime_error
{
    explicit empty_connectivity(int idx) : std::runtime_error("Both c0 and c1 are NULL on face " + std::to_string(idx) + ".") {}
};
struct inconsistent_connectivity : public std::runtime_error
{
    explicit inconsistent_connectivity(const std::string &msg) : std::runtime_error(msg) {}
};
struct unexpected_patch : public std::runtime_error
{
    unexpected_patch(const std::string &name) : std::runtime_error("Patch \"" + name + "\" is not expected to be a boundary patch.") {}
};


#endif
