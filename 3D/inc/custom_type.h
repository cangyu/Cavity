#ifndef CUSTOM_TYPE_H
#define CUSTOM_TYPE_H

#include <vector>
#include <string>
#include <array>
#include <cstddef>
#include <stdexcept>
#include "../3rd_party/Eigen/Dense"
#include "../3rd_party/Eigen/Sparse"
#include "../3rd_party/SXAMG/include/sxamg.h"

/***************************************************** Math types *****************************************************/

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 3, 3> Tensor;

/******************************************************* Marcos *******************************************************/

#define ZERO_INDEX 0
#define ZERO_SCALAR 0.0
#define ZERO_VECTOR Vector::Zero()
#define ZERO_TENSOR Tensor::Zero()

/****************************************************** BC types ******************************************************/

enum BC_CATEGORY { Dirichlet = 0, Neumann, Robin };
static const std::array<std::string, 3> BC_CATEGORY_STR = { "Dirichlet", "Neumann", "Robin" };

/******************************************************* Types ********************************************************/

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

    /// 1-based indexing
    T &operator()(int i)
    {
        if (i >= 1)
            return std::vector<T>::at(i - 1);
        else if (i <= -1)
            return std::vector<T>::at(std::vector<T>::size() + i);
        else
            throw index_is_zero();
    }

    const T &operator()(int i) const
    {
        if (i >= 1)
            return std::vector<T>::at(i - 1);
        else if (i <= -1)
            return std::vector<T>::at(std::vector<T>::size() + i);
        else
            throw index_is_zero();
    }
};

/******************************************************* Types ********************************************************/

/* Geom elements */
struct Cell;
struct Patch;
struct Point
{
    /// 1-based global index
    int index = ZERO_INDEX;

    /// 3D cartesian location
    Vector coordinate = ZERO_VECTOR;

    /// Boundary flag
    bool atBdry = false;

    /// Connectivity to cells
    NaturalArray<Cell*> dependentCell;
    NaturalArray<Scalar> cellWeightingCoef; /// Harmonic by default

    /// Primitive variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;
};
struct Face
{
    /// 1-based global index
    int index = ZERO_INDEX;

    /// 3D cartesian location of face centroid
    Vector center = ZERO_VECTOR;

    /// Area of the face
    Scalar area = ZERO_SCALAR;

    /// Unit normal vector from either side
    Vector n01 = ZERO_VECTOR, n10 = ZERO_VECTOR;

    /// Connectivity to nodes
    NaturalArray<Point*> vertex;

    /// Connectivity to cells
    Cell *c0 = nullptr, *c1 = nullptr;
    Vector r0 = ZERO_VECTOR, r1 = ZERO_VECTOR; /// Displacement vector
    Scalar ksi0 = ZERO_SCALAR, ksi1 = ZERO_SCALAR; /// Displacement ratio

    /// Boundary flag
    bool atBdry = false;

    /// Possible connection to high-level
    Patch *parent = nullptr;

    /// B.C. specification for each variable
    BC_CATEGORY rho_BC = Dirichlet;
    std::array<BC_CATEGORY, 3> U_BC = { Dirichlet, Dirichlet, Dirichlet };
    BC_CATEGORY p_BC = Neumann, p_prime_BC = Neumann;
    BC_CATEGORY T_BC = Neumann;

    /// Ghost variables if needed
    Scalar rho_ghost = ZERO_SCALAR;
    Vector U_ghost = ZERO_VECTOR;
    Scalar p_ghost = ZERO_SCALAR;
    Scalar T_ghost = ZERO_SCALAR;

    /// Physical properties
    Scalar mu = ZERO_SCALAR;

    /// Physical variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar p_prime = ZERO_SCALAR; /// Used by pressure correction
    Scalar T = ZERO_SCALAR;
    Tensor tau = ZERO_TENSOR;
    Vector rhoU = ZERO_VECTOR;
    Vector rhoU_star = ZERO_VECTOR; /// Used by the Fractional-Step Method

    /// Gradient of physical variables
    /// Only used on internal faces
    Vector grad_rho = ZERO_VECTOR;
    Tensor grad_U = ZERO_TENSOR;
    Vector grad_p = ZERO_VECTOR;
    Vector grad_p_prime = ZERO_VECTOR;
    Vector grad_T = ZERO_VECTOR;

    /// Gradient of physical variables in surface outward normal direction
    /// Only used on boundary faces
    Scalar sn_grad_rho = ZERO_SCALAR;
    Vector sn_grad_U = ZERO_VECTOR;
    Scalar sn_grad_p = ZERO_SCALAR;
    Scalar sn_grad_p_prime = ZERO_SCALAR;
    Scalar sn_grad_T = ZERO_SCALAR;
};
struct Cell
{
    /// 1-based global index
    int index = ZERO_INDEX;

    /// 3D cartesian location of cell centroid
    Vector center = ZERO_VECTOR;

    /// Volume of the cell
    Scalar volume = ZERO_SCALAR;

    /// Connectivity
    NaturalArray<Point*> vertex;
    NaturalArray<Face*> surface;
    NaturalArray<Cell*> adjCell;

    /// Surface outward normal vector
    /// Follow the order in "surface"
    NaturalArray<Vector> S;

    /// Surface vectors related to non-orthogonality
    /// Follow the order in "S"
    NaturalArray<Vector> Se;
    NaturalArray<Vector> St;

    /// Variables used by the least-squares method
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_rho;
    std::array<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3> J_INV_U;
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_p, J_INV_p_prime;
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> J_INV_T;

    /* Variables at current time-level */
    Scalar rho0 = ZERO_SCALAR;
    Vector U0 = ZERO_VECTOR;
    Scalar p0 = ZERO_SCALAR;
    Scalar T0 = ZERO_SCALAR;
    Vector rhoU0 = ZERO_VECTOR;

    /* Variables at new time-level */
    /// Physical properties
    Scalar mu = ZERO_SCALAR;

    /// Physical variables
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar p_prime = ZERO_SCALAR; /// Used by pressure correction
    Scalar T = ZERO_SCALAR;
    Vector rhoU_star = ZERO_VECTOR; /// Used by the Fractional-Step Method

    /// Gradient of physical variables
    Vector grad_rho = ZERO_VECTOR;
    Tensor grad_U = ZERO_TENSOR;
    Vector grad_p = ZERO_VECTOR;
    Vector grad_T = ZERO_VECTOR;
    Vector grad_p_prime = ZERO_VECTOR; /// Used by pressure correction

    /// Equation residuals
    Scalar continuity_res = ZERO_SCALAR;
    Vector momentum_res = ZERO_VECTOR;
    Scalar energy_res = ZERO_SCALAR;

    /// Flux within momentum equation
    Vector pressure_flux = ZERO_VECTOR;
    Vector convection_flux = ZERO_VECTOR;
    Vector viscous_flux = ZERO_VECTOR;
};
struct Patch
{
    std::string name;
    int BC;
    NaturalArray<Face*> surface;
    NaturalArray<Point*> vertex;
};

/********************************************** Errors and Exceptions *************************************************/

struct failed_to_open_file : public std::runtime_error
{
    explicit failed_to_open_file(const std::string &fn) : std::runtime_error("Failed to open target file: \"" + fn + "\".") {}
};
struct unsupported_boundary_condition : public std::invalid_argument
{
    explicit unsupported_boundary_condition(BC_CATEGORY x) : std::invalid_argument("\"" + BC_CATEGORY_STR[x] + "\" condition is not supported.") {}
};
struct robin_bc_is_not_supported : public unsupported_boundary_condition
{
    robin_bc_is_not_supported() : unsupported_boundary_condition(Robin) {}
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
    explicit unexpected_patch(const std::string &name) : std::runtime_error("Patch \"" + name + "\" is not expected to be a boundary patch.") {}
};

struct insufficient_vertexes : public std::runtime_error
{
    explicit insufficient_vertexes(size_t i) : std::runtime_error("No enough vertices within cell " + std::to_string(i) + ".") {}
};

/****************************************************** END ***********************************************************/

#endif
