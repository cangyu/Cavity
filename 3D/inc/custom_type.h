#ifndef CUSTOM_TYPE_H
#define CUSTOM_TYPE_H

#include <string>
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../3rd_party/sxamg/include/sxamg.h"

/***************************************************** Math types *****************************************************/

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;
typedef Eigen::Matrix<Scalar, 3, 3> Tensor;

/******************************************************* Marcos *******************************************************/

#define ZERO_INDEX 0
#define ZERO_SCALAR 0.0
#define ZERO_VECTOR Vector::Zero()
#define ZERO_TENSOR Tensor::Zero()

/******************************************************* Types ********************************************************/

/* BC */
enum BC_CATEGORY { Dirichlet = 0, Neumann, Robin };
enum class BC_PHY : int {
    Wall = 0,
    Inlet = 1,
    Outlet = 2,
    Symmetry = 3
};

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

/* Geom elements */
struct Cell;
struct Patch;
struct Point
{
    /// 1-based global index
    int index = ZERO_INDEX;

    /// Boundary flag
    bool at_boundary;

    /// 3D cartesian location
    Vector coordinate = ZERO_VECTOR;

    /// Connectivity to cells
    NaturalArray<Cell*> dependent_cell;
    NaturalArray<Scalar> cell_weights; /// Harmonic by default

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

    /// Boundary flag
    bool at_boundary;

    /// 3D cartesian location of face centroid
    Vector centroid = ZERO_VECTOR;

    /// Area of the face
    Scalar area = ZERO_SCALAR;

    /// Unit normal vector
    Vector n01 = ZERO_VECTOR; /// From c0 to c1
    Vector n10 = ZERO_VECTOR; /// From c1 to c0

    /// Connectivity to nodes
    NaturalArray<Point*> vertex;

    /// Connectivity to cells
    Cell *c0 = nullptr, *c1 = nullptr;
    Vector r0 = ZERO_VECTOR, r1 = ZERO_VECTOR; /// Displacement vector
    Scalar ksi0 = ZERO_SCALAR, ksi1 = ZERO_SCALAR; /// Displacement ratio

    /// Possible connection to high-level
    Patch *parent = nullptr;

    /// (n)
    Scalar viscosity = ZERO_SCALAR;
    Scalar conductivity = ZERO_SCALAR;
    Scalar specific_heat_p = ZERO_SCALAR;
    Scalar specific_heat_v = ZERO_SCALAR;
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;
    Scalar h = ZERO_SCALAR;
    Vector rhoU = ZERO_VECTOR;
    Scalar rhoh = ZERO_SCALAR;
    Tensor tau = ZERO_TENSOR;
    Vector grad_rho = ZERO_VECTOR;
    Tensor grad_U = ZERO_TENSOR;
    Vector grad_p = ZERO_VECTOR;
    Vector grad_T = ZERO_VECTOR;

    /// (*)
    Scalar T_star;
    Vector rhoU_star;
    Vector grad_p_prime, grad_p_prime_sn;

    /// (m-1)
    Vector U_prev;
    Vector rhoU_prev;
    Scalar p_prev;
    Scalar h_prev;
    Vector grad_T_prev;
    Tensor tau_prev;

    /// (m)
    Scalar rho_next;
    Vector U_next;
    Scalar p_next;
    Scalar T_next;
    Scalar h_next;
    Vector rhoU_next;
    Scalar rhoh_next;
    Tensor grad_U_next;
    Vector grad_p_next;
    Vector grad_T_next;
    Tensor tau_next;

    /// B.C. only
    Vector sn_grad_U = ZERO_VECTOR;
    Scalar sn_grad_p = ZERO_SCALAR;
    Scalar sn_grad_T = ZERO_SCALAR;
};
struct Cell
{
    /// 1-based global index
    int index = ZERO_INDEX;

    /// 3D cartesian location of cell centroid
    Vector centroid = ZERO_VECTOR;

    /// Volume of the cell
    Scalar volume = ZERO_SCALAR;

    /// Connectivity
    NaturalArray<Point*> vertex;
    NaturalArray<Face*> surface;
    NaturalArray<Cell*> adjCell;

    /// Surface outward normal vector
    /// Follow the order in "surface"
    NaturalArray<Vector> S;

    /// Displacement vector between adjacent cell centroids
    /// Follow the order in "surface"
    NaturalArray<Vector> d;

    /// Surface vectors related to non-orthogonality
    /// Follow the order in "S"
    NaturalArray<Vector> Se;
    NaturalArray<Vector> St;

    /// (n)
    Scalar viscosity = ZERO_SCALAR;
    Scalar conductivity = ZERO_SCALAR;
    Scalar specific_heat_p = ZERO_SCALAR;
    Scalar specific_heat_v = ZERO_SCALAR;
    Scalar rho = ZERO_SCALAR;
    Vector U = ZERO_VECTOR;
    Scalar p = ZERO_SCALAR;
    Scalar T = ZERO_SCALAR;
    Scalar h = ZERO_SCALAR;
    Vector rhoU = ZERO_VECTOR;
    Scalar rhoh = ZERO_SCALAR;
    Tensor tau = ZERO_TENSOR;
    Vector grad_rho = ZERO_VECTOR;
    Tensor grad_U = ZERO_TENSOR;
    Vector grad_p = ZERO_VECTOR;
    Vector grad_T = ZERO_VECTOR;

    /// (*)
    Scalar rho_star, drhodt;
    Vector rhoU_star = ZERO_VECTOR;
    Vector U_star;
    Scalar p_prime = ZERO_SCALAR;
    Vector grad_p_prime = ZERO_VECTOR;
    Scalar h_star;
    Scalar T_star;
    Vector grad_T_star;
    Tensor TeC_INV = ZERO_TENSOR; /// Reconstruction matrix

    /// (m-1)
    Scalar rho_prev;
    Vector U_prev;
    Scalar p_prev;
    Scalar T_prev;
    Vector grad_p_prev;
    Tensor grad_U_prev;
    Tensor tau_prev;

    /// (m)
    Scalar rho_next;
    Vector U_next;
    Scalar p_next;
    Scalar T_next;
    Scalar h_next;
    Vector rhoU_next;
    Scalar rhoh_next;
    Vector grad_rho_next;
    Tensor grad_U_next;
    Vector grad_p_next;
    Vector grad_T_next;
    Tensor tau_next;
};
struct Patch
{
    /// Identifier
    std::string name;

    /// Components
    NaturalArray<Face*> surface;
    NaturalArray<Point*> vertex;

    /// B.C. classification
    BC_PHY BC;

    /// B.C. specification for each variable
    BC_CATEGORY U_BC;
    BC_CATEGORY p_BC;
    BC_CATEGORY T_BC;
};

/********************************************** Errors and Exceptions *************************************************/

struct failed_to_open_file : public std::runtime_error
{
    explicit failed_to_open_file(const std::string &fn) : std::runtime_error("Failed to open target file: \"" + fn + "\".") {}
};
struct unsupported_boundary_condition : public std::invalid_argument
{
    explicit unsupported_boundary_condition(BC_CATEGORY x) : std::invalid_argument("BC \"" + std::to_string((int)x) + "\" is not supported.") {}

    explicit unsupported_boundary_condition(const std::string &bc) : std::invalid_argument("\"" + bc + "\" is not supported.") {}

    explicit unsupported_boundary_condition(BC_PHY bc) : std::invalid_argument("\"" + std::to_string((int)bc) + "\" is not supported.") {}
};
struct dirichlet_bc_is_not_supported : public unsupported_boundary_condition
{
    dirichlet_bc_is_not_supported() : unsupported_boundary_condition(Dirichlet) {}
};
struct neumann_bc_is_not_supported : public unsupported_boundary_condition
{
    neumann_bc_is_not_supported() : unsupported_boundary_condition(Neumann) {}
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
    explicit unexpected_patch(const std::string &name) : std::runtime_error("\"" + name + "\" is not a pre-defined boundary patch.") {}
};
struct insufficient_vertexes : public std::runtime_error
{
    explicit insufficient_vertexes(size_t i) : std::runtime_error("No enough vertices within cell " + std::to_string(i) + ".") {}
};

/****************************************************** END ***********************************************************/

#endif
