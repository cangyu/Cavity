#include <chrono>
#include <sstream>
#include <iomanip>
#include "../inc/BC.h"
#include "../inc/Miscellaneous.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;

const std::array<std::string, 3> unsupported_boundary_condition::BC_CATEGORY_STR = {"Dirichlet","Neumann","Robin"};

double duration(const clock_t &startTime, const clock_t &endTime)
{
    return static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
}

std::string time_stamp_str()
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d-%H%M%S");
    return ss.str();
}

void interp_nodal_primitive_var()
{
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);

        const auto &dc = n_dst.dependent_cell;
        const auto &wf = n_dst.cell_weights;

        const auto N = dc.size();
        if (N != wf.size())
            throw std::runtime_error("Inconsistency detected!");

        n_dst.rho = ZERO_SCALAR;
        n_dst.U = ZERO_VECTOR;
        n_dst.p = ZERO_SCALAR;
        n_dst.T = ZERO_SCALAR;
        for (int j = 0; j < N; ++j)
        {
            const auto cwf = wf.at(j);
            n_dst.rho += cwf * dc.at(j)->rho;
            n_dst.U += cwf * dc.at(j)->U;
            n_dst.p += cwf * dc.at(j)->p;
            n_dst.T += cwf * dc.at(j)->T;
        }
    }

    set_bc_nodal();
}

/**
 * Calculate vectors used for NON-ORTHOGONAL correction locally.
 * @param opt Choice of method.
 * 1 - Minimum Correction
 * 2 - Orthogonal Correction
 * 3 - Over-Relaxed Correction
 * @param d Local displacement vector.
 * @param S Local surface outward normal vector.
 * @param E Orthogonal part after decomposing "S".
 * @param T Non-Orthogonal part after decomposing "S", satisfying "S = E + T".
 */
void calc_noc_vec(int opt, const Vector &d, const Vector &S, Vector &E, Vector &T)
{
    Vector e = d;
    e /= d.norm();
    const Scalar S_mod = S.norm();
    const Scalar cos_theta = e.dot(S) / S_mod;

    if(opt == 1)
        E = S_mod * cos_theta * e;
    else if(opt == 2)
        E = S_mod * e;
    else if(opt == 3)
        E = S_mod / cos_theta * e;
    else
        throw std::invalid_argument("Invalid NON-ORTHOGONAL correction option!");

    T = S - E;
}

Scalar double_dot(const Tensor &A, const Tensor &B)
{
    Scalar ret = 0.0;

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            ret += A(i, j) * B(j, i);

    return ret;
}
