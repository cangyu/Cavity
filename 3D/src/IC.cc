#include "../inc/custom_type.h"
#include "../inc/IO.h"
#include "../inc/IC.h"

extern int NumOfPnt;
extern int NumOfFace;
extern int NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

static const Scalar rho0 = 1.225; /// kg/m^3
static const Scalar P0 = 0.0; /// Pa
static const Scalar T0 = 300.0; /// K

static void interp_cell_to_node()
{
    for (int i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);

        const auto &dc = n_dst.dependentCell;
        const auto &wf = n_dst.cellWeightingCoef;

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
            n_dst.rho += cwf * dc.at(j)->rho0;
            n_dst.U += cwf * dc.at(j)->U0;
            n_dst.p += cwf * dc.at(j)->p0;
            n_dst.T += cwf * dc.at(j)->T0;
        }
    }
}

static void interp_cell_to_face()
{
    for (int i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);

        if (f_dst.atBdry)
        {
            Cell *c = nullptr;
            if(f_dst.c0)
                c = f_dst.c0;
            else if(f_dst.c1)
                c = f_dst.c1;
            else
                throw empty_connectivity(f_dst.index);

            f_dst.rho = c->rho0;
            f_dst.U = c->U0;
            f_dst.p = c->p0;
            f_dst.T = c->T0;
        }
        else
        {
            f_dst.rho = f_dst.ksi0 * f_dst.c0->rho0 + f_dst.ksi1 * f_dst.c1->rho0;
            f_dst.U = f_dst.ksi0 * f_dst.c0->U0 + f_dst.ksi1 * f_dst.c1->U0;
            f_dst.p = f_dst.ksi0 * f_dst.c0->p0 + f_dst.ksi1 * f_dst.c1->p0;
            f_dst.T = f_dst.ksi0 * f_dst.c0->T0 + f_dst.ksi1 * f_dst.c1->T0;
        }
    }
}

/**
 * Initial conditions on all nodes, faces and cells.
 * Boundary elements are also set identical to interior,
 * will be corrected in BC routine.
 */
void IC(const std::string &inp)
{
    /// Cell primitive vars
    if (inp.empty())
    {
        for (int i = 1; i <= NumOfCell; ++i)
        {
            auto &c_dst = cell(i);
            c_dst.rho0 = rho0;
            c_dst.U0 = ZERO_VECTOR;
            c_dst.p0 = P0;
            c_dst.T0 = T0;
        }
    }
    else
        read_tec_solution(inp);

    /// Cell conservative vars
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c_dst = cell(i);
        c_dst.rhoU0 = c_dst.rho0 * c_dst.U0;
    }

    /// Nodal values
    interp_cell_to_node();

    /// Face primitive vars
    interp_cell_to_face();

    /// Face conservative vars
    for (size_t i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);
        f_dst.rhoU = f_dst.rho * f_dst.U;
    }

    /// Face gradient
    for (size_t i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);
        f_dst.grad_p_prime = ZERO_VECTOR;
    }
}
