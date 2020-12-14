#include "../inc/BC.h"
#include "../inc/Spatial.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

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

    BC_Nodal();
}

void prepare_first_run()
{
    /// TODO
}
