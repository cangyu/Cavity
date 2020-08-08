#include "../inc/custom_type.h"
#include "../inc/Discretization.h"
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

/**
 * Initial conditions on all cells.
 */
void IC(const std::string &inp)
{
    /// Cell primitive vars
    if (inp.empty())
    {
        for (auto &c_dst : cell)
        {
            c_dst.rho0 = rho0;
            c_dst.U0 = ZERO_VECTOR;
            c_dst.p0 = P0;
            c_dst.T0 = T0;
        }
    }
    else
        read_tec_solution(inp);

    /// Cell conservative vars
    for (auto &c_dst : cell)
    {
        c_dst.rhoU0 = c_dst.rho0 * c_dst.U0;
    }

    prepare_next_run();

    for(auto &f : face)
    {
        if(!f.at_boundary)
            f.rhoU = f.rho * f.U;
    }
}
