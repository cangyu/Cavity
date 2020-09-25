#include "../inc/custom_type.h"
#include "../inc/Miscellaneous.h"
#include "../inc/IC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
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
void IC()
{
    /// Cell
    for (auto &c_dst : cell)
    {
        /// Primitive variables
        c_dst.rho = c_dst.rho_prev = rho0;
        c_dst.U = ZERO_VECTOR;
        c_dst.p = c_dst.p_prev = P0;
        c_dst.T = c_dst.T_prev = T0;

        /// Conservative variables
        c_dst.rhoU = ZERO_VECTOR;
    }

    /// Internal Face
    for(auto &f : face)
    {
        if(!f.at_boundary)
        {
            /// Primitive variables
            f.rho = rho0;
            f.U = ZERO_VECTOR;
            f.p = P0;
            f.T = T0;

            /// Conservative variables
            f.rhoU = ZERO_VECTOR;
        }
    }

    /// Node
    interp_nodal_primitive_var();
}
