#include "../inc/custom_type.h"
#include "../inc/IC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/**
 * Initial conditions on all nodes, faces and cells.
 * Boundary elements are also set identical to interior, will be corrected in BC routine.
 */
void IC()
{
    const Scalar rho0 = 1.225; //kg/m^3	
    const Scalar P0 = 0.0; // Pa
    const Scalar T0 = 300.0; // K

    // Node
    for (size_t i = 1; i <= NumOfPnt; ++i)
    {
        auto &n_dst = pnt(i);
        n_dst.rho = rho0;
        n_dst.U = ZERO_VECTOR;
        n_dst.p = P0;
        n_dst.T = T0;
    }

    // Face
    for (size_t i = 1; i <= NumOfFace; ++i)
    {
        auto &f_dst = face(i);
        f_dst.rho = rho0;
        f_dst.U = ZERO_VECTOR;
        f_dst.p = P0;
        f_dst.grad_p_prime = ZERO_VECTOR;
        f_dst.T = T0;
        f_dst.rhoU = f_dst.rho * f_dst.U;
    }

    // Cell
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c_dst = cell(i);
        c_dst.rho0 = rho0;
        c_dst.U0 = ZERO_VECTOR;
        c_dst.p0 = P0;
        c_dst.T0 = T0;
        c_dst.rhoU0 = c_dst.rho0 * c_dst.U0;
    }
}
