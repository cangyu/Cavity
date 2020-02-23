#include "custom_type.h"
#include "IC.h"

extern size_t NumOfPnt;
extern size_t NumOfFace;
extern size_t NumOfCell;

extern NaturalArray<Point> pnt; // Node objects
extern NaturalArray<Face> face; // Face objects
extern NaturalArray<Cell> cell; // Cell objects


/******************************************************** I.C. *******************************************************/

/**
 * Initial conditions on all nodes, faces and cells.
 * Boundary elements are also set identical to interior, will be corrected in BC routine.
 */
void IC()
{
    const Scalar rho0 = 1.225; //kg/m^3	
    const Scalar P0 = 101325.0; // Pa
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
        f_dst.T = T0;
    }

    // Cell
    for (size_t i = 1; i <= NumOfCell; ++i)
    {
        auto &c_dst = cell(i);
        c_dst.rho0 = rho0;
        c_dst.U0 = ZERO_VECTOR;
        c_dst.p0 = P0;
        c_dst.T0 = T0;
    }
}
