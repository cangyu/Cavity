#include <fstream>
#include "../inc/Spatial.h"
#include "../inc/CHEM.h"
#include "../inc/IO.h"
#include "../inc/BC.h"
#include "../inc/IC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

void IC_Zero()
{
    const Scalar P0 = 101325.0; /// Pa
    const Scalar T0 = 300.0; /// K
    const Scalar rho0 = EOS(P0, T0); /// kg/m^3

    /// Cell
    for (auto &C : cell)
    {
        C.U.setZero();
        C.p = P0;
        C.T = T0;
        C.rho = rho0;
    }

    /// Face(Internal)
    for(auto &f : face)
    {
        if(!f.at_boundary)
        {
            f.U.setZero();
            f.p = P0;
            f.T = T0;
            f.rho = rho0;
        }
    }

    BC_Primitive();



    /// Node
    INTERP_Node_Primitive();
}

void IC_File(const std::string &DATA_PATH, int &iter, Scalar &t)
{
    std::ifstream dts(DATA_PATH);
    if(dts.fail())
        throw failed_to_open_file(DATA_PATH);
    read_data(dts, iter, t);
    dts.close();

    BC_Primitive();
}
