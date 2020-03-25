#include <cmath>
#include "../inc/CHEM.h"

extern size_t NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/**
 * Dynamic viscosity of ideal gas.
 * @param T
 * @return
 */
Scalar Sutherland(Scalar T)
{
    return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
}

static const Scalar Re = 100.0;

void calcCellProperty()
{
    for (auto &c : cell)
    {
        // Dynamic viscosity
        // c.mu = Sutherland(c.T);
        c.mu = 1.225 / Re;
    }
}

void calcFaceProperty()
{
    for (auto &f : face)
    {
        // Dynamic viscosity
        // f.mu = Sutherland(f.T);
        f.mu = 1.225 / Re;
    }
}
