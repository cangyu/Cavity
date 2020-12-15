#include <fstream>
#include "../inc/Spatial.h"
#include "../inc/Gradient.h"
#include "../inc/CHEM.h"
#include "../inc/IO.h"
#include "../inc/BC.h"
#include "../inc/IC.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;

/// TODO gradient on face

void IC_Zero()
{
    const Scalar P0 = 101325.0; /// Pa
    const Scalar T0 = 300.0; /// K
    const Scalar rho0 = EOS(P0, T0); /// kg/m^3

    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Cell
    for (auto &C : cell)
    {
        C.U.setZero();
        C.p = P0;
        C.T = T0;
        C.rho = rho0;
    }

    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Face(Internal)
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

    /// Enforce B.C. for {$\vec{U}$, $p$, $T$} @(n)
    BC_Primitive();

    /// {$\nabla \rho$, $\nabla \vec{U}$, $\nabla p$, $\nabla T$} @(n), Cell
    GRAD_Cell_Density();
    GRAD_Cell_Velocity();
    GRAD_Cell_Pressure();
    GRAD_Cell_Temperature();

    /// {$\vec{U}$, $p$, $T$} @(n), Face(Boundary)
    INTERP_BoundaryFace_Velocity();
    INTERP_BoundaryFace_Pressure();
    INTERP_BoundaryFace_Temperature();

    /// {$\rho$} @(n), Face(Boundary)
    for (auto &f : face)
    {
        if(f.at_boundary)
        {
            f.rho = EOS(f.p, f.T);
        }
    }

    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Node
    INTERP_Node_Primitive();

    /// {$\tau$} @(n), Cell, Face(Boundary+Internal)
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();
}

void IC_File(const std::string &DATA_PATH, int &iter, Scalar &t)
{
    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Cell, Face(Boundary+Internal), Node
    std::ifstream dts(DATA_PATH);
    if(dts.fail())
        throw failed_to_open_file(DATA_PATH);
    read_data(dts, iter, t);
    dts.close();

    BC_Primitive();

    /// {$\nabla \rho$, $\nabla \vec{U}$, $\nabla p$, $\nabla T$} @(n), Cell
    GRAD_Cell_Density();
    GRAD_Cell_Velocity();
    GRAD_Cell_Pressure();
    GRAD_Cell_Temperature();

    /// {$\vec{U}$, $p$, $T$} @(n), Face(Boundary)
    INTERP_BoundaryFace_Velocity();
    INTERP_BoundaryFace_Pressure();
    INTERP_BoundaryFace_Temperature();

    /// {$\rho$} @(n), Face(Boundary)
    for (auto &f : face)
    {
        if(f.at_boundary)
        {
            f.rho = EOS(f.p, f.T);
        }
    }

    /// {$\tau$} @(n), Cell, Face(Boundary+Internal)
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();
}
