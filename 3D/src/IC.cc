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

void IC_Zero()
{
    const Scalar P0 = 101325.0; /// Pa
    const Scalar T0 = 300.0; /// K

    /// {$\vec{U}$, $p$, $T$} @(n), Cell
    for (auto &C : cell)
    {
        C.U.setZero();
        C.p = P0;
        C.T = T0;
    }

    /// {$\vec{U}$, $p$, $T$} @(n), Face(Internal)
    for(auto &f : face)
    {
        if(!f.at_boundary)
        {
            f.U.setZero();
            f.p = P0;
            f.T = T0;
        }
    }

    /// Enforce B.C. for {$\vec{U}$, $p$, $T$} @(n), Face(Boundary)
    BC_Primitive();

    /// {$\nabla \vec{U}$, $\nabla p$, $\nabla T$} @(n), Cell & Face(Boundary+Internal)
    GRAD_Cell_Velocity();
    GRAD_Cell_Pressure();
    GRAD_Cell_Temperature();
    GRAD_Face_Velocity();
    GRAD_Face_Pressure();
    GRAD_Face_Temperature();

    /// {$\vec{U}$, $p$, $T$} @(n), Face(Boundary)
    INTERP_BoundaryFace_Velocity();
    INTERP_BoundaryFace_Pressure();
    INTERP_BoundaryFace_Temperature();

    /// {$\rho$} @(n), Cell & Face(Boundary+Internal)
    for (auto &C : cell)
    {
        C.rho = EOS(C.p, C.T);
    }
    for (auto &f : face)
    {
        f.rho = EOS(f.p, f.T);
    }

    /// {$\nabla \rho$} @(n), Cell & Face(Boundary+Internal)
    GRAD_Cell_Density();
    GRAD_Face_Density();

    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Node
    INTERP_Node_Primitive();

    /// Property
    CALC_Cell_Viscosity();
    CALC_Cell_Conductivity();
    CALC_Cell_SpecificHeat();
    CALC_Face_Viscosity();
    CALC_Face_Conductivity();
    CALC_Face_SpecificHeat();

    /// {$\tau$} @(n), Cell & Face(Boundary+Internal)
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();

    /// {$h$, $\rho h$, $\rho \vec{U}$} @(n), Cell & Face(Boundary+Internal)
    for (auto &C : cell)
    {
        C.h = Enthalpy(C.specific_heat_p, C.T);
        C.rhoh = C.rho * C.h;
        C.rhoU = C.rho * C.U;
    }
    for (auto &f : face)
    {
        f.h = Enthalpy(f.specific_heat_p, f.T);
        f.rhoh = f.rho * f.h;
        f.rhoU = f.rho * f.U;
    }
}

void IC_File(const std::string &DATA_PATH, int &iter, Scalar &t)
{
    /// {$\rho$, $\vec{U}$, $p$, $T$} @(n), Cell, Face(Boundary+Internal), Node
    std::ifstream dts(DATA_PATH);
    if(dts.fail())
        throw failed_to_open_file(DATA_PATH);
    read_data(dts, iter, t);
    dts.close();

    /// {$\nabla \rho$, $\nabla \vec{U}$, $\nabla p$, $\nabla T$} @(n), Cell & Face(Boundary+Internal)
    GRAD_Cell_Density();
    GRAD_Cell_Velocity();
    GRAD_Cell_Pressure();
    GRAD_Cell_Temperature();
    GRAD_Face_Density();
    GRAD_Face_Velocity();
    GRAD_Face_Pressure();
    GRAD_Face_Temperature();

    /// Property
    CALC_Cell_Viscosity();
    CALC_Cell_Conductivity();
    CALC_Cell_SpecificHeat();
    CALC_Face_Viscosity();
    CALC_Face_Conductivity();
    CALC_Face_SpecificHeat();

    /// {$\tau$} @(n), Cell & Face(Boundary+Internal)
    CALC_Cell_ViscousShearStress();
    CALC_Face_ViscousShearStress();

    /// {$h$, $\rho h$, $\rho \vec{U}$} @(n), Cell & Face(Boundary+Internal)
    for (auto &C : cell)
    {
        C.h = Enthalpy(C.specific_heat_p, C.T);
        C.rhoh = C.rho * C.h;
        C.rhoU = C.rho * C.U;
    }
    for (auto &f : face)
    {
        f.h = Enthalpy(f.specific_heat_p, f.T);
        f.rhoh = f.rho * f.h;
        f.rhoU = f.rho * f.U;
    }
}
