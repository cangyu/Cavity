#include <cmath>
#include "../inc/CHEM.h"

extern int NumOfPnt, NumOfFace, NumOfCell;
extern NaturalArray<Point> pnt;
extern NaturalArray<Face> face;
extern NaturalArray<Cell> cell;
extern NaturalArray<Patch> patch;
extern Scalar Re;

/**
 * Dynamic viscosity of ideal gas.
 * @param T Temperature in Kelvin.
 * @return Dynamic viscosity with unit "Kg / (m * s)".
 */
Scalar Sutherland(Scalar T)
{
    return 1.45e-6 * std::pow(T, 1.5) / (T + 110.0);
}

/**
 * Viscous shear stress of newtonian fluid under stokes hypothesis.
 * @param mu Dynamic viscosity.
 * @param grad_U Velocity gradient.
 * @param tau Shear stress.
 */
void Stokes(Scalar mu, const Tensor &grad_U, Tensor &tau)
{
    const Scalar loc_div3 = (grad_U(0, 0) + grad_U(1, 1) + grad_U(2, 2)) / 3.0;

    tau(0, 0) = 2.0 * mu * (grad_U(0, 0) - loc_div3);
    tau(1, 1) = 2.0 * mu * (grad_U(1, 1) - loc_div3);
    tau(2, 2) = 2.0 * mu * (grad_U(2, 2) - loc_div3);

    tau(0, 1) = tau(1, 0) = mu * (grad_U(0, 1) + grad_U(1, 0));
    tau(1, 2) = tau(2, 1) = mu * (grad_U(1, 2) + grad_U(2, 1));
    tau(2, 0) = tau(0, 2) = mu * (grad_U(2, 0) + grad_U(0, 2));
}

static const Scalar GAMMA = 1.4;
static const Scalar Pr = 0.72;
static const Scalar Rg = 287.7; // J / (Kg * K)
static const Scalar Cp = 3.5 * Rg;
static const Scalar Cv = Cp / GAMMA;

Scalar EOS(Scalar p, Scalar T)
{
    return p / (Rg * T);
}

Scalar Enthalpy(Scalar SH_CP, Scalar T)
{
    return SH_CP * T;
}

void CALC_Cell_Viscosity()
{
    for (auto& C : cell)
    {
        C.viscosity = C.rho / Re;
    }
}

void CALC_Face_Viscosity()
{
    for (auto& f : face)
    {
        f.viscosity = f.rho / Re;
    }
}

void CALC_Cell_Conductivity()
{
    for (auto& C : cell)
    {
        C.conductivity = C.specific_heat_p * C.viscosity / Pr;
    }
}

void CALC_Face_Conductivity()
{
    for (auto& f : face)
    {
        f.conductivity = f.specific_heat_p * f.viscosity / Pr;
    }
}

void CALC_Cell_SpecificHeat()
{
    for (auto &C : cell)
    {
        C.specific_heat_p = Cp;
        C.specific_heat_v = Cv;
    }
}

void CALC_Face_SpecificHeat()
{
    for (auto &f : face)
    {
        f.specific_heat_p = Cp;
        f.specific_heat_v = Cv;
    }
}
