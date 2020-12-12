#include <cmath>
#include "../inc/CHEM.h"

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

Scalar EOS(Scalar p, Scalar T)
{
    return p / (Rg * T);
}

Scalar Cp()
{
    return 3.5 * Rg;
}

Scalar Cv()
{
    return Cp() / GAMMA;
}

Scalar Conductivity(Scalar specific_heat_p, Scalar viscosity)
{
    return specific_heat_p * viscosity / Pr;
}
