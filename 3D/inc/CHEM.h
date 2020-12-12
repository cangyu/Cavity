#ifndef CHEM_H
#define CHEM_H

#include "custom_type.h"

Scalar Sutherland(Scalar T);
void Stokes(Scalar mu, const Tensor &grad_U, Tensor &tau);
Scalar EOS(Scalar p, Scalar T);
Scalar Cp();
Scalar Cv();
Scalar Conductivity(Scalar specific_heat_p, Scalar viscosity);

#endif
