#ifndef CHEM_H
#define CHEM_H

#include "custom_type.h"

Scalar Sutherland(Scalar T);
void Stokes(Scalar mu, const Tensor &grad_U, Tensor &tau);
Scalar EOS(Scalar p, Scalar T);
Scalar Enthalpy(Scalar SH_CP, Scalar T);

/* Face */
void CALC_Face_Viscosity();
void CALC_Face_Conductivity();
void CALC_Face_SpecificHeat();

/* Cell */
void CALC_Cell_Viscosity();
void CALC_Cell_Conductivity();
void CALC_Cell_SpecificHeat();

#endif
