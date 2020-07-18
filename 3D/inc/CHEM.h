#ifndef CHEM_H
#define CHEM_H

#include "custom_type.h"

Scalar Sutherland(Scalar T);
void Stokes(Scalar mu, const Tensor &grad_U, Tensor &tau);

#endif
