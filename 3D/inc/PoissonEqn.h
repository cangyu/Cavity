#ifndef POISSON_EQN_H
#define POISSON_EQN_H

#include "custom_type.h"

void PC_calcSteadyPart(SX_MAT &B);
void PC_updateDiagonalPart(SX_MAT *A, SX_VEC *base, SX_VEC *variation);
void PC_updateRHS(SX_VEC *b, Scalar dt);
void PC_prepareSolver(SX_MAT &A, SX_AMG &mg);

#endif
