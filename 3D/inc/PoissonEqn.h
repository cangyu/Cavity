#ifndef __POISSON_EQN_H__
#define __POISSON_EQN_H__

#include "custom_type.h"

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A);
void calcPressureCorrectionEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs);

void calcPressureCorrectionEquationCoef(SX_MAT &B);
void calcPressureCorrectionEquationRHS(SX_VEC &rhs);
void prepare_dp_solver(SX_MAT &A, SX_AMG &mg);

#endif