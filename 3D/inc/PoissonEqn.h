#ifndef POISSON_EQN_H
#define POISSON_EQN_H

#include "custom_type.h"

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A);
void calcPressureCorrectionEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs, Scalar dt);

void calcPressureCorrectionEquationCoef(SX_MAT &B);
void calcPressureCorrectionEquationRHS(SX_VEC &rhs, Scalar dt);

void calcPressureCorrectionEquationCoef(SX_MAT &B, const std::vector<Scalar> &ppe_diag);

void calcPressureCorrectionEquationRHS_compressible(SX_VEC &rhs, double dt);

void prepare_dp_solver(SX_MAT &A, SX_AMG &mg);

void calcPressureCorrectionEquationCoef(SX_MAT &B, const std::vector<Scalar> &ppe_diag);

#endif
