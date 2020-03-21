#ifndef __POISSON_EQN_H__
#define __POISSON_EQN_H__

#include "custom_type.h"

void calcPressureCorrectionEquationCoef(Eigen::SparseMatrix<Scalar> &A);
void calcPressureCorrectionEquationRHS(Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &rhs);

#endif
