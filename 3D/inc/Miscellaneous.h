#ifndef MISCELLANEOUS_H
#define MISCELLANEOUS_H

#include <ctime>
#include <string>
#include "../inc/custom_type.h"

double duration(const clock_t &startTime, const clock_t &endTime);
std::string time_stamp_str();
void interp_nodal_primitive_var();
void calc_noc_vec(int opt, const Vector &d, const Vector &S, Vector &E, Vector &T);

Scalar double_dot(const Tensor &A, const Tensor &B);

#endif
