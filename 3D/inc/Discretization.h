#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "custom_type.h"

void interp_nodal_primitive_var();
void prepare_first_run();
void ForwardEuler(Scalar TimeStep);

#endif
