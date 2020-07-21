#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "custom_type.h"

Scalar calcTimeStep();
void interp_nodal_primitive_var();
void prepare_next_run();
void ForwardEuler(Scalar TimeStep);

#endif
