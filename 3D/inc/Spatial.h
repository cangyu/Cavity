#ifndef SPATIAL_H
#define SPATIAL_H

#include "custom_type.h"

void interp_nodal_primitive_var();

void calc_face_primitive_var();

void calc_face_viscous_shear_stress();

void calc_cell_viscous_shear_stress_next();

void calc_face_viscous_shear_stress_next();

void calc_face_pressure_correction();

void calc_face_temperature_next();

#endif
