#ifndef GRADIENT_H
#define GRADIENT_H

#include "custom_type.h"

void prepare_lsq();

void prepare_gg();

void prepare_gpc_rm();

void calc_cell_primitive_gradient();

void calc_cell_velocity_gradient_next();

void calc_cell_pressure_gradient_next();

void calc_cell_temperature_gradient_next();

void calc_cell_density_gradient_next();

Scalar calc_cell_pressure_correction_gradient();

void calc_face_primitive_gradient();

void calc_face_pressure_correction_gradient();

void calc_face_temperature_gradient_next();

#endif
