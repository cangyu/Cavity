#ifndef SPATIAL_H
#define SPATIAL_H

#include "custom_type.h"

void INTERP_Face_Temperature_star();
void INTERP_Face_Temperature_next();
void INTERP_Face_MassFlux_star(Scalar TimeStep);
void INTERP_Face_Velocity_next();
void INTERP_Face_Pressure_next();
void INTERP_Face_snGrad_PressureCorrection();
void RECONST_Cell_Grad_PressureCorrection();
void INTERP_Node_Primitive();
void prepare_first_run();

#endif