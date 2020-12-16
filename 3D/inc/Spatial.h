#ifndef SPATIAL_H
#define SPATIAL_H

#include "custom_type.h"

/* Node */
/// @(n)
void INTERP_Node_Velocity();
void INTERP_Node_Pressure();
void INTERP_Node_Temperature();
void INTERP_Node_Primitive();

/* Face */
/// @(n)
void INTERP_BoundaryFace_Velocity();
void INTERP_BoundaryFace_Pressure();
void INTERP_BoundaryFace_Temperature();
void CALC_Face_ViscousShearStress();

/// @(*)
void INTERP_Face_Temperature_star();
void INTERP_Face_MassFlux_star(Scalar TimeStep);
void INTERP_Face_snGrad_PressureCorrection();

/// @(m+1)
void INTERP_Face_Velocity_next();
void INTERP_Face_Pressure_next();
void INTERP_Face_Temperature_next();
void CALC_Face_ViscousShearStress_next();

/* Cell */
/// @(n)
void CALC_Cell_ViscousShearStress();

/// @(*)
void RECONST_Cell_Grad_PressureCorrection();

/// @(m+1)
void CALC_Cell_ViscousShearStress_next();

#endif
