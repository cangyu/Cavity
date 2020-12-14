#ifndef GRADIENT_H
#define GRADIENT_H

void prepare_lsq();
void prepare_gg();
void prepare_TeC_operator();

void calc_cell_primitive_gradient();
void calc_face_primitive_gradient();

void GRAD_Cell_Temperature_star();
void GRAD_Cell_Temperature_next();
Scalar GRAD_Cell_PressureCorrection();
void GRAD_Face_PressureCorrection();
void GRAD_Cell_Velocity_next();
void GRAD_Face_Velocity_next();
void GRAD_Cell_Pressure_next();

#endif
