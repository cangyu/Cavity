#ifndef GRADIENT_H
#define GRADIENT_H

void prepare_lsq();

void prepare_gg();

void prepare_gpc_rm();

void calc_cell_primitive_gradient();


void calc_face_primitive_gradient();

Scalar GRAD_Cell_PressureCorrection();
void GRAD_Face_PressureCorrection();
void GRAD_Cell_Velocity_next();
void GRAD_Cell_Pressure_next();

#endif
