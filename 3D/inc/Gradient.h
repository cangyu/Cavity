#ifndef GRADIENT_H
#define GRADIENT_H

void prepare_lsq();
void prepare_gg();
void prepare_TeC_operator();

void GRAD_Cell_Density();
void GRAD_Cell_Velocity();
void GRAD_Cell_Pressure();
void GRAD_Cell_Temperature();

void GRAD_Face_Density();
void GRAD_Face_Velocity();
void GRAD_Face_Pressure();
void GRAD_Face_Temperature();

void GRAD_Cell_Temperature_star();
void GRAD_Cell_Temperature_next();

Scalar GRAD_Cell_PressureCorrection();
void GRAD_Face_PressureCorrection();

void GRAD_Cell_Velocity_next();
void GRAD_Face_Velocity_next();

void GRAD_Cell_Pressure_next();

#endif
