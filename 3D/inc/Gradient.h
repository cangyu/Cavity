#ifndef GRADIENT_H
#define GRADIENT_H

void prepare_lsq();

void prepare_gg();

void calc_cell_primitive_gradient();

void calc_cell_pressure_correction_gradient();

void calc_face_primitive_gradient();

void calc_face_pressure_correction_gradient();

#endif
