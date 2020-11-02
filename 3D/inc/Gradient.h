#ifndef GRADIENT_H
#define GRADIENT_H

void calc_least_square_coefficient_matrix();

void calc_cell_primitive_gradient();
void calc_cell_pressure_correction_gradient();
void calc_face_primitive_gradient();
void calc_face_pressure_correction_gradient();

#endif
