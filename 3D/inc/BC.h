#ifndef BC_H
#define BC_H

#include <string>

std::string get_bc_name(int bc);

bool bc_is_wall(int bc);
bool bc_is_symmetry(int bc);
bool bc_is_inlet(int bc);
bool bc_is_outlet(int bc);

void BC_TABLE();
void set_bc_of_primitive_var();
void set_bc_of_pressure_correction();
void set_bc_of_conservative_var();

#endif
