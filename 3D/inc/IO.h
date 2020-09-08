#ifndef IO_H
#define IO_H

#include <fstream>
#include "../inc/custom_type.h"

void read_mesh(std::istream &fin);
void write_data(std::ostream &f_out, int iter, Scalar t);
void read_data(std::istream &fin, int &iter, Scalar &t);

#endif
