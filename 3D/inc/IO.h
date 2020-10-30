#ifndef IO_H
#define IO_H

#include <fstream>
#include "custom_type.h"

void read_mesh(std::istream &fin);
void write_data(std::ostream &out, int iter, Scalar t);
void read_data(std::istream &in, int &iter, Scalar &t);

#endif
