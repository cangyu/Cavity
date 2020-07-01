#ifndef IO_H
#define IO_H

#include <string>
#include <ostream>
#include "../inc/custom_type.h"

void read_fluent_mesh(const std::string &MESH_PATH, std::ostream &LOG_OUT);
void write_tec_grid(const std::string &fn, const std::string &title);
void write_tec_solution(const std::string &fn, double t, const std::string &title);
void record_computation_domain(const std::string &prefix, int n, Scalar t);
void read_tec_solution(const std::string &fn);

#endif
