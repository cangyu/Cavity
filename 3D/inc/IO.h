#ifndef __IO_H__
#define __IO_H__

#include <string>
#include <ostream>

void read_fluent_mesh(const std::string &MESH_PATH, std::ostream &LOG_OUT);
void write_tec_grid(const std::string &fn, int type, const std::string &title);
void write_tec_solution(const std::string &fn, double t, const std::string &title);
void read_tec_solution(const std::string &fn);

#endif
