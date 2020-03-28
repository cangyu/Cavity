#ifndef __IO_H__
#define __IO_H__

#include <string>
#include <ostream>

void readMESH(const std::string &MESH_PATH, std::ostream &LOG_OUT);
void writeTECPLOT_Nodal(const std::string &fn, const std::string &title, const std::string &text, double t_sol);
void writeTECPLOT_Centered(const std::string &fn, const std::string &title, const std::string &text, double t_sol);
void readTECPLOT_Nodal(const std::string &fn);
void readTECPLOT_Centered(const std::string &fn);

#endif
