#ifndef __IO_H__
#define __IO_H__

#include <string>

void readMESH(const std::string &MESH_PATH);
void writeTECPLOT_Nodal(const std::string &fn, const std::string &title);
void writeTECPLOT_CellCentered(const std::string &fn, const std::string &title);
void readTECPLOT_Nodal(const std::string &fn);
void readTECPLOT_CellCentered(const std::string &fn);

#endif
