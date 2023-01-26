#ifndef READFILE_HPP_
#define READFILE_HPP_

#include "definitions.hpp"
#include <fstream>

int read_file(std::vector<std::vector<double>> &image,
               const char *filename);
void ToHex(const std::string& s, bool upper_case);

#endif /* READFILE_HPP_ */