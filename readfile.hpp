#ifndef READFILE_HPP_
#define READFILE_HPP_

#include "definitions.hpp"
#include <fstream>

void read_file(std::vector<std::vector<double>> &train_x_in_buff,
               std::vector<std::vector<double>> &train_yref_out_buff,
               const char *filename);

#endif /* READFILE_HPP_ */