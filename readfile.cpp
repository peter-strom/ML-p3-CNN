#include "readfile.hpp"
/**
 * @brief read training data from file and store the data in vectors
 *
 * @param[out] train_x_in_buff vector buffert for indata
 * @param[out] train_yref_out_buff vector puffert for outdata
 * @param[in] filename filename of the trainingdata
 */
void read_file(std::vector<std::vector<double>> &train_x_in_buff,
               std::vector<std::vector<double>> &train_yref_out_buff,
               const char *filename)
{

    std::ifstream infile(filename);

    double x1, x2, x3, x4, y1;
    int i = 0;
    while (infile >> x1 >> x2 >> x3 >> x4 >> y1)
    {
        train_x_in_buff.push_back({x1, x2, x3, x4});
        train_yref_out_buff.push_back({y1});
        i++;
    }
    std::cout << i << " rows inserted from file \n";
}