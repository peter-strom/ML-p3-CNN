#ifndef CONVLAYER_HPP_
#define CONVLAYER_HPP_
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <iomanip>

class ConvLayer
{
public:
    ConvLayer(void) {}
    ~ConvLayer() {}
    int import_from_bmp(const char *filename);
    void print();

private:
    std::vector<std::vector<double>> m_image;
    std::vector<std::vector<double>> m_kernel;
    std::vector<std::vector<double>> m_output;
};

#endif /* CONVLAYER_HPP_ */