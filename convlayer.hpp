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
    enum class print_option
    {
        IMAGE,
        KERNEL,
        OUTPUT
    };

    enum class pooling_option
    {
        AVERAGE,
        MAX
    };

    ConvLayer(void) {}
    ~ConvLayer() {}
    int import_image_from_bmp(const char *filename);
    void import_image_from_vector(std::vector<std::vector<double>> image);
    void print(print_option print_option);
    void zero_padd();
    void init_kernel(uint8_t size);
    void feedforward(uint8_t stride = 0);
    std::vector<std::vector<double>> get_output();
    void pooling(pooling_option pooling_option = pooling_option::MAX, size_t pooling_size = 2);
    

private:
    std::vector<std::vector<double>> m_image;
    std::vector<std::vector<double>> m_kernel;
    std::vector<std::vector<double>> m_output;
    double convolute(size_t y_height, size_t x_width);
    uint8_t pool(pooling_option pooling_option, size_t pooling_size, size_t y_height, size_t x_width);
};

#endif /* CONVLAYER_HPP_ */