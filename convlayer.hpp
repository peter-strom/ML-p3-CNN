#ifndef CONVLAYER_HPP_
#define CONVLAYER_HPP_
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <iomanip>

#define SIZE_OF_HEADER 54

/**
 * @brief class for Convolutional layers.
 * Used for extract the details from images.
 *
 */
class ConvLayer
{

public:
    enum class PrintOption
    {
        IMAGE,
        KERNEL,
        OUTPUT
    };

    enum class PoolingOption
    {
        AVERAGE,
        MAX
    };

    ConvLayer(void) {}
    ~ConvLayer() {}
    int import_image_from_bmp(const char *filename);
    void import_image_from_vector(std::vector<std::vector<double>> image);
    void print(PrintOption print_option);
    void zero_padding();
    void init_kernel(uint8_t size = 3);
    void convolute(uint8_t stride = 0);
    std::vector<std::vector<double>> get_output();
    void pooling(PoolingOption pooling_option = PoolingOption::MAX, size_t pooling_size = 2);
    std::vector<double> get_flatend_output();

private:
    std::vector<std::vector<double>> m_image;
    std::vector<std::vector<double>> m_kernel;
    std::vector<std::vector<double>> m_output;
    uint8_t conv_calc(size_t y_height, size_t x_width);
    uint8_t pool(PoolingOption pooling_option, size_t pooling_size, size_t y_height, size_t x_width);
};

#endif /* CONVLAYER_HPP_ */