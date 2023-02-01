#include "convlayer.hpp"

/**
 * @brief 
 * imports a 24-bit bitmap file.
 * @param[in] filename path and file name to bitmap
 * @return int 0 if no errors
 */
int ConvLayer::import_image_from_bmp(const char *filename)
{

    std::ifstream file(filename, std::ios::binary);
    std::ifstream::pos_type filesize;
    std::string content;
    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        filesize = file.tellg();
        content.resize(filesize);
        file.seekg(0, std::ios::beg);
        file.read(&content[0], filesize);
        file.close();
    }

    //checks the image if its of type BNP
    if (((content[0]) != 0x42) || ((content[1]) != 0x4d))
    {
        return 1;
    }
    
    // chek if BNP is 24-bit
    if ((content[28]) != 0x18)
    {
        return 2;
    }

    // get number of pixels  (little-endian) 
    //      lsb      msb
    //  x = 18 19 20 21
    //  y = 22 23 24 25   
    const uint32_t x_width = (content[18] | (content[19] << 8) | (content[20] << 16) | (content[21] << 24));
    const uint32_t y_height = (content[22] | (content[23] << 8) | (content[24] << 16) | (content[25] << 24));

    // resize image vector
    m_image.resize(y_height, std::vector<double>(x_width, 0.0));

    // fill the container with "rgb-flatten" pixeldata
    // pixeldata begins at index 54
    // each pixel have 3 entries.
    // each row ends with 00 00 00
    // invert the row order
    std::size_t i = 54;
    for (int32_t row = m_image.size() - 1; row >= 0; row--)
    {
        for (std::size_t pixel = 0; pixel < m_image[row].size(); pixel++)
        {
            m_image[row][pixel] = (double)((uint8_t)content[i] + (uint8_t)content[i + 1] + (uint8_t)content[i + 2]) / 3;
            i += 3;
        }
        i += 3;
    }
    return 0;
}

/**
 * @brief 
 * imports an image from a vector
 * 
 * @param[in] image image as a vector container
 */
void ConvLayer::import_image_from_vector(std::vector<std::vector<double>> image)
{
    m_image = image;
}

/**
 * @brief prints content from the instance container memebers
 * 
 * @param[in] print_option IMAGE/KERNEL/OUTPUT
 */
void ConvLayer::print(PrintOption print_option)
{
    std::vector<std::vector<double>> *vector_ref = nullptr;
    if (print_option == PrintOption::IMAGE)
    {
        vector_ref = &m_image;
    }
    else if (print_option == PrintOption::KERNEL)
    {
        vector_ref = &m_kernel;
    }
    else if (print_option == PrintOption::OUTPUT)
    {
        vector_ref = &m_output;
    }

    for (std::size_t row = 0; row < (*vector_ref).size(); row++)
    {
        for (std::size_t pixel = 0; pixel < (*vector_ref)[row].size(); pixel++)
        {
            std::cout << std::setfill('0') << std::setw(3) << std::dec << (*vector_ref)[row][pixel] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief 
 * creats a padding around the image 
 */
void ConvLayer::zero_padding()
{
    for (std::size_t row = 0; row < m_image.size(); row++)
    {
        m_image[row].insert(m_image[row].begin(), 0.0);
        m_image[row].push_back(0.0);
    }
    m_image.insert(m_image.begin(), std::vector<double>(m_image[0].size()));
    m_image.push_back(std::vector<double>(m_image[0].size()));
}

/**
 * @brief 
 * creates a kernel with desired size (size*size)
 * Note random weights disabled.
 * @param[in] size the desired size (default = 0)
 */
void ConvLayer::init_kernel(uint8_t size)
{
    m_kernel.resize(size, std::vector<double>(size));
    for (size_t row = 0; row < m_kernel.size(); row++)
    {
        for (size_t pixel = 0; pixel < m_kernel[row].size(); pixel++)
        {
            // m_kernel[row][pixel] = (double)(std::rand()) / RAND_MAX;
            m_kernel[row][pixel] = 0.5;
        }
    }
}

/**
 * @brief 
 * performes the feedforward with the option of stride.
 * @param[in] stride higher values skips pixels and reduces details and output size (default = 0)
 */
void ConvLayer::convolute(uint8_t stride)
{
    stride = stride + 1;
    std::size_t output_size_height = ((m_image.size() - m_kernel.size()) / stride)+1;
    std::size_t output_size_width = ((m_image[0].size() - m_kernel.size()) / stride)+1;

    m_output.resize(output_size_height, std::vector<double>(output_size_width, 0));

    for (std::size_t row = 0; row < m_output.size(); row++)
    {
        for (std::size_t pixel = 0; pixel < m_output[row].size(); pixel++)
        {
            m_output[row][pixel] = conv_calc(row, pixel);
        }
    }
}

/**
 * @brief 
 * the detail extraction methood used in convolute for every pixel in the output container
 * @param[in] y_height output container y coordinates
 * @param[in] x_width output container x coordinates
 * @return double 
 */
uint8_t ConvLayer::conv_calc(size_t y_height, size_t x_width)
{
    double sum = 0;
    int i = 0;
    for (std::size_t y = 0; y < m_kernel.size(); y++)
    {
        for (std::size_t x = 0; x < m_kernel[y].size(); x++)
        {
            sum += m_image[y_height + y][x_width + x] * m_kernel[y][x];
            i++;
        }
    }

    sum = i > 0 ? sum / i : 0;

    return uint8_t(sum);
}

/**
 * @brief 
 * returns the output image for use in the image import for the next layer
 * @return std::vector<std::vector<double>> 
 */
std::vector<std::vector<double>> ConvLayer::get_output()
{
    return m_output;
}

/**
 * @brief 
 * usedd for reduce unnecessery data to reduce image size
 * @param[in] pooling_option method for the calculation MAX(default)/AVERAGE
 * @param[in] pooling_size the size of the pooling
 */
void ConvLayer::pooling(PoolingOption pooling_option, size_t pooling_size)
{
    init_kernel(pooling_size);
    m_output.resize(m_image.size() / pooling_size, std::vector<double>(m_image[0].size() / pooling_size, 0));

    for (std::size_t row = 0; row < m_output.size(); row++)
    {
        for (std::size_t pixel = 0; pixel < m_output[row].size(); pixel++)
        {
            m_output[row][pixel] = pool(pooling_option, pooling_size, row, pixel);
        }
    }
}

/**
 * @brief 
 * used in pooling to calculate a value for every pixel in the output container
 * @param pooling_option method for the calculation MAX/AVERAGE
 * @param pooling_size the size of the pooling
 * @param[in] y_height output container y coordinates
 * @param[in] x_width output container x coordinates
 * @return uint8_t 
 */
uint8_t ConvLayer::pool(PoolingOption pooling_option, size_t pooling_size, size_t y_height, size_t x_width)
{
    double sum = 0;
    int i = 0;
    for (std::size_t y = 0; y < m_kernel.size(); y++)
    {
        for (std::size_t x = 0; x < m_kernel[y].size(); x++)
        {
            double val = m_image[y_height * pooling_size + y][x_width * pooling_size + x];
            if (pooling_option == PoolingOption::MAX)
            {
                sum = sum < val ? val : sum;
            }
            else if (pooling_option == PoolingOption::AVERAGE)
            {
                sum += val;
            }
            i++;
        }
    }

    if (pooling_option == PoolingOption::AVERAGE)
    {
        sum = i > 0 ? (sum / i) : 0;
    }

    return uint8_t(sum);
}

/**
 * @brief 
 * returns a flattened version of the image that can be used as training data for the neural net.
 * @return std::vector<double> 
 */
std::vector<double> ConvLayer::get_flatend_output()
{
    //std::size_t max_size = m_output.size() * m_output[0].size();
    std::vector<double> flat_output;
    
    int i = 0;
    for (std::size_t row = 0; row < m_output.size(); row++)
    {
        for (std::size_t pixel = 0; pixel < m_output[row].size(); pixel++)
        {
            flat_output.push_back(m_output[row][pixel]);
            i++;
        }
    }
    return flat_output;
}