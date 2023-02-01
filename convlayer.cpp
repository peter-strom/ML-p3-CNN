#include "convlayer.hpp"

int ConvLayer::import_from_bmp(const char *filename)
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
        std::cout << "entire file in memory" << std::endl;
    }

    // chek if file is 24bit BNP
    if (((content[0] & 0xff) != 0x42) || ((content[1] & 0xff) != 0x4d))
    {
        std::cout << " error: only bmp-files allowed!" << std::endl;
        return 1;
    }

    if ((content[28] & 0xff) != 0x18)
    {
        std::cout << " error: " << (content[28] & 0xff) << "-bit colors! 24-bit is required!" << std::endl;
        return 2;
    }
    // big endian(BE) = MSB on lowest memroy adress
    // get number of pixels
    //      lsb      msb
    //  x = 18 19 20 21
    //  y = 22 23 24 25
    //
    const uint32_t x_width = (content[18] | (content[19] << 8) | (content[20] << 16) | (content[21] << 24));
    const uint32_t y_height = (content[22] | (content[23] << 8) | (content[24] << 16) | (content[25] << 24));
    std::cout << "width :" << x_width << " heigth :" << y_height << std::endl;

    // resize image vector
    m_image.resize(y_height, std::vector<double>(x_width, 0.0));

    // fill the 2D-vector with flatten pixeldata
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

void ConvLayer::print(print_option print_option)
{
    std::vector<std::vector<double>> *vector_ref = nullptr;
     if (print_option == print_option::IMAGE)
    {
        vector_ref = &m_image;
    }
    if (print_option == print_option::KERNEL)
    {
        vector_ref = &m_kernel;
    }
    if (print_option == print_option::OUTPUT)
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

void ConvLayer::zero_padd()
{
    for (std::size_t row = 0; row < m_image.size(); row++)
    {
        m_image[row].insert(m_image[row].begin(), 0.0);
        m_image[row].push_back(0.0);
    }
    m_image.insert(m_image.begin(), std::vector<double>(m_image[0].size()));
    m_image.push_back(std::vector<double>(m_image[0].size()));
}

void ConvLayer::init_kernel(uint8_t size)
{
    m_kernel.resize(size, std::vector<double>(size));
    for (size_t row = 0; row < m_kernel.size(); row++)
    {
        for (size_t pixel = 0; pixel < m_kernel[row].size(); pixel++)
        {
            //m_kernel[row][pixel] = (double)(std::rand()) / RAND_MAX;
            m_kernel[row][pixel] = 0.5;
        }
    }
}

void ConvLayer::feedforward(uint8_t stride)
{
    uint8_t margin = stride + m_kernel.size();
    m_output.resize(m_image.size()- margin, std::vector<double>(m_image[0].size()-margin, 0));

    for (size_t row = 0; row < m_output.size(); row++)
    {
        for (size_t pixel = 0; pixel < m_output[row].size(); pixel++)
        {
            m_output[row][pixel] = convolute(row,pixel);
        }
        
    }
    
   
}

double ConvLayer::convolute(size_t y_height, size_t x_width)
{
    double sum=0;
    int i = 0;
    for (size_t y = 0; y < m_kernel.size() ; y++)
    {
        for (size_t x = 0; x < m_kernel[y].size(); x++)
        {
           sum += m_image[y_height+y][x_width+x] * m_kernel[y][x];
           i++;
        }
    }
    return uint8_t(sum / i);
}