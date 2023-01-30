#include "convlayer.hpp"

int ConvLayer::import_from_bmp(const char *filename)
{

    std::ifstream file(filename, std::ios::binary);
    std::ifstream::pos_type filesize;
    char *memblock;
    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        filesize = file.tellg();
        memblock = new char[filesize];
        file.seekg(0, std::ios::beg);
        file.read(memblock, filesize);
        file.close();
        std::cout << "entire file in memory" << std::endl;
    }

    std::string s(memblock, filesize);

    // chek if file is 24bit BNP
    if (((s[0] & 0xff) != 0x42) || ((s[1] & 0xff) != 0x4d))
    {
        std::cout << " error: only bmp-files allowed!" << std::endl;
        return 1;
    }

    if ((s[28] & 0xff) != 0x18)
    {
        std::cout << " error: " << (s[28] & 0xff) << "-bit colors! 24-bit is required!" << std::endl;
        return 2;
    }
    // big endian(BE) = MSB on lowest memroy adress
    // get number of pixels
    //      lsb      msb
    //  x = 18 19 20 21
    //  y = 22 23 24 25
    //
    const uint32_t x_width = (s[18] | (s[19] << 8) | (s[20] << 16) | (s[21] << 24));
    const uint32_t y_height = (s[22] | (s[23] << 8) | (s[24] << 16) | (s[25] << 24));
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
            m_image[row][pixel] = (double)((uint8_t)s[i] + (uint8_t)s[i + 1] + (uint8_t)s[i + 2]) / 3;
            i += 3;
        }
        i += 3;
    }
    return 0;
}

void ConvLayer::print()
{
    for (std::size_t row = 0; row < m_image.size(); row++)
    {
        for (std::size_t pixel = 0; pixel < m_image[row].size(); pixel++)
        {
            std::cout << std::setfill('0') << std::setw(3) << std::dec << m_image[row][pixel] << " ";
        }
        std::cout << std::endl;
    }
}