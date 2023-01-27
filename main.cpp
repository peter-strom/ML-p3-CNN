#include "header.hpp"
/** @todo
   * Läsa in en bild från en fil, 6 siffor för varje färg ska adderas och lagras som ett värde.
   * varje värde lagras i en 2D array för att köras genom en 2x2 kernel.
   * 
**/



int main(void)
{


    std::vector<std::vector<double>> image;
    std::vector<std::vector<double>> train_x_in;
    char filename[] = "bitmaps/4_bw.bmp";
    read_file(image, filename);

    /*
    Neural_network numbrONE(4, 0, 0, 1, activation_option::TANH);
    numbrONE.print_network();
    //numbrONE.add_hidden_layers(2, 2, activation_option::TANH);
    numbrONE.print_network();
    numbrONE.set_training_data(train_x_in, train_yref_out);
    numbrONE.train(6000, 0.03);
    numbrONE.print_result();
    numbrONE.print_network(print_option::FULL);
    */
    while (1)
    {
        usleep(1000 * 20); // to prevent 100% cpu usage
    }

    return 0;
}