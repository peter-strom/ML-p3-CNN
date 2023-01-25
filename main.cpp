#include "header.hpp"
/** @todo
   * Kunna välja lärhastighet samt epoker. OK
   * kunna välja antalet noder i in och utgångslagret. OK
   * Kunna välja antalet dolda lager och antalet noder i de dolda lagren. OK

   * Träningsdatan ska kunna läsas in från en textfil via en array. OK
   * träning ska ske direkt vid uppstart och knapptryckning ska inte
     medföra en prediktion förens träningen är slutförd. OK
   *
**/



int main(void)
{


    std::vector<std::vector<double>> train_yref_out;
    std::vector<std::vector<double>> train_x_in;
    char filename[] = "test.bmp";
    read_file(train_x_in, train_yref_out, filename);

    Neural_network numbrONE(4, 0, 0, 1, activation_option::TANH);
    numbrONE.print_network();
    //numbrONE.add_hidden_layers(2, 2, activation_option::TANH);
    numbrONE.print_network();
    numbrONE.set_training_data(train_x_in, train_yref_out);
    numbrONE.train(6000, 0.03);
    numbrONE.print_result();
    numbrONE.print_network(print_option::FULL);
    while (1)
    {
        usleep(1000 * 20); // to prevent 100% cpu usage
    }

    return 0;
}