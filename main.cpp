#include "main.hpp"

int main(void)
{
    char filename[] = "bitmaps/4_bw.bmp";
    ConvLayer image;
    int s8ret = image.import_image_from_bmp(filename);
    if(s8ret != 0)
    {
        std::cout << "import error: " << s8ret << std::endl;
        return 0;
    }
    std::cout << "the imported bitmap:" << std::endl;
    image.print(ConvLayer::PrintOption::IMAGE);
    image.zero_padding();
    std::cout << "the image after zero_padding():" << std::endl;
    image.print(ConvLayer::PrintOption::IMAGE);
    image.init_kernel(3);
    std::cout << "the kernel:" << std::endl;
    image.print(ConvLayer::PrintOption::KERNEL);
    image.convolute(0);
    std::cout << "the output:" << std::endl;
    image.print(ConvLayer::PrintOption::OUTPUT);
    ConvLayer pooling1;
    pooling1.import_image_from_vector(image.get_output());
    pooling1.pooling();
    std::cout << "after max pooling 2x2:" << std::endl;
    pooling1.print(ConvLayer::PrintOption::OUTPUT);
    ConvLayer pooling2;
    pooling2.import_image_from_vector(image.get_output());
    pooling2.pooling(ConvLayer::PoolingOption::AVERAGE);
    std::cout << "after average pooling 2x2:" << std::endl;
    pooling2.print(ConvLayer::PrintOption::OUTPUT);

    std::vector<std::vector<double>> train_x_in ;
    train_x_in.push_back(pooling1.get_flatend_output());
    std::vector<std::vector<double>> train_yref_out = {{0,1,0,0}};

    NeuralNetwork nnOne(7*7, 0, 0, 4, activation_option::TANH);
    nnOne.add_hidden_layers(3, 10, activation_option::TANH);
    std::cout << "the neural net:" << std::endl;
    nnOne.print_network();
    nnOne.set_training_data(train_x_in, train_yref_out);
    nnOne.train(200, 0.03);
    std::cout << "results afer 200 epochs and a learning rate of 0.03 :" << std::endl;
    nnOne.print_result();
    //nnOne.print_network(print_option::FULL);
    
    while (1)
    {
        usleep(1000 * 20); // to prevent 100% cpu usage
    }

    return 0;
}