#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>

enum class activation_option
{
    RELU,
    TANH
};
enum class print_option
{
    LITE,
    FULL
};

/**
 * @brief Class for hidden layers and output layers.
 * parts of a neural network.
 *
 */
class Dense_layer
{
public:    
    std::vector<double> output;
    std::vector<double> error;
    std::vector<double> bias;
    std::vector<std::vector<double>> weights;
    activation_option ao;
    Dense_layer(void) {}
    Dense_layer(const std::size_t num_nodes,
                const std::size_t num_weights);
    ~Dense_layer();
    std::size_t num_nodes(void) const;
    std::size_t num_weights(void) const;
    void set_activation(const activation_option ao = activation_option::TANH);
    void clear(void);
    void resize(const std::size_t num_nodes,
                const std::size_t num_weights);
    void feedforward(const std::vector<double> &input);
    void backpropagate(const std::vector<double> &reference);
    void backpropagate(const Dense_layer &next_layer);
    void optimize(const std::vector<double> &input,
                  const double learning_rate);
    void print(print_option po = print_option::LITE, std::ostream &ostream = std::cout);

private: 
    inline double get_random(void);
    inline double activation(const double sum);
    inline double delta_activation(const double output);
    double get_rounded(const double number,
                       const double threshold = 0.001);
};

#endif /* DENSE_LAYER_HPP_ */