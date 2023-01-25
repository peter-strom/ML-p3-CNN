#ifndef Neural_network_HPP_
#define Neural_network_HPP_
#include "definitions.hpp"
#include "Dense_layer.hpp"

/**
 * @brief Class for neural network and output layers.
 * parts of a neural network.
 * 
 * @param[in] num_inputs number of input signals (training data)
 * @param[in] num_hidden_layers number of hidden layers
 * @param[in] num_hidden_nodes number of nodes per hidden layer
 * @param[in] num_outputs number of output signals (training data)
 * @param[in] ao option to select an activation method
 */
class Neural_network
{
protected: 
    std::vector<Dense_layer> hidden_layers_;     
    Dense_layer output_layer_;                   
    std::vector<std::vector<double>> train_x_in_;  
    std::vector<std::vector<double>> train_yref_out_; 
    std::vector<std::size_t> train_order_;  

    void check_training_data_size(void);
    void init_training_order(void);
    void feedforward(const std::vector<double> &input);
    void backpropagate(const std::vector<double> &reference);
    void optimize(const std::vector<double> &input,
                  const double learning_rate);
    void randomize_training_order(void);

public:
    Neural_network(void) {}
    Neural_network(const std::size_t num_inputs,
                   const std::size_t num_hidden_layers,
                   const std::size_t num_hidden_nodes,
                   const std::size_t num_outputs,
                   const activation_option ao = activation_option::TANH);
    ~Neural_network(void) { this->clear(); }
    void init(const std::size_t num_inputs,
              std::size_t num_hidden_layers,
              std::size_t num_hidden_nodes,
              const std::size_t num_outputs,
              const activation_option ao = activation_option::TANH);
    void add_hidden_layers(std::size_t num_hidden_layers,
                           std::size_t num_hidden_nodes,
                           const activation_option ao = activation_option::TANH);
    void clear(void);
    void set_training_data(const std::vector<std::vector<double>> &train_in,
                           const std::vector<std::vector<double>> &train_out);
    void train(const std::size_t num_epochs,
               const double learning_rate);
    const std::vector<double> &predict(const std::vector<double> &input);
    void print_result(const std::size_t num_decimals = 1,
                      std::ostream &ostream = std::cout);
    void print_network(print_option po = print_option::LITE, 
                       std::ostream &ostream = std::cout);
};

#endif