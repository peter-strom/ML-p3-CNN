#include "Neural_network.hpp"

/**
 * @brief Construct a new Neural Network object
 *
 * @param[in] num_inputs number of input signals (training data)
 * @param[in] num_hidden_layers number of hidden layers
 * @param[in] num_hidden_nodes number of nodes per hidden layer
 * @param[in] num_outputs number of output signals (training data)
 * @param[in] ao option to select an activation method
 */
Neural_network::Neural_network(const std::size_t num_inputs,
                               const std::size_t num_hidden_layers,
                               const std::size_t num_hidden_nodes,
                               const std::size_t num_outputs,
                               const activation_option ao)
{
    this->init(num_inputs, num_hidden_layers, num_hidden_nodes, num_outputs, ao);
}

/**
 * @brief initiates a neural network with chosen number of nodes for each layer
 *        and chosen activation function.
 *
 * @details sets number of inputs, hidden layers, hidden nodes and output nodes,
 *          if 0 hidden layers are chosen, 1 is created. If 0 nodes are chosen,
 *           same number as numbers of inputs are created.
 * @param[in] num_inputs number of input signals (training data)
 * @param[in] num_hidden_layers number of hidden layers
 * @param[in] num_hidden_nodes number of nodes per hidden layer
 * @param[in] num_outputs number of output signals (training data)
 * @param[in] af option to select an activation method
 */
void Neural_network::init(const std::size_t num_inputs,
                          std::size_t num_hidden_layers,
                          std::size_t num_hidden_nodes,
                          const std::size_t num_outputs,
                          const activation_option ao)
{
    if (num_hidden_layers == 0)
    {
        num_hidden_layers = 1;
    }
    if (num_hidden_nodes == 0)
    {
        num_hidden_nodes = num_inputs;
    }
    this->output_layer_.set_activation(ao);
    this->output_layer_.resize(num_outputs, num_hidden_nodes);
    this->hidden_layers_.resize(num_hidden_layers);

    for (size_t i = 0; i < num_hidden_layers; i++)
    {
        this->hidden_layers_[i].set_activation(ao);
        if (i == 0)
        {
            this->hidden_layers_[i].resize(num_hidden_nodes, num_inputs);
        }
        else
        {
            this->hidden_layers_[i].resize(num_hidden_nodes, num_hidden_nodes);
        }
    }
}

/**
 * @brief function to add additional layers
 *
 * @param[in] num_hidden_layers number of hidden layers
 * @param[in] num_hidden_nodes number of nodes per hidden layer
 * @param[in] ao option to select an activation method
 */
void Neural_network::add_hidden_layers(std::size_t num_hidden_layers,
                                       std::size_t num_hidden_nodes,
                                       const activation_option ao)
{
    std::size_t old_size = this->hidden_layers_.size();
    std::size_t last_layer_nodes = this->hidden_layers_[old_size - 1].num_nodes();

    if (num_hidden_layers == 0)
    {
        num_hidden_layers = 1;
    }
    if (num_hidden_nodes == 0)
    {
        num_hidden_nodes = last_layer_nodes;
    }

    std::size_t new_size = old_size + num_hidden_layers;
    this->hidden_layers_.resize(new_size);

    for (size_t i = old_size; i < new_size; i++)
    {
        this->hidden_layers_[i].set_activation(ao);
        if (i == old_size)
        {
            this->hidden_layers_[i].resize(num_hidden_nodes, last_layer_nodes);
        }
        else
        {
            this->hidden_layers_[i].resize(num_hidden_nodes, num_hidden_nodes);
        }
    }

    std::size_t output_layer_nodes = this->output_layer_.num_nodes();
    this->output_layer_.clear();
    this->output_layer_.resize(output_layer_nodes, num_hidden_nodes);
}

/**
 * @brief initiates training data
 * 
 * @param[in] train_x_in training input data 
 * @param[in] train_yref_out traingin output data (target)
 */
void Neural_network::set_training_data(const std::vector<std::vector<double>> &train_x_in,
                                       const std::vector<std::vector<double>> &train_yref_out)
{
    this->train_x_in_ = train_x_in;
    this->train_yref_out_ = train_yref_out;
    this->check_training_data_size();
    this->init_training_order();
}

/**
 * @brief function that handles the training of the neural network.
 *         
 * 
 * @param[in] num_epochs number of training epochs
 * @param[in] learning_rate amount of error adjustment used for optimisation
 */
void Neural_network::train(const std::size_t num_epochs,
                           const double learning_rate)
{
    for (std::size_t i = 0; i < num_epochs; i++)
    {
        this->randomize_training_order();
        for (std::size_t j = 0; j < this->train_order_.size(); j++)
        {
            const auto index = this->train_order_[j];
            const auto &input = this->train_x_in_[index];
            const auto &reference = this->train_yref_out_[index];

            this->feedforward(input);
            this->backpropagate(reference);
            this->optimize(input, learning_rate);
        }
    }
}

/**
 * @brief compairs the size of input and output training data
 * and fix variations betwen them
 */
void Neural_network::check_training_data_size(void)
{

    if (this->train_x_in_.size() < this->train_yref_out_.size())
    {
        this->train_yref_out_.resize(this->train_x_in_.size());
    }
    else if (this->train_x_in_.size() > this->train_yref_out_.size())
    {
        this->train_x_in_.resize(this->train_yref_out_.size());
    }
}

/**
 * @brief initiates the training order vector and sets it to the size of train_x_in 
 * 
 */
void Neural_network::init_training_order(void)
{
    this->train_order_.resize(this->train_x_in_.size());
    for (std::size_t i = 0; i < this->train_order_.size(); i++)
    {
        this->train_order_[i] = i;
    }
}

/**
 * @brief calculates output for all nodes in the entire neural network
 * 
 * @param[in] input input signals 
 */
void Neural_network::feedforward(const std::vector<double> &input)
{
    for (size_t i = 0; i < this->hidden_layers_.size(); i++)
    {
        if (i == 0)
        {
            this->hidden_layers_[i].feedforward(input);
        }
        else
        {
            this->hidden_layers_[i].feedforward(this->hidden_layers_[i - 1].output);
        }
    }
    this->output_layer_.feedforward(this->hidden_layers_[this->hidden_layers_.size() - 1].output);
}

/**
 * @brief calculates the error for all nodes in the entire neural network
 * @details running backwards hidden_layer_0 <- hidden_layer_1 <- output_layer
 * 
 * @param[in] reference training data (y_ref, target)
 */
void Neural_network::backpropagate(const std::vector<double> &reference)
{
    this->output_layer_.backpropagate(reference);

    for (int i = this->hidden_layers_.size() - 1; i >= 0; i--)
    {

        if (i == (int)this->hidden_layers_.size() - 1)
        {
            this->hidden_layers_[i].backpropagate(this->output_layer_);
        }
        else
        {

            this->hidden_layers_[i].backpropagate(this->hidden_layers_[i + 1]);
        }
    }
}

/**
 * @brief calculates new bias and weights for all nodes in the entire neural network
 * 
 * @param[in] input training input data
 * @param[in] learning_rate amount of error adjustment
 */
void Neural_network::optimize(const std::vector<double> &input,
                              const double learning_rate)
{
    for (size_t i = 0; i < this->hidden_layers_.size(); i++)
    {
        if (i == 0)
        {
            this->hidden_layers_[i].optimize(input, learning_rate);
        }
        else
        {
            this->hidden_layers_[i].optimize(this->hidden_layers_[i - 1].output, learning_rate);
        }
    }
    this->output_layer_.optimize(this->hidden_layers_[this->hidden_layers_.size() - 1].output, learning_rate);
}

/**
 * @brief randomizes the training order to prevent overfitting.
 * 
 */
void Neural_network::randomize_training_order(void)
{
    for (std::size_t i = 0; i < this->train_order_.size(); ++i)
    {
        const auto r = std::rand() % this->train_order_.size();
        const auto temp = this->train_order_[i];
        this->train_order_[i] = this->train_order_[r];
        this->train_order_[r] = temp;
    }
}

/**
 * @brief function to reset all the values in the neural network 
 * 
 */
void Neural_network::clear(void)
{
    for (size_t i = 0; i < this->hidden_layers_.size(); i++)
    {
        this->hidden_layers_[i].clear();
    }
    this->hidden_layers_.clear();
    this->output_layer_.clear();
    this->train_x_in_.clear();
    this->train_yref_out_.clear();
    this->train_order_.clear();
}

/**
 * @brief runs inputs signals through the trained neural network and returns the answer
 * 
 * @param[in] input input signals
 * @return const std::vector<double>& 
 */
const std::vector<double> &Neural_network::predict(const std::vector<double> &input)
{
    this->feedforward(input);
    return this->output_layer_.output;
}

/**
 * @brief function to print the results after succesfully training and running the 
*         neural network
 * 
 * @param[in] num_decimals sets the number of decimals for the result print.
 * @param[in] ostream chosen output stream
 */
void Neural_network::print_result(const std::size_t num_decimals,
                                  std::ostream &ostream)
{
    if (this->train_x_in_.size() == 0)
        return;
    ostream << "-=( training result )=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n";
    for (size_t i = 0; i < this->train_x_in_.size(); i++)
    {
        ostream << "Input: ";
        for (size_t j = 0; j < this->train_x_in_[i].size(); j++)
        {
            ostream << std::setprecision(num_decimals) << this->train_x_in_[i][j] << " ";
        }
        ostream << "Target: ";
        ostream << std::setprecision(num_decimals) << this->train_yref_out_[i][0];

        ostream << " Pred: ";
        double print;
        for (auto &j : this->predict(this->train_x_in_[i]))
        {
            print = j < 0.1 ? 0 : j;
            ostream << print << std::setprecision(5) << "\t Real_value: " << j;
        }
        ostream << "\n";
    }
    ostream << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n";
}

/**
 * @brief prints information about the enitre neural net
 * @details
 *                          LITE    FULL
 * no of weights per node     x       x
 * no of nodes                x       x
 * activation mode            x       x
 * weight and bias data               x
 *
 * @param[in] po chose print option FULL or LITE
 * @param[in] ostream chosen output stream
 */
void Neural_network::print_network(print_option po, std::ostream &ostream)
{
    for (size_t i = 0; i < this->hidden_layers_.size(); i++)
    {
        ostream << "-=( hidden layer " << i + 1 << "  )=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n";
        this->hidden_layers_[i].print(po);
        ostream << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n";
    }
    ostream << "-=( output layer    )=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n";
    this->output_layer_.print(po);
    ostream << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n";
}