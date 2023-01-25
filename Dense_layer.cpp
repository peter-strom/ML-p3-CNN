#include "Dense_layer.hpp"
#include "definitions.hpp"

/**
 * @brief Construct a new Dense_layer::Dense_layer object
 *
 * @details sets the number of nodes and weights in the new dens-layer.
 *
 * @param[in] num_nodes
 * @param[in] num_weights
 */
Dense_layer::Dense_layer(const std::size_t num_nodes,
                         const std::size_t num_weights)
{
    this->resize(num_nodes, num_weights);
}

/**
 * @brief Destructor erases all values in each nodes vector.
 *
 */
Dense_layer::~Dense_layer()
{
    this->clear();
}

/**
 * @brief returns the number of nodes for selected dense layer.
 *
 * @return number of nodes.
 */
std::size_t Dense_layer::num_nodes(void) const
{
    return this->output.size();
}

/**
 * @brief returns the number of weights per node for selected dense-layer.
 *
 * @return number of weights.
 */
std::size_t Dense_layer::num_weights(void) const
{
    if (this->weights.size() == 0)
    {
        return 0;
    }
    else
    {
        return this->weights[0].size();
    }
}

/**
 * @brief sets the desired activation function for selected dense layer.
 *
 * @param[in] af RELU or TANH
 */
void Dense_layer::set_activation(const activation_option ao)
{
    this->ao = ao;
}

/**
 * @brief Erases all the elements in the vector containers for selected dense layer
 *
 */
void Dense_layer::clear(void)
{
    this->output.clear();
    this->error.clear();
    this->bias.clear();
    this->weights.clear();
}

/**
 * @brief sets the size of all the elements in the vector containers for selected dense layer
 *
 * @param[in] num_nodes number of nodes
 * @param[in] num_weights number of weights per node
 */
void Dense_layer::resize(const std::size_t num_nodes,
                         const std::size_t num_weights)
{
    this->output.resize(num_nodes, 0.0);
    this->error.resize(num_nodes, 0.0);
    this->bias.resize(num_nodes, 0.0);
    this->weights.resize(num_nodes, std::vector<double>(num_weights, 0.0));

    for (std::size_t i = 0; i < num_nodes; ++i)
    {
        this->bias[i] = this->get_random();

        for (std::size_t j = 0; j < num_weights; ++j)
        {
            this->weights[i][j] = this->get_random();
        }
    }
}

/**
 * @brief calculates new output for each node in selected dense-layer
 *
 * @details output[i] = (bias + (input * weight)) tanh
 * | outside |          layer         |
 *  [input0] - [weight 0 0] [        ]
 *  [input1] - [weight 0 1] [ node 0 ]
 *  [input2] - [weight 0 2] [        ]
 * @param[in] input indata from training data or previous layer
 */
void Dense_layer::feedforward(const std::vector<double> &input)
{
    for (std::size_t i = 0; i < this->num_nodes(); i++)
    {
        double sum = bias[i];
        for (std::size_t j = 0; j < this->num_weights() && j < input.size(); j++)
        {
            sum += input[j] * this->weights[i][j];
        }
        this->output[i] = this->activation(sum);
    }
}

/**
 * @brief calculates the error for each node in output layer.
 *
 * @details
 * Backpropagation is going backwards.
 * ie.  hidden_layer_0 <- output_layer
 *                             |
 *                         you are here
 *
 *
 * |  layer   |                    outside the layer                |
 *  [        ]
 *  [ node 0 ] =  [reference or train_yref] - [output or y_predict]
 *  [ error  ]
 *
 * @param[in] reference target value from training data (yref)
 */
void Dense_layer::backpropagate(const std::vector<double> &reference)
{
    for (std::size_t i = 0; i < this->num_nodes(); i++)
    {
        double dev = reference[i] - this->output[i];
        this->error[i] = dev * delta_activation(this->output[i]);
    }
}

/**
 * @brief calculates the error for each node in a dense layer.
 * @details
 * Backpropagation is going backwards.
 * ie.  hidden_layer_0 <- output_layer
 *            |                |
 *        you are here        this is next layer
 *
 *
 * |  layer    |          nextlayer                |
 *  [        ]   [weight 0 0] * [node 0 error] +
 *  [ node 0 ] = [weight 1 0] * [node 1 error] +
 *  [ error  ]   [weight 2 0] * [node 2 error] ...
 * @param[in] next_layer mext dense layer
 */
void Dense_layer::backpropagate(const Dense_layer &next_layer)
{
    for (std::size_t i = 0; i < this->num_nodes(); i++)
    {
        double dev = 0.0;
        {
            for (std::size_t j = 0; j < next_layer.num_nodes(); j++)
            {
                dev += next_layer.error[j] * next_layer.weights[j][i];
            }
            this->error[i] = dev * this->delta_activation(this->output[i]);
        }
    }
}

/**
 * @brief calculates new bias and new weights for the dense layer
 *
 * @details
 * Gradient descent (optimize)
 * m1(new) = m1 + e1 * LR
 * k11(new) = k11 + e1 * LR * x1
 * k = weight, m = bias
 *
 * | outside |          layer          |
 *  [input0] - [weight 0 0] [        ]
 *  [input1] - [weight 0 1] [ node 0 ]
 *  [input2] - [weight 0 2] [        ]
 * @param[in] input in-data from training data or previous layer
 * @param[in] learning_rate amount of error adjustment
 */
void Dense_layer::optimize(const std::vector<double> &input,
                           const double learning_rate)
{
    for (std::size_t i = 0; i < this->num_nodes(); i++)
    {
        this->bias[i] += this->error[i] * learning_rate;
        for (std::size_t j = 0; j < this->num_weights() && j < input.size(); j++)
        {
            this->weights[i][j] += this->error[i] * learning_rate * input[j];
        }
    }
}
/**
 * @brief returns a value beteween 0  and 1
 *
 * @return double
 */
inline double Dense_layer::get_random(void)
{
    return (double)(std::rand()) / RAND_MAX;
}

/**
 * @brief function to choose from ReLU or Tanh in feedforward
 *
 * @details option "RELU" is a linear function that returns
 *          sum if sum is over 0.
 *          option "TANH" is a non-linear function that returns
 *          a value beween -1 and 1. lower values will be mapped closer
 *          to 0 while "higher" values will be mapped closer to 1 or -1.
 *
 * @param[in] sum
 * @return sum
 */
inline double Dense_layer::activation(const double sum)
{
    if (this->ao == activation_option::TANH)
    {
        return tanh(sum);
    }
    else
    {
        return sum > 0.0 ? sum : 0.0;
    }
}

/**
 * @brief function to choose from ReLU or Tanh in backproagation.
 *
 * @details option "RELU" returns 1 if output is higher than 0 (node activated)
 *          and 0 if output is 0 (node not activated)
 *          Option "TANH" returns a value between 0 and 1, so error value will
 *          be smaller compared to using ReLU in backpropagation.
 * @param[in] output
 * @return output
 */
inline double Dense_layer::delta_activation(const double output)
{
    if (this->ao == activation_option::TANH)
    {
        return 1 - tanh(output) * tanh(output);
    }
    else
    {
        return output > 0.0 ? 1.0 : 0.0;
    }
}

/**
 * @brief prints information about the dense layer
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
void Dense_layer::print(print_option po, std::ostream &ostream)
{
    ostream << this->num_weights() << " weights per node.\n";
    ostream << this->num_nodes() << " nodes.\n";
    if (this->ao == activation_option::TANH)
    {
        ostream << "Activation: TANH \n";
    }
    if (this->ao == activation_option::RELU)
    {
        ostream << "Activation: RELU \n";
    }

    if (po == print_option::FULL)
    {
        for (std::size_t i = 0; i < this->num_nodes(); i++)
        {
            ostream << "Node: [" << i << "]   bias: " << bias[i] << "   weights: ";
            for (std::size_t j = 0; j < this->num_weights(); j++)
            {
                ostream << "[" << j << "] " << weights[i][j] << " , ";
            }
            ostream << "\n";
        }
    }
}