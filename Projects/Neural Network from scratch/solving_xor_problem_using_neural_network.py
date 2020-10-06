# @author https://github.com/NirmalSilwal

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation function

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation

def sigmoid_derivative(x):
  return x * (1 - x)

# Initialzing input and output values - as in XOR table

# input parameters
inputs = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

# output parameters
outputs = np.array([[0], [1], [1], [0]])

# Feed Forward network

def forward_pass(inputs, hidden_weights, hidden_bias, output_weights, output_bias):

  hidden_layer_activation = np.dot(inputs, hidden_weights)
  hidden_layer_activation += hidden_bias

  hidden_layer_output = sigmoid(hidden_layer_activation)

  output_layer_activation = np.dot(hidden_layer_output, output_weights)
  output_layer_activation += output_bias

  predicted_output = sigmoid(output_layer_activation)

  return predicted_output, hidden_layer_output

# Backward propagation step

def backward_pass(expected_output, predicted_output, output_weights, hidden_layer_output):

  error = expected_output - predicted_output

  d_predicted_output = error * sigmoid_derivative(predicted_output)

  error_hidden_layer = d_predicted_output.dot(output_weights.T)
  d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

  return d_predicted_output, d_hidden_layer

# Setting hyperparamters

# initializing learning rate
lr = 0.1

# total epochs
epochs = 15000

# we'r taking 2 hidden layer neuron since XOR couldn't be solved using single decision plane
# 1 output layer neuron for single output
# 2 input layer neuron to pass 2 inputs to the Gate (0/1)

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Training Neural Network

def train(epochs, lr, inputs, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons):

  # Initializing random weights and biases for training

  hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))

  hidden_bias =np.random.uniform(size=(1, hiddenLayerNeurons))

  output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))

  output_bias = np.random.uniform(size=(1, outputLayerNeurons))

  for epoch in range(epochs):

    # forward pass
    predicted_output, hidden_layer_output = forward_pass(inputs, hidden_weights, hidden_bias, output_weights, output_bias)

    # backward pass
    d_predicted_output, d_hidden_layer = backward_pass(outputs, predicted_output, output_weights, hidden_layer_output)

    #Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr

    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

    hidden_weights += inputs.T.dot(d_hidden_layer) * lr

    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
      print(f"Training.... epoch: {epoch}")
    if epoch == (epochs - 1):
      print('\nTraining completed!')
      
  return predicted_output

predicted_output = train(epochs, lr, inputs, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons)

# Predicted Output
print('\nPredicted output:')
print(predicted_output)

print('\n Round value of the predicted labels:\n')
for val in predicted_output:
  print(np.round(val))

# Note: As you can see above, the prediction are almost accurate (97% accurate) as the round value 
# is close to 0 and 1 for respective inputs. Recall from above XOR truth table, this predicted values
# are same as expected output for XOR gate i.e it yeilds true output only when both its inputs 
# differ from each other.

# Thank You!
# Feel free to make the suggestions!
