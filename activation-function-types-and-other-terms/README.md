# Activation Function and It's Types, Epoch, Batch Size, Iterations and Learning Rate

## Activation Function

An activation function is a very important feature of an artificial neural network, they basically decide whether the neuron should be activated or not.

In artificial neural networks, the activation function defines the output of that node given an input or set of inputs. In simple term , it calculates a “weighted sum(Wi)” of its input(xi), adds a bias and then decides whether it should be “fired” or not.

### Types of Activation Function

1. **Linear Function**

A linear activation function takes the form:
**y = mx + c** (m is line equation represents W and c is represented as b in neural nets so equation can be modified as y = Wx + b)

It takes the inputs (Xi’s), multiplied by the weights(Wi’s) for each neuron, and creates an output proportional to the input. In simple term, weighted sum input is proportional to output.

2. **Binary Step Function**

Binary step function are popular known as “Threshold function”. It is very simple function.

It gives an output of either 0 or 1 depending on whether the input crosses a fixed threshold.

3. **Sigmoid or Logistic Function**

It converts the input into a value between 0 and 1. It is commonly used for binary classification output.

4. **ReLU (ReLU — Rectified Linear Units) Function**

ReLU gives 0 for negative inputs and returns the input itself for positive inputs. It is widely used in hidden layers of neural networks.

5. **Leaky ReLU Function**

Leaky ReLU is similar to ReLU, but it gives a small negative output for negative inputs instead of giving 0. This helps the neuron continue learning even when its input is negative.

6. **Tanh Function**

Tanh is the modified version of sigmoid function. Hence have similar properties of sigmoid function.

It converts the input into a value between -1 and 1.

7. **Softmax Function**

Softmax converts a group of values into probabilities that add up to 1. It is commonly used in the output layer for classification problems with multiple classes.

## Epoch

An epoch refers to one complete pass through the entire training dataset once by the neural network. This includes:

- One forward pass: The model takes inputs, makes predictions, and computes errors.
- One backward pass: The model adjusts its weights based on the errors to improve future predictions.

By repeating the process for multiple epochs, the model gradually learns and improves its performance.

## Batch

A batch refers to a subset of the training data that is processed together during one forward pass and one backward pass of the neural network. Since the entire dataset is often too large to process all at once, it is divided into smaller groups called batches.

## Batch Size

Batch size is a hyperparameter that defines the number of training examples included in one batch. It determines how much data is processed before the model updates its weights.

## Iteration

An iteration refers to a single step of training where the model processes one batch of data and updates its parameters (weights and biases). Each iteration includes two main parts:

- Forward Propagation: The input data in the batch is passed through the network to generate predictions, and the loss (error) is calculated.
- Backward Propagation: The gradients are calculated, and the model’s parameters are updated to reduce the loss.

### Relation Between Epoch, Batch Size and Iteration

**Iterations per epoch = Total Training Samples / Batch Size**

**Example:** If we have 1,000 training samples and a batch size of 100, it takes 10 iterations to complete 1 epoch.

## Learning Rate

The learning rate is a core hyperparameter that sets the step size used by optimization algorithms (like SGD or Adam) to update model weights based on the gradient of the loss. In short, it controls how aggressively or cautiously your model learns.

If the learning rate is too small, training crawls and can get stuck in bad regions. If it’s too large, loss can bounce around or even explode, failing to converge.

### Common Starting Defaults

Depending on the model and optimizer you choose, standard baseline learning rates are typically:

- Adam / AdamW: 0.001 (1e-3) or 0.0003 (3e-4)
- SGD (Stochastic Gradient Descent with momentum): 0.01 (1e-2) or 0.1
- Transformers / Fine-Tuning: 3e-5 to 1e-4

---

*Notes | Day 12*