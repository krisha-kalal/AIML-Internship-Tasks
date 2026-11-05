# ANN, CNN, RNN

## ANN (Artificial Neural Network)

An Artificial Neural Network (also known as neural network) is a computational model inspired by the way biological neural networks in the human brain to process information. ANNs are a key technology in machine learning and artificial intelligence, used for a variety of applications such as classification, regression, clustering, and more.

It maps complex, non-linear relationships in data by passing information through interconnected layers of artificial neurons.

### Core Components of ANN

1. **Neurons (Nodes)**
- Basic units of an ANN.
- Receive input, process it, and pass the output to the next layer.

2. **Layers**
- **Input Layer:** The layer that receives the input data.
- **Hidden Layers:** Intermediate layers where the processing is done. There can be one or more hidden layers.
- **Output Layer:** Produces the final output of the network.

3. **Weights**
- Parameters that are adjusted during training.
- Represent the strength of the connection between neurons.

4. **Bias**
- An additional parameter in each neuron that allows the model to fit the data better.

5. **Activation Function**
- A mathematical function is applied to the output of each neuron.
- Introduces non-linearity (allows the network to learn complex patterns) to the model.
- Common activation functions include Sigmoid, Tanh, ReLU (Rectified Linear Unit), and Softmax.

### Architecture of ANN

1. **Forward Propagation**

- Input data is fed into the input layer.
- Data is passed through hidden layers, where neurons process it using weights, biases, and activation functions.
- Final output is produced at the output layer.

2. **Loss Calculation**

- The output is compared to the actual target values using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
- The loss function measures how far the predicted output is from the actual target.

3. **Backward Propagation**

- The error is propagated back through the network.
- Weights and biases are adjusted to minimize the loss using optimization algorithms like Gradient Descent.

4. **Training**

- The processes of forward propagation, loss calculation, and backward propagation are repeated iteratively.
- The network learns to make better predictions by adjusting the weights and biases.

## CNN (Convolutional Neural Network)

Convolutional Neural Networks, commonly referred to as CNNs are a specialized type of neural network designed to process and classify images.

Digital images are essentially grids of tiny units called pixels. Each pixel represents the smallest unit of an image and holds information about the color and intensity at that particular point.

Typically, each pixel is composed of three values corresponding to the red, green, and blue (RGB) color channels. These values determine the color and intensity of that pixel. In contrast, in a grayscale image, each pixel carries a single value that represents the intensity of light at that point. — usually ranging from black (0) to white (255).

### Core Components of CNN

1. **Convolutional Layers:** They are the fundamental building blocks of CNNs. These layers perform a critical mathematical operation known as convolution. This process entails the application of specialized filters known as kernels, that traverse through the input image to learn complex visual patterns.

2. **Kernels:** They are essentially small matrices of numbers. These filters move across the image performing element-wise multiplication with the part of the image they cover, extracting features such as edges, textures, and shapes.

3. **Convolution Operation:** The convolution operation involves multiplying the kernel values by the original pixel values of the image and then summing up the results. The output matrix of this process is known as the Feature map.

4. **Pooling Layers:** They downsample the spatial dimensions of the input, reducing the computational complexity and the number of parameters in the network. Max pooling is a common pooling operation where we select a maximum value from a group of neighboring pixels.

5. **Fully Connected Layers:** These layers are responsible for making predictions based on the high-level features learned by the previous layers. They connect every neuron in one layer to every neuron in the next layer.

### Architecture of CNN

1. **Neurons:** The most basic unit in a neural network. They are composed of a sum of linear functions and a non-linear function known as the activation function is applied to them.

2. **Input layer:** Each neuron in the input layer corresponds to one of the input features.
- Example: For instance, in an image classification task where the input is a 28 x 28-pixel image, the input layer would have 784 neurons (one for each pixel).

3. **Hidden Layer:** The layers between the input and the output layer. Each neuron in this layer is summed by the result of the neurons in the previous layers and multiplied by a non-linear function.

4. **Output Layer:** The number of neurons in the output layer corresponds to the number of output classes (In case if facing a regression problem the output layer will only have one neuron).
- For example, in a classification task with digits from 0 to 9, the output layer would have 10 neurons.

## RNN (Recurrent Neural Network)

In any language, sentences aren’t just random words thrown together, they follow grammatical rules and convey specific meanings. To generate meaningful sentences, a model needs to understand the sequence and context of words. This is where Recurrent Neural Networks (RNNs) come into play.

RNNs are designed to handle sequential data by taking the output from previous steps and using it as input for the current step. They are essential for tasks that involve time series data or any data where the order matters. In simpler words, RNNs are designed to recognize patterns in sequences of data. They have a “memory” that captures information about what has been calculated so far. This memory allows RNNs to use prior inputs to influence the current output.

### Core Components of RNN

1. **Recurrent Neurons** 

The fundamental processing unit in RNN is a Recurrent Unit. They hold a hidden state that maintains information about previous inputs in a sequence. Recurrent units can "remember" information from prior steps by feeding back their hidden state, allowing them to capture dependencies across time.

2. **RNN Unfolding**

RNN unfolding or unrolling is the process of expanding the recurrent structure over time steps. During unfolding each step of the sequence is represented as a separate layer in a series illustrating how information flows across each time step.

This unrolling enables backpropagation through time (BPTT) a learning process where errors are propagated across time steps to adjust the network’s weights enhancing the RNN’s ability to learn dependencies within sequential data.

### Architecture of RNN

RNNs can be structured in various ways to suit different tasks:

1. **One-to-One:** Standard neural network (eg, image classification).
2. **One-to-Many:** Single input, sequence output (eg, image captioning).
3. **Many-to-One:** Sequence input, single output (eg, sentiment analysis).
4. **Many-to-Many (Same Length):** Sequence input and output of the same length (eg, named entity recognition).
5. **Many-to-Many (Different Length):** Sequence input and output of different lengths (eg, machine translation).

### Variants of RNN

1. **Bidirectional RNN (BRNN)**

BRNNs process data in both forward and backward directions, allowing the network to have both past and future context.

2. **Long Short-Term Memory Networks (LSTM)**

LSTMs are designed to handle the vanishing gradient problem by introducing a memory cell that can maintain information over long periods.

Components:
Cell State: Stores long-term dependencies.
Gates:
- Input Gate: Controls how much new information flows into the cell.
- Forget Gate: Decides what information to discard from the cell.
- Output Gate: Regulates what information should be output at the current step.

3. **Gated Recurrent Units (GRU)**

GRUs are a simplified version of LSTMs with only two gates: reset gate and update gate.

---

*Notes | Day 18*