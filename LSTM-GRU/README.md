# LSTM and GRU

## LSTM (Long Short-Term Memory)

LSTM stands for **Long Short-Term Memory**. It is a special type of Recurrent Neural Network (RNN) used to process sequential data. It is commonly used in Natural Language Processing (NLP), speech recognition, time-series forecasting, and other sequence-based tasks.

A traditional RNN can struggle to remember information from earlier time steps because of the **vanishing gradient problem**. LSTM helps solve this problem by using a separate memory called the **Cell State** and three gates that control how information is stored, removed, and used. Unlike a basic RNN, which mainly uses a hidden state, LSTM maintains both a hidden state and a cell state.

### Key Components of LSTM

1. **Input Gate**
   - Controls how much new information should be added to the cell state.
   - Helps the network decide which information is important to remember.

2. **Forget Gate**
   - Decides which information from the previous cell state should be kept or removed.
   - Helps the network forget information that is no longer useful.

3. **Cell State (Memory Cell)**
   - Stores information over multiple time steps.
   - Acts as the long-term memory of the LSTM.
   - It is updated using the input and forget gates.

4. **Output Gate**
   - Controls how much information from the cell state should be passed to the hidden state.
   - Determines what information is used to produce the current output.

### Architecture and Working of LSTM

An LSTM consists of repeating units called **LSTM Cells**. Each cell processes the current input and information received from the previous time step.

1. **Input (\(x_t\))**
   - At each time step, the LSTM receives an input vector \(x_t\).
   - It represents the current element in the sequence, such as a word in a sentence or a value in a time series.

2. **Hidden State (\(h_t\))**
   - Stores information used by the network to produce the current output.
   - It also passes information to the next time step.
   - The initial hidden state is generally set to zero.

3. **Cell State (\(C_t\))**
   - Stores long-term information from previous time steps.
   - It is updated using the forget gate and input gate.
   - The initial cell state is generally set to zero.

4. **Gates**
   - LSTM uses three gates to control the flow of information.
   - **Forget Gate:** Takes the previous hidden state and current input to decide how much information from the previous cell state should be retained or removed.
   - **Input Gate:** Decides how much new information should be added to the cell state.
   - **Output Gate:** Decides how much information from the updated cell state should be used to generate the current hidden state.

5. **Output (\(h_t\))**
   - The updated hidden state is passed to the next time step.
   - Depending on the task, it can also be used to generate a prediction or final output.

### Equations of LSTM Gates

Let:
- \(x_t\) = Current input
- \(h_{t-1}\) = Previous hidden state
- \(C_{t-1}\) = Previous cell state
- \(C_t\) = Current cell state
- \(h_t\) = Current hidden state
- \(\sigma\) = Sigmoid activation function
- \(\tanh\) = Hyperbolic tangent activation function
- \(W\) = Weight matrices
- \(b\) = Bias vectors
- \([h_{t-1}, x_t]\) = Concatenation of the previous hidden state and current input

**1. Forget Gate**
$$f_t = \sigma(W_f \cdot [h_{t-1},x_t] + b_f)$$
- Produces values between 0 and 1.
- A value closer to 0 means more information is forgotten.
- A value closer to 1 means more information is retained.

**2. Input Gate**
First, the input gate decides how much new information should be added:
$$i_t = \sigma(W_i \cdot [h_{t-1},x_t] + b_i)$$
Next, a candidate cell state is created:
$$\widetilde{C}_t = \tanh(W_C \cdot [h_{t-1},x_t] + b_C)$$
- The input gate \(i_t\) controls how much of the candidate information is added.
- The candidate cell state \(\widetilde{C}_t\) contains the new information that can be stored.

**3. Cell State Update**
The cell state is updated using the forget gate and input gate:
$$C_t = f_t * C_{t-1} + i_t * \widetilde{C}_t$$
- The first part controls how much previous information is retained.
- The second part controls how much new information is added.

**4. Output Gate**
First, the output gate decides how much information should be used:
$$o_t = \sigma(W_o \cdot [h_{t-1},x_t] + b_o)$$
Then, the current hidden state is calculated:
$$h_t = o_t * \tanh(C_t)$$
- The output gate controls how much information from the cell state becomes part of the hidden state.
- The hidden state is passed to the next time step and can also be used for prediction.

Here, \(*\) represents element-wise multiplication.

## GRU (Gated Recurrent Unit)

GRU stands for **Gated Recurrent Unit**. It is another type of RNN architecture used to process sequential data. Like LSTM, GRU helps the network remember important information and forget unnecessary information over time.

However, GRU has a simpler architecture than LSTM. It uses only two gates (Reset Gate and Update Gate) and does not maintain a separate cell state. Instead, it uses a single hidden state to store and update information.

This simpler structure means GRUs generally have fewer parameters and can require less computation than LSTMs. However, which model performs better depends on the dataset and task.

### How GRU Works

Like other RNN architectures, GRU processes sequential data one element at a time. At each time step, it takes the current input and previous hidden state, calculates a candidate hidden state, and updates its memory using two gates.

1. **Reset Gate**
   - Decides how much information from the previous hidden state should be ignored while creating the candidate hidden state.
   - Helps the network forget unnecessary past information.

2. **Update Gate**
   - Decides how much of the previous hidden state should be retained and how much new information should be added.
   - Helps the network update its memory.

3. **Candidate Hidden State**
   - A new representation created using the current input and the previous hidden state.
   - The reset gate controls how much previous information is used to create it.
   - The Tanh activation function keeps its values between -1 and 1.

4. **New Hidden State**
   - Combines the previous hidden state and candidate hidden state.
   - The update gate controls how much information is retained or replaced.

### Equations of GRU

Let:
- \(x_t\) = Current input
- \(h_{t-1}\) = Previous hidden state
- \(h_t\) = Current hidden state
- \(\widetilde{h}_t\) = Candidate hidden state
- \(r_t\) = Reset gate
- \(z_t\) = Update gate
- \(\sigma\) = Sigmoid activation function
- \(\tanh\) = Hyperbolic tangent activation function
- \(W\) = Weight matrices
- \(b\) = Bias vectors

**1. Reset Gate**
$$r_t = \sigma(W_r \cdot [h_{t-1},x_t] + b_r)$$
- Determines how much of the previous hidden state should be used when creating the candidate hidden state.
- Values closer to 0 indicate less use of previous information, while values closer to 1 indicate more use.

**2. Update Gate**
$$z_t = \sigma(W_z \cdot [h_{t-1},x_t] + b_z)$$
- Determines how much of the previous hidden state should be retained and how much new information should be incorporated.

**3. Candidate Hidden State**
$$\widetilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1},x_t] + b_h)$$
- Combines the current input with the previous hidden state after it has been controlled by the reset gate.
- Produces a candidate hidden state containing new information.

**4. New Hidden State**
$$h_t = (1-z_t) * h_{t-1} + z_t * \widetilde{h}_t$$
- Combines the previous hidden state with the candidate hidden state.
- The update gate controls the contribution of each:
  - If \(z_t\) is close to 0, more previous information is retained.
  - If \(z_t\) is close to 1, more candidate information is used.

Here, \(*\) represents element-wise multiplication.

### Architecture of GRU

The GRU architecture consists of the following components:

1. **Input Layer**
   - Receives sequential data, such as words in a sentence or values in a time series.
   - Passes the input to the GRU cell at each time step.

2. **Hidden State**
   - Stores information from previous time steps.
   - Gets updated whenever a new input is processed.
   - Acts as the memory of the GRU.

3. **Reset Gate**
   - Controls how much previous information is used while calculating the candidate hidden state.

4. **Update Gate**
   - Controls how much previous information is retained and how much candidate information is added to the new hidden state.

5. **Candidate Hidden State**
   - Represents the new information generated using the current input and the reset-controlled previous hidden state.
   - Uses the Tanh activation function.

6. **Output Layer**
   - Uses the hidden state to generate the final output.
   - Depending on the task, the output can be a single value, a sequence of values, or class probabilities.

---

*Notes | Day 19*