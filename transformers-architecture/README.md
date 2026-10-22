# Transformer Architecture

## 1. What is Transformer Architecture?

The Transformer architecture is a deep learning model that is mainly used to understand and generate text. It takes a sequence of tokens as input and can produce another sequence as output.

It is used in tasks such as translation, summarization, text generation, and many other applications.

## 2. Self-Attention Mechanism

Self-attention helps the model understand the meaning of each word by looking at the other words in the same sequence.
It creates **contextual embeddings**, meaning the representation of a word changes depending on its context.
**Example:**
- *Light as a feather* — Light refers to something that is not heavy.
- *Turn on the light* — Light refers to a source of illumination.
The model uses the surrounding words to understand the intended meaning of *light*.

## 3. Components of the Self-Attention Block

### Query (Q)

The Query represents what the current word is looking for in other words. It helps the model decide which words are important for understanding the current word.

### Key (K)

The Key represents information about each word. The model compares the Query with the Keys to determine how much attention to give to each word.

### Value (V)

The Value contains the information that is combined according to the attention scores to produce the final output representation.

### How Q, K, and V Work Together

1. The model generates Query, Key, and Value vectors for each token.
2. The Query is compared with the Keys to calculate attention scores.
3. The scores determine how much attention each token receives.
4. The scores are used to combine the Value vectors.
5. The result is a context-aware representation of each token.

## 4. Main Components of Transformer Architecture

### Tokens

Tokens are small units of text that the model processes. A token can be a word, part of a word, or a punctuation mark.

### Embeddings

Embeddings convert tokens into numerical vectors that represent their information in a form the model can process.

### Positional Information

Transformers process tokens in parallel, so they need positional information to understand the order of tokens in a sequence.
Positional information helps the model distinguish between words based on their positions.

### Parallel Processing

Transformers can process multiple tokens at the same time during training instead of processing each token one by one. This can make training faster.

### Encoder

The encoder processes the input sequence and creates representations that contain information about the tokens and their context.

### Decoder

The decoder uses the encoded information and previously generated tokens to produce the output sequence. In a standard encoder-decoder Transformer, it is commonly used to generate output one token at a time.

### Feed-Forward Network

The feed-forward network processes each token's representation after the attention operation. It helps the model learn more complex patterns and relationships.

### Training and Inference

- **Training:** The model learns patterns from data by adjusting its parameters.
- **Inference:** The trained model uses what it has learned to make predictions or generate outputs for new inputs.

### Multi-Head Attention

Multi-Head Attention uses multiple attention heads to understand different relationships between tokens at the same time. Each head learns to focus on different patterns or parts of the input. Their outputs are combined to create a richer representation of the input.

### Add & Norm

Add & Norm refers to two operations used in Transformer blocks: **residual connection (Add)** and **layer normalization (Norm)**.
- **Add:** Adds the input of a layer to its output. This helps information flow through the network and supports training deeper models.
- **Norm:** Layer normalization adjusts the values in the representation to help keep them stable during training.
These operations are commonly applied around the attention and feed-forward sublayers in the original Transformer architecture.

### Softmax Function

The softmax function is a mathematical tool that turns a list of raw, messy numbers (called logits) into clean percentages (probabilities) that always add up to 1 (or 100%).

## 5. Advantages of Transformer Architecture

- Helps improve performance on many language tasks.
- Understands the context of words.
- Processes multiple tokens in parallel during training.
- Can learn complex relationships between words.
- Can handle long-range relationships in text.

## 6. Applications of Transformer Architecture

- Machine Translation
- Text Summarization
- Text Generation
- Code Generation
- Image Processing and Generation
- Audio Processing
- Named Entity Recognition
- Speech Recognition

## 7. How Are All These Components Connected?

The Transformer architecture combines these components to process input data and produce useful outputs.

1. **Tokenization:** The input text is divided into tokens.
2. **Embeddings:** The tokens are converted into numerical vectors.
3. **Positional Information:** Information about token positions is added to the embeddings.
4. **Self-Attention:** The model uses Queries, Keys, and Values to identify relationships between tokens and understand their context.
5. **Feed-Forward Network:** The token representations are further processed to learn complex patterns.
6. **Encoder:** In an encoder-decoder Transformer, the encoder processes the input and creates contextual representations. The attention and feed-forward layers are repeated in its blocks.
7. **Decoder:** The decoder uses the encoded representations and previously generated tokens to produce the output sequence. It uses attention and feed-forward layers to process the information.
8. **Parallel Processing:** During training, the model can process multiple input tokens in parallel. During autoregressive generation, output tokens are generally generated one at a time.
9. **Training:** The model learns by adjusting its parameters using training data.
10. **Inference:** The trained model processes new inputs and generates predictions or outputs.
11. **Multi-Head Attention:** It runs multiple attention mechanisms in parallel. This lets the model capture different semantic patterns and relationships at the same time.
12. **Add and Norm:** The add and norm helps preserve information and stabilize the network.
13. **Softmax Function:** It is mostly used in the “final layer,” where the output values produced by AI are finally converted.

**Note:** Not every Transformer uses both an encoder and a decoder. Encoder-only models are commonly used for understanding text, decoder-only models for text generation, and encoder-decoder models for tasks such as translation and summarization.

---

*Notes | Day 8*