# BERT and RoBERTa – Transformer Models

## BERT (Bidirectional Encoder Representations from Transformers)

BERT is a Transformer-based language model developed by Google for understanding the meaning and context of words in text. It is **bidirectional**, meaning it uses information from both the left and right sides of a token to understand its meaning.

BERT is based on the Transformer Encoder architecture and is commonly used for tasks such as text classification, sentiment analysis, Named Entity Recognition (NER), and question answering.

### Working of BERT

**1. Input Processing**
- BERT first divides the input text into smaller subword tokens using **WordPiece Tokenization**.
- It also adds special tokens to identify the structure of the input.
Common special tokens:
- **[CLS]:** Added at the beginning of the input. Its final representation can be used for tasks such as text classification.
- **[SEP]:** Marks the end of a sentence or separates two sentences.
The tokens are then converted into numerical representations and passed through BERT's Transformer encoder.

**2. Masked Language Modeling (MLM)**
- During pretraining, BERT selects some tokens from the input and learns to predict their original values using the surrounding context.
Since BERT is bidirectional, it uses both the words before and after the masked token to predict the missing word.

**3. Next Sentence Prediction (NSP)**
- BERT is also trained to understand the relationship between two sentences.
- It receives two sentences and predicts whether the second sentence actually follows the first sentence in the original training text.

**4. Combined Pretraining**
- BERT is originally pretrained using two objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- MLM helps BERT understand words using their surrounding context.
- NSP helps BERT learn relationships between sentence pairs.
Together, these objectives help BERT learn general language patterns before being used for specific NLP tasks.

**5. Fine-Tuning**
- After pretraining, BERT can be adapted to specific NLP tasks. These are called **downstream tasks**.
- A task-specific output layer is added, and the model is further trained using labeled data for that task.
Examples:
- **Sentiment Analysis:** Classifies text as positive or negative.
- **Named Entity Recognition (NER):** Identifies names of people, places, organizations, etc.
- **Question Answering:** Finds answers to questions from a given text.

The same pretrained BERT model can be adapted to different tasks without training a language model from scratch.

### Architecture of BERT

BERT consists of multiple **Transformer Encoder layers** stacked together. Each encoder layer contains:
1. **Multi-Head Self-Attention**
   - Allows each token to focus on other tokens in the input sequence.
   - Helps the model understand relationships between words.
2. **Feed-Forward Network (FFN)**
   - Further processes the representations produced by the attention layer.
   - Helps the model learn more complex patterns.
3. **Residual Connections**
   - Add the input of a sub-layer to its output.
   - Help information flow through the network and support stable training.
4. **Layer Normalization**
   - Normalizes the values in the network.
   - Helps make training more stable.
Multiple encoder layers are stacked to produce increasingly meaningful contextual representations.

### Input Representation
BERT combines three types of embeddings for each input token:
1. **Token Embeddings**
   - Represent the individual tokens in the input text.
2. **Segment Embeddings**
   - Indicate which sentence a token belongs to.
   - Mainly useful when the input contains a pair of sentences.
3. **Position Embeddings**
   - Provide information about the position of each token in the sequence.
   - Help BERT understand the order of tokens.
These three embeddings are added together and passed through the Transformer encoder.

### BERT Base and BERT Large
BERT was introduced in two main configurations:
| Feature | BERT Base | BERT Large |
|---|---:|---:|
| Transformer Encoder Layers | 12 | 24 |
| Attention Heads | 12 | 16 |
| Hidden Size | 768 | 1024 |
| Parameters | Approximately 110 million | Approximately 340 million |
| Maximum Input Length | 512 tokens | 512 tokens |
- **BERT Base:** Has fewer layers and parameters, making it smaller.
- **BERT Large:** Has more layers and parameters, allowing it to learn more complex representations but requiring more computational resources.

### Output Representations
- BERT produces a contextual representation for every input token.
- Unlike traditional static word embeddings, the representation of a word can change depending on the surrounding words.
Example:
```text
I deposited money in the bank.
We sat near the river bank.
```
The word **bank** has different meanings in these sentences. BERT can generate different representations for it based on the context.
- **[CLS] Output:** Its final representation can be used for sequence-level tasks such as text classification.
- **Token-Level Outputs:** Used for tasks such as Named Entity Recognition (NER).

### BERT Tokenization

BERT uses **WordPiece Tokenization** to divide text into smaller subword units. This helps BERT handle words that are not directly available in its vocabulary by splitting them into smaller known tokens. For example, a word may be divided into multiple subword tokens. The `##` symbol indicates that a token is a continuation of the previous subword.

**Python Example:**
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
text = "BERT understands language context."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```
Output:
```text 
Tokens: ['B', '##ER', '##T', 'understands', 'language', 'context', '.']```
Special tokens such as `[CLS]` and `[SEP]` are also added when required by the model.

## 2. RoBERTa (Robustly Optimized BERT Approach)

RoBERTa is a Transformer-based language model developed by Meta AI to improve BERT's performance through better pretraining strategies. It uses the same basic Transformer Encoder architecture as BERT but introduces improvements in the way the model is trained.

RoBERTa is also an **encoder-only pretrained language model** and is available through the Hugging Face Transformers library. These training improvements help RoBERTa learn better contextual representations and improve its performance on various NLP tasks.

### Need for RoBERTa

Although BERT achieved strong results in NLP tasks, researchers found that its pretraining process could be improved. RoBERTa was introduced to address these limitations through several changes:

- Improves BERT's pretraining strategy.
- Uses larger and more diverse training datasets.
- Trains for more steps with larger batch sizes.
- Uses dynamic masking instead of static masking.
- Removes the Next Sentence Prediction (NSP) objective and focuses on Masked Language Modeling (MLM).
- Achieves better performance than BERT on several NLP benchmark tasks.

### Architecture of RoBERTa

RoBERTa uses the same basic Transformer Encoder architecture as BERT. The main improvements are in its pretraining strategy rather than the overall network structure.

Its architecture consists of the following components:
1. **Input Embeddings**
   - The input text is tokenized using Byte-Pair Encoding (BPE).
   - Each token is converted into a numerical embedding that is passed to the Transformer encoder.
2. **Positional Embeddings**
   - Added to token embeddings to provide information about the position of each token.
   - Help the model understand the order of tokens in a sequence.
3. **Transformer Encoder Layers**
   - The embedded input passes through multiple Transformer encoder layers.
   - These layers learn contextual information by analyzing relationships between tokens.
4. **Multi-Head Self-Attention**
   - Allows each token to focus on other tokens in the input sequence.
   - Helps capture different relationships between words at the same time.
5. **Feed-Forward Network (FFN)**
   - Processes the attention outputs further.
   - Helps the model learn more meaningful representations.
6. **Residual Connections and Layer Normalization**
   - Residual connections help preserve information across layers.
   - Layer normalization helps improve training stability.
7. **Contextual Output Representations**
   - The final encoder outputs are context-aware representations of the input tokens.
   - These representations can be used for tasks such as text classification, question answering, and Named Entity Recognition.

### Working of RoBERTa

RoBERTa processes input text through multiple Transformer encoder layers to understand the context and meaning of words.
1. **Input Text**
   - A sentence or document is provided as input to the model.
2. **Tokenization**
   - The text is divided into tokens or subword units using Byte-Pair Encoding (BPE).
3. **Embedding Generation**
   - Each token is converted into an embedding.
   - Positional embeddings are added to provide information about token positions.
4. **Transformer Encoding**
   - The embeddings pass through multiple Transformer encoder layers.
   - Multi-Head Self-Attention helps each token capture contextual information from other tokens in the sequence.
5. **Masked Language Modeling**
   - During pretraining, selected tokens are masked dynamically.
   - The model learns to predict the original tokens using the surrounding context.
6. **Contextual Representation**
   - The encoder produces context-aware representations that capture the meaning of each token based on the input sequence.
7. **Fine-Tuning**
   - The pretrained model can be fine-tuned for specific NLP tasks such as text classification, question answering, sentiment analysis, and Named Entity Recognition.

---

*Notes | Day 20*