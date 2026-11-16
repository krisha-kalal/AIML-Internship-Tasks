# RAG, LLM

## RAG (Retrieval-Augmented Generation)

RAG is a technique that allows LLMs to access and use external data, including private or company-specific information, without requiring the model to be retrained.

LLMs are trained on large amounts of data, but they may not have access to the latest information or an organization's private documents. RAG solves this problem by connecting an LLM to an external knowledge source. It retrieves relevant information and provides it to the LLM as context before generating an answer.

In simple terms, **RAG combines information retrieval with text generation**. Instead of relying only on the knowledge learned during training, the LLM can use relevant information retrieved from external sources.

### Need of RAG?

LLMs have certain limitations:

- **Limited Context Window:** An LLM can process only a limited amount of information at one time.
- **Outdated Knowledge:** The model may not know about recent updates or information published after its training.
- **Private Data:** The model may not have access to an organization's internal documents, databases, or records.
- **Hallucinations:** An LLM may generate incorrect or unsupported information when it does not have sufficient context.

RAG helps address these limitations by retrieving relevant and updated information from external knowledge sources and providing it to the LLM.

### RAG Architecture / Pipeline

The RAG pipeline is generally divided into two main stages:

#### 1. Ingestion Pipeline

This stage prepares documents so that they can be searched later.

1. **Document Loading:** Documents are collected from different sources, such as PDFs, text files, websites, and databases.

2. **Chunking:** Large documents are divided into smaller, manageable pieces called *chunks*. This makes information easier to process and retrieve. For example, a document can be divided into chunks of approximately 1,000 tokens.

3. **Embedding Generation:** Each chunk is passed through an embedding model, which converts the text into a numerical vector representation called an *embedding*. These vectors capture the semantic meaning of the text.

4. **Storage:** The embeddings, along with their corresponding text and metadata, are stored in a vector database such as ChromaDB or Pinecone. FAISS can also be used for efficient vector similarity search.

#### 2. Retrieval and Generation Pipeline

This stage retrieves relevant information when a user asks a question.

1. **User Query:** The user enters a question or prompt.

2. **Query Embedding:** The same embedding model used during ingestion converts the user's query into a numerical vector.

3. **Similarity Search:** The retriever compares the query embedding with the stored document embeddings and identifies chunks that are semantically similar to the query.

4. **Retrieval:** The most relevant chunks are selected and retrieved from the knowledge base. These are often called the *top-k results*, where k represents the number of chunks retrieved.

5. **Augmentation:** The retrieved chunks are combined with the original user query to create an enriched prompt.

6. **Generation:** The LLM receives this prompt and uses the retrieved information as context to generate a relevant and informed response.

### Types of RAG Retrieval

- **Dense Retrieval:** Uses vector embeddings to find documents based on semantic similarity.
- **Sparse Retrieval:** Uses keyword-based techniques such as BM25 to find documents containing relevant terms.
- **Hybrid Retrieval:** Combines dense and sparse retrieval to use both semantic meaning and keyword matching.

Hybrid retrieval can improve search quality because a query may contain important keywords that dense retrieval alone does not rank highly.

### Advantages of RAG

- Provides access to external and private information.
- Helps answer questions using updated information.
- Reduces the need to retrain an LLM whenever documents change.
- Can reduce hallucinations by grounding responses in retrieved context.
- Supports source-based answers and citations when document metadata is maintained.
- Makes it possible to build domain-specific question-answering systems.

## LLM (Large Language Model)

A LLM is an AI model trained on large amounts of text data to understand and generate human-like language. 

LLMs are commonly based on the Transformer architecture and learn patterns, relationships, and context from language during training. They can perform tasks such as: Text generation and summarization, Question answering, Translation, Sentiment analysis, Code generation, etc. Examples include GPT, Gemini, Claude, and Llama.

### How does an LLM work?

1. **Input:** The user provides a prompt or question.
2. **Tokenization:** The input text is converted into smaller units called tokens.
3. **Embedding:** Tokens are converted into numerical vector representations.
4. **Processing:** The Transformer architecture processes the tokens using mechanisms such as self-attention to understand their relationships and context.
5. **Prediction:** The model predicts the probability of the next token based on the previous context.
6. **Generation:** Tokens are generated one after another until the response is complete or a stopping condition is reached.

### Context Window

An LLM's **context window** refers to the maximum amount of information, measured in tokens, that the model can process in a single interaction or request.

**Points to remember:**

- **Token Limit:** The context window has a maximum token capacity. Tokens include the input prompt, conversation history, retrieved documents, and sometimes the generated response.
- **Limited Capacity:** If the information provided exceeds the model's context window, the entire input cannot be processed at once. The system may need to truncate, summarize, or select relevant information.
- **Not Permanent Memory:** The context window is not the same as the model's long-term memory. It represents the information available to the model for its current processing.
- **Model Differences:** Different models have different context window sizes, and these limits can vary between model versions.

**Why is context window needed in RAG?**

Imagine an organization has thousands of documents containing millions of tokens. It is not practical to send all these documents to an LLM every time a user asks a question. The information may exceed the model's context window, and processing unnecessary content can also increase cost and response time. RAG solves this by retrieving only the most relevant document chunks and providing them to the LLM.

## Difference between RAG and LLM

| RAG | LLM |
|---|---|
| A technique or system that combines retrieval with generation. | An AI model trained to understand and generate language. |
| Retrieves relevant information from external sources. | Generates responses based on learned patterns and provided context. |
| Can access private or updated external data. | Does not automatically know an organization's private data. |
| Uses components such as embedding models, retrievers, and vector databases. | Commonly uses the Transformer architecture. |
| Provides relevant external context to the LLM. | Uses the provided context to generate a response. |

## Working of RAG with an LLM

1. Documents are collected and divided into smaller chunks.
2. An embedding model converts the chunks into vector representations.
3. The embeddings and related document information are stored in a vector database.
4. A user submits a question.
5. The question is converted into an embedding.
6. The retriever searches for the most relevant document chunks.
7. The retrieved chunks are combined with the user's question.
8. The LLM uses the provided context to generate the final answer.

**RAG = Information Retrieval + Context Augmentation + LLM Generation**

---

*Notes | Day 23*