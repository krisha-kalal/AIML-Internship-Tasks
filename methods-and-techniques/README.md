# Train-Test Split Method, Cross Validation Techniques, Vectorization Techniques: Count Vectorizer and TFIDF

- **Train-Test Split and Cross-Validation are techniques used to evaluate how well a machine learning model performs on unseen data and help detect overfitting.**

## Train-Test Split Method

A train-test split divides your dataset into two parts i.e, one for training and one for final testing. This method is the heplful when the amount of data is huge. 

### Bias

It is the error caused by a model being too simple and making strong assumptions, which leads to underfitting.

### Variance

It is the error caused by a model being too complex and sensitive to small changes in training data, which leads to overfitting.

- **The higher the variance, the lower the bais and vice versa. Both of them help diagnose why a machine learning model fails and guide how to tune its complexity.**

## Cross Validation

Cross-validation (specifically K-fold) repeatedly splits the data into multiple subsets to train and validate the model across several rounds.

**Techniques:**

- **Holdout Validation:** This is the simplest form of validation. You take your entire dataset and split it into two parts: one large chunk for training and a smaller chunk for testing.

- **LOOCV (Leave One Out):** This is the most extreme form of K-Fold. If you have N total data points, you set K = N. The model trains on every single data point except one, and tests itself on that solitary point. This repeats until every single individual data point has been left out once.

- **LPOCV (Leave P Out):** Similar to LOOCV, but instead of leaving out just 1 point, you leave out a custom number (P) of data points for testing. The model trains on N-P points and tests on P points. This is repeated for every possible mathematical combination of P points.

- **Stratified CV:** This is a specialized version of K-Fold used for imbalanced datasets (where one category vastly outnumbers another). It forces each fold to have the exact same percentage of categories as the original whole dataset.

- **K-Fold CV:** Instead of splitting the data just once, you split your dataset into K equal parts/folds. You run the training process K times. Each time, a different fold acts as the test set, while the remaining K-1 folds are used for training. Finally, you average the scores from all K runs.

- **Time-Series CV:** This technique uses a progressive window: you train on past data and test on the immediate future data.

## Difference between Cross validation Techniques

| Method | Best For | Advantages | Limitations |
|---|---|---|---|
| Train/Test Split | Large datasets | Fast | Depends on one split |
| K-Fold Cross-Validation | Most ML projects | Reliable evaluation | Slower |
| Stratified K-Fold | Classification | Preserves class balance | Slightly slower |
| Leave-One-Out | Tiny datasets | Maximum training data | Very slow |
| Time Series CV | Time-dependent data | Respects chronology | Cannot shuffle data |

## Vectorization

Vectorization means turning different types of data—like text, images, or computer code into numerical formats or scalable mathematical shapes that computers can process faster and understand more easily.

**Techniques:**

1. **Bag of Words (BoW):** It converts text into a vector representing the frequency of words, disregarding grammar and word order. It counts the occurrences of each word in a document and generates a vector based on these counts. 
  
  - **Count Vectorizer:** It is a practical tool in Python's scikit-learn library that automatically implements the BoW model.

3. **TF-IDF (Term Frequency-Inverse Document Frequency):** Multiplies term frequency (how often a word appears in a document) by inverse document frequency (how rare the word is in the whole dataset).

**TF-IDF = TF × IDF**

  - **Term Frequency (TF):**  
    TF = (Number of times a word appears in a document) / (Total number of words in the document)

  - **Inverse Document Frequency (IDF):**  
    IDF = log(Total number of documents / Number of documents containing the word)

---

*Notes | Day 7*