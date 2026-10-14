# Machine Learning and Types of ML

## What is Machine Learning?

**Machine Learning** is a computer science, data science and artificial intelligence (AI) subset that enables systems to learn and improve from data without additional programming interventions. It can also make predictions, identify patterns and learn from feedback.

## Lifecyecle of Machine Learning

### 1. Problem Definition
Identify the problem to be solved using Machine Learning and define the objectives and expected outcomes.

### 2. Data Collection
Collect relevant data from sources such as databases, CSV files, APIs, sensors, or websites.

### 3. Data Preprocessing
Clean the collected data by handling missing values, removing duplicates, correcting errors, and converting data into a suitable format.

### 4. Exploratory Data Analysis (EDA)
Analyze and visualize the dataset to understand its structure, distributions, patterns, relationships, and outliers.

### 5. Feature Engineering and Selection
Create useful features from existing data and select the most relevant features to improve model performance.

### 6. Model Selection
Choose a suitable Machine Learning algorithm based on the problem, such as Linear Regression, Decision Trees, or Random Forest.

### 7. Model Training
Train the selected model using the training dataset so it can learn patterns and relationships from the data.

### 8. Model Evaluation
Evaluate the trained model using appropriate metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the problem.

### 9. Model Deployment
Deploy the trained model into a real-world application or system so users can make predictions using new data.

### 10. Monitoring and Maintenance
Monitor the model's performance after deployment, detect changes in data or prediction quality, and retrain or update the model when necessary.

## Types of Machine Learning

### Supervised Learning

**Supervised Learning** is a type of machine learning where the model is trained on  labeled data. It is commonly used for risk assessment, image recognition, predictive analytics and fraud detection, and comprises several types of algorithms.

- **Linear Regression:** Used to predict continuous values (e.g., price, temperature). It is simple and widely used.
- **Logistic Regression:** It is a type of classification algorithm that is used to predict a binary output variable.
- **SVM (Support Vector Machines):** Separates data into classes using a boundary (hyperplane). Support vectors help define this boundary.
- **K-Nearest Neighbors:** Predicts based on the closest data points. Results depend on the value of k and distance measure.
- **Decision Trees:** Uses a tree-like structure where each node represents a decision and each leaf represents an outcome.
- **Random Forests:** Combines multiple decision trees to improve accuracy and reduce overfitting.

### Unsupervised Learning

**Unsupervised Learning** is a type of machine learning where the model is trained on unlabeled data. It is commonly used for customer segmentation, anomaly detection, image and text clustering, recommendation systems, and comprises several types of algorithms.

- **K-means Clustering:** Groups data into K clusters based on how close the points are to each other.
- **Hierarchical Clustering:** Creates clusters by building a tree step-by-step, either merging or splitting groups.
- **Apriori Algorithm:** Finds patterns by exploring frequent item combinations step-by-step.

### Semi-supervised Learning

**Semi-supervised Learning** is a type of machine learning approach that uses a small amount of labeled data and a large amount of unlabeled data to train predictive models. The goal is to learn a function that accurately predicts outputs based on inputs, similar to supervised learning, but with much less labelled data.

### Reinforcement Learning

**Reinforcement Learning** is a type of machine learning where an agent learns to make decisions through trial and error. It allows machines to learn by interacting with an environment and receiving feedback based on their actions. This feedback comes in the form of rewards or penalties. It is commonly used for robotics for automated tasks, personalized training systems, and comprises several types of algorithms.

- **Q-Learning:** is a method where an AI learns by building a giant score sheet. It lists every possible situation and every possible action, then updates the scores based on trial and error until it knows the best move for every single spot.

- **Deep Q-Networks (DQN):** is a method where the giant score sheet is replaced by an AI brain (a neural network). Instead of looking up scores in a massive list, the AI looks at the entire situation—like a video game screen—and uses its brain to guess the best move instantly.

---

*Notes | Day 3*