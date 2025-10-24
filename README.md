# Next-Word Prediction using LSTM

![Screenshot 2024-12-21 184842](https://github.com/user-attachments/assets/c253d428-afb2-4690-bee8-ce3a65cb234e)

This project demonstrates the use of **Long Short-Term Memory (LSTM)** networks for **Next-Word Prediction**, a natural language processing (NLP) task. The goal is to predict the next word in a given sequence based on previous context, using a deep learning approach.


## Learning Approach

### 1. **Problem Definition**
The task is to predict the next word in a sequence of words. Given a sequence of words, we want to build a model that can predict the most probable next word. This is commonly used in various NLP applications such as text completion, chatbots, and language models.

### 2. **Data Representation**
To make the model learn and predict, the text data must be represented in a form that a neural network can understand. The text is preprocessed in the following manner:

- **Tokenization**: The text is broken down into individual words (tokens). Each word is assigned a unique integer index.
- **Sequence Creation**: A sliding window approach is used to create sequences of words. For each sequence, the last word is used as the target label (i.e., the next word), while the preceding words serve as the input.
- **Padding**: Since the sequences may have different lengths, padding is used to make all input sequences of the same length.

### 3. **Model Choice: LSTM (Long Short-Term Memory)**
The core of the model is an **LSTM**, a type of **Recurrent Neural Network (RNN)**. LSTM is chosen because:

- **Capturing Sequential Dependencies**: Unlike standard feedforward neural networks, RNNs (and particularly LSTMs) are capable of capturing sequential patterns in data. This is crucial for text, where the meaning of a word is often dependent on its context (previous words).
- **Handling Long-Range Dependencies**: LSTMs are particularly effective at remembering long-term dependencies in sequences, addressing the vanishing gradient problem often encountered in traditional RNNs.

### 4. **Learning Process**
The learning process follows these key steps:

- **Model Training**: The LSTM model is trained on a large corpus of text data, learning to map input sequences of words to their respective next words. During training, the model adjusts its internal parameters (weights) to minimize the prediction error (loss).
  
- **Loss Function**: The model uses **categorical cross-entropy** as the loss function. This is suitable for classification tasks where each possible word (from the vocabulary) is considered a separate class.

- **Optimization**: The model is trained using the **Adam optimizer**, a popular gradient-based optimization method. Adam adapts the learning rate during training, helping the model converge efficiently.

### 5. **Hyperparameter Tuning**
To improve model performance, hyperparameters such as the number of LSTM units, the learning rate, and the embedding dimension are tuned. The process of **hyperparameter tuning** ensures that the model learns in the most effective way, by experimenting with different configurations and selecting the one that yields the best results.

### 6. **Prediction**
Once the model is trained, it can be used to predict the next word in a given sequence. The process works by:

- **Tokenizing the Input**: The input sequence is tokenized into integers, which are then passed to the model.
- **Prediction**: The model outputs a probability distribution over all possible words in the vocabulary. The word with the highest probability is selected as the predicted next word.
- **Word Generation**: This process can be repeated iteratively, where each predicted word is added to the input sequence, and the model predicts the next word in the sequence.

### 7. **Evaluation**
The performance of the model is evaluated using **accuracy** and **loss**. The accuracy metric measures how often the model's predicted word matches the actual next word. A well-trained model will have high accuracy and low loss, indicating that it is correctly predicting the next word in most cases.

## Conclusion
This project leverages the power of LSTM networks to predict the next word in a sequence, a task that is fundamental to many NLP applications. By training on a large dataset of text, the model learns to understand the structure of language and can generate meaningful predictions. The approach demonstrates the effectiveness of deep learning for sequence modeling and natural language understanding.

