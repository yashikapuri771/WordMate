import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('Next-Word-Predictor-Model.h5')

# App title
st.title("Next Word Predictor")
st.write("Type a sentence, and I'll suggest the next five words!")

# User input
input_text = st.text_input("Enter your text here:", "")

# Function to predict the next words
def predict_next_words(text, n=5):
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([sequence], maxlen=model.input_shape[1], padding='pre')
    predictions = model.predict(padded_sequence)[0]  # Get probability distribution
    top_indices = np.argsort(predictions)[-n:][::-1]  # Get top n indices

    suggestions = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                suggestions.append(word)
                break
    return suggestions

# Display suggestions
if input_text:
    suggestions = predict_next_words(input_text, n=5)
    if suggestions:
        st.markdown("### Suggestions:")
        for i, word in enumerate(suggestions, 1):
            st.markdown(f"**{i}. {input_text}  {word}**")
    else:
        st.write("Sorry, no suggestions available. Try again!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow.")
