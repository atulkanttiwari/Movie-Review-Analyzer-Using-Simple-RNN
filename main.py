# Step 1: Imports
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

MAX_FEATURES = 10000
MAX_LEN = 500

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple_rnn_imdb.keras')

# Step 2: Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    words = text.split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        if index is not None and index < MAX_FEATURES:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # <UNK>
    
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )
    
    return padded_review

# Step 3: Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        preprocessed_input = preprocess_text(user_input)
        
        prediction = model.predict(preprocessed_input)
        
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a movie review.')
