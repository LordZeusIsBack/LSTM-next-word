import streamlit as st
import tensorflow as tf
from keras.models import load_model
from pickle import load
from keras.preprocessing.sequence import pad_sequences
import numpy as np


model = load_model('next_word_lstm.keras')

with open('tokenizer.pkl', 'rb') as fp: tokenizer = load(fp)

reversed_index = {index: value for value, index in tokenizer.word_index.items()}

def predict_next_word(model, tokenizer, text, max_sequence_length):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_length:
    token_list = token_list[-(max_sequence_length - 1):]
  token_list = pad_sequences([token_list], maxlen=max_sequence_length)
  predicted = model.predict(token_list)
  predicted_word_index = int(np.argmax(predicted, axis=1)[0])
  return reversed_index.get(predicted_word_index, None)

st.title('Next Word Predictor with LSTM')
input_text = st.text_input('Input text', "to be or not to be")
if st.button('Predict'):
    next_word = predict_next_word(model, tokenizer, input_text, model.input_shape[1])
    st.write(next_word)
