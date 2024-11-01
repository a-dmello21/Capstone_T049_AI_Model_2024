import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('trained_model.h5')

# Load and prepare the LabelEncoder and Tokenizer
encoder = LabelEncoder()
encoder.classes_ = np.load('label_classes.npy')  # Ensure the label classes are saved during training

# Recreate the tokenizer with the same configuration used during training
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = np.load('tokenizer_word_index.npy', allow_pickle=True).item()

# Define a function for inference
def predict_tags(input_text):
    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([input_text])
    input_data = pad_sequences(sequence, maxlen=model.input_shape[1])  # Ensure the same maxlen as training

    # Get predictions
    predictions = model.predict(input_data)
    predicted_tags = [encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

    return predicted_tags

# Example usage
input_text = "Sample input text for the model"  # Replace with your input
predicted_tags = predict_tags(input_text)
print("Predicted Tags:", predicted_tags)
