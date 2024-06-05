import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load data
file_path = 'training_data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path).fillna('NaN')

required_columns = ['Input', 'Tag1', 'Tag2', 'Tag3']
print(df.columns)
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in the DataFrame.")
    
# Preprocess the data
inputs = df['Input'].values
tags = df[['Tag1', 'Tag2', 'Tag3']].values

# Encode tags as integers
encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags.ravel()).reshape(tags.shape)

# Tokenize inputs
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(inputs)
sequences = tokenizer.texts_to_sequences(inputs)
max_sequence_length = max(len(x) for x in sequences)
input_data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(input_data, encoded_tags, test_size=0.3, random_state=42)


input_layer = tf.keras.layers.Input(shape=(max_sequence_length,))
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=max_sequence_length)(input_layer)
lstm_layer = tf.keras.layers.LSTM(128, return_sequences=True)(embedding_layer)
dropout_layer = tf.keras.layers.Dropout(0.5)(lstm_layer)
lstm_layer_2 = tf.keras.layers.LSTM(64)(dropout_layer)

# Output layers for each tag
output1 = tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')(lstm_layer_2)
output2 = tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')(lstm_layer_2)
output3 = tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')(lstm_layer_2)

model = tf.keras.models.Model(inputs=input_layer, outputs=[output1, output2, output3])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy','accuracy','accuracy'])

# Train the model
history = model.fit(X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]], epochs=10, validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]), batch_size=32)

# Predicting tags for a new input
#def predict_tags(input_text):
    #seq = tokenizer.texts_to_sequences([input_text])
    #padded = pad_sequences(seq, maxlen=max_sequence_length)
    #pred = model.predict(padded)
    #return encoder.inverse_transform(pred)

