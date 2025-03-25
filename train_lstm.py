import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle

# Load tokenizer
with open("../models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1

# Load dataset
dataset = [
    "Artificial intelligence is the future of technology",
    "Machine learning powers modern AI applications",
    "Deep learning improves computer vision and NLP",
    "Neural networks mimic the human brain",
    "Data science is transforming industries",
    "AI chatbots use natural language processing",
    "Reinforcement learning enhances AI decision making",
    "Computer vision enables facial recognition",
    "The evolution of AI is accelerating",
    "AI ethics is crucial for responsible development"
]

# Prepare sequences
sequences = []
for line in dataset:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        sequences.append(token_list[:i+1])

# Padding
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

# Inputs and labels
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Model
def create_model():
    model = Sequential([
        Embedding(vocab_size, 50, input_length=max_sequence_length - 1),
        LSTM(100, return_sequences=True),
        LSTM(100),
        Dense(100, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.fit(X, y, epochs=50, verbose=2)

# Save model
model.save("../models/lstm_model.h5")
print("LSTM model saved successfully.")
