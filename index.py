# Cell 1: Install dependencies (run first)
%pip install numpy tensorflow transformers torch

# Cell 2: LSTM Implementation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
data = """
Artificial intelligence is transforming industries across the world.
Machine learning algorithms can now recognize patterns in data that humans would miss.
Natural language processing enables computers to understand human language.
Deep learning models use neural networks with many layers to solve complex problems.
AI applications range from healthcare diagnostics to autonomous vehicles.
The field of AI continues to advance rapidly with new breakthroughs each year.
"""

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
for line in data.split('\n'):
    if line.strip() == '':
        continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Prepare data
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train (with reduced epochs for demo)
model.fit(X, y, epochs=10, verbose=1)

# Generation function
def generate_text_lstm(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        seed_text += " " + predicted_word
    return seed_text

# Cell 3: GPT-2 Implementation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model (might take a while first time)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text_gpt(prompt, max_length=50):
    inputs = gpt_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Cell 4: Demonstration
print("=== LSTM Model ===")
print(generate_text_lstm("Artificial intelligence", next_words=10))

print("\n=== GPT-2 Model ===")
print(generate_text_gpt("Machine learning is", max_length=30))
