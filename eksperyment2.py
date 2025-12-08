import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -------------------------
# Load data
# -------------------------
data = pd.read_csv("HateSpeechDataset.csv")
data = data[data["Label"].isin(['0','1'])]

texts = data["Content"].astype(str).values
labels = data["Label"].astype(int).values

# -------------------------
# Tokenize text
# -------------------------
MAX_WORDS = 30000
MAX_LEN = 100  # max tokens per message

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, stratify=labels
)

# -------------------------
# Build network
# -------------------------
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")   # binary output
])

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(1e-3),
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64
)

# -------------------------
# Evaluate
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)