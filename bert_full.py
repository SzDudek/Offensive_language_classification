import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

print(tf.__version__)
# -------------------------------
# ALBERT handles
# -------------------------------
TFHUB_PREPROCESS = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
TFHUB_ENCODER = "https://tfhub.dev/tensorflow/albert_en_base/3"

# -------------------------------
# Build ALBERT classifier
# -------------------------------
def build_albert_classifier():
    text_input = tf.keras.layers.Input(
        shape=(),
        dtype=tf.string,
        name="text"
    )

    preprocess = hub.KerasLayer(
        TFHUB_PREPROCESS,
        name="preprocess"
    )

    encoder = hub.KerasLayer(
        TFHUB_ENCODER,
        trainable=True,
        name="albert"
    )

    encoder_inputs = preprocess(text_input)
    outputs = encoder(encoder_inputs)

    x = outputs["pooled_output"]
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs=text_input,
        outputs=output
    )

    return model, encoder

# -------------------------------
# Load data
# -------------------------------
data = pd.read_csv("HateSpeechDataset.csv")
data = data[data["Label"].isin(["0", "1"])]

X = data["Content"].reset_index(drop=True)
y = data["Label"].astype(int).reset_index(drop=True)

# (Optional) subsample for speed
X = X.iloc[::10]
y = y.iloc[::10]

# -------------------------------
# Cross-validation
# -------------------------------
rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=1,
    random_state=42
)

bac_scores = []

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y), start=1):
    print(f"\n===== Fold {fold} =====")

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # Build and compile model (NEW MODEL PER FOLD)
    model, encoder = build_albert_classifier()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = tf.metrics.BinaryAccuracy()

    batch_size = 32
    epochs = 5
    steps_per_epoch = len(X_tr) // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=loss,
        metrics=metrics
    )

    # Optional: handle imbalance
    class_weight = {0: 1.0, 1: 5.0}

    # Train
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=1
    )

    encoder.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=metrics
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weight,
        verbose=1
    )

    # Predict probabilities
    y_val_probs = model.predict(X_val, batch_size=batch_size)

    # Convert to labels
    y_val_pred = (y_val_probs > 0.5).astype(int).ravel()

    # Balanced accuracy
    bac = balanced_accuracy_score(y_val, y_val_pred)
    bac_scores.append(bac)

    print(f"Fold {fold} BAC: {bac:.4f}")

# -------------------------------
# Final results
# -------------------------------
mean_bac = np.mean(bac_scores)
std_bac = np.std(bac_scores)

print("\n===== Final Results =====")
print(f"Mean BAC: {mean_bac:.4f}")
print(f"Std  BAC: {std_bac:.4f}")
