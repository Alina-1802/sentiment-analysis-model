import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("dane.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['text'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette="Set2")
plt.title("Distribution of emotion classes in the dataset")
plt.xlabel("Emotions")
plt.ylabel("Number of examples")
plt.xticks(rotation=45)
plt.savefig("distribution_emotion_classes.png")
plt.close()


df['text_length'] = df['text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
sns.histplot(df['text_length'], bins=30, color="teal", kde=True)
plt.title("Distribution of text length (in words)")
plt.xlabel("Number of words in the text")
plt.ylabel("Number of examples")
plt.savefig("distribution_text_length.png")
plt.close()


# ==================== PARAMETERS ====================

MAX_VOCAB_SIZE = 10000      # Vocabulary size (number of unique tokens)
MAX_SEQUENCE_LENGTH = 40    # Max sequence length (padded/truncated)
EMBEDDING_DIM = 100         # Embedding vector size
LSTM_UNITS = 32             # Number of LSTM units
DROPOUT_RATE = 0.2          # Dropout rate
TEST_SIZE = 0.2             # Test set ratio
EPOCHS = 5                  # Number of training epochs
BATCH_SIZE = 32             # Batch size

# ==================== DATA PREPARATION ====================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Tokenization
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Label encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])
labels_categorical = to_categorical(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels_categorical,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=42
)

actual_vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
print(f"Vocabulary size used (input_dim): {actual_vocab_size}")

# ==================== MODEL DEFINITION ====================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential()
model.add(Embedding(input_dim=actual_vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(LSTM_UNITS))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build model explicitly (needed for summary before training)
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
model.summary()

# ==================== TRAINING AND EVALUATION ====================


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.close()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Prediction
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()

print("Classification Report:\n")
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_true, y_pred, target_names=target_names))