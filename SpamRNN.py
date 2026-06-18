from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from tensorflow.keras import layers
import random


SEED = 96
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


df = pd.read_csv("example_spam.csv", sep=",", encoding='latin-1')

df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.rename(columns={"v1":"label", "v2": "text"}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])

df['text_clean'] = df['text'].apply(remove_stopwords)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=100
)

encoder.adapt(train_df['text'])
encoded_example = encoder(train_df['text']).numpy()

train_df = train_df.dropna(subset=['text'])
train_df = train_df[train_df['text'].str.strip() != ""]

sns.countplot(x='label', data=train_df)
# plt.show()

df_majority = train_df[train_df['label'] == 0]
df_minority = train_df[train_df['label'] == 1]

df_majority_downsampled = df_majority.sample(n=len(df_minority))
train_df = pd.concat([df_majority_downsampled, df_minority])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

sns.countplot(x='label', data=train_df)
# plt.show()

model = tf.keras.Sequential([
    encoder,
    layers.Embedding(
        input_dim=1000,
        output_dim=100,
        mask_zero=True
    ),
    layers.SimpleRNN(10),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adamw',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    tf.convert_to_tensor(train_df['text'].values, dtype=tf.string),
    train_df['label'].values.astype('float32'),
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

x_test = tf.constant(test_df['text'].tolist())
y_test = tf.constant(test_df['label'].tolist())


print("Evaluating on test data...")
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(x_test)
y_pred_labels = (y_pred_probs > 0.5).astype(int)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

model.save("spam_model.keras")