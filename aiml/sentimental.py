# --------------------------------------------
# Sentiment Analysis using Neural Network (Keras)
# --------------------------------------------

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1️⃣ Dataset
# -----------------------------
data = {
    'review': [
        'I loved the movie, it was fantastic!',
        'What a great film, I will watch it again.',
        'An amazing experience, truly a masterpiece.',
        'I hated the movie, it was terrible.',
        'What a boring film, I will not watch it again.',
        'A dreadful experience, truly a disaster.'
    ],
    'sentiment': [
        'positive',
        'positive',
        'positive',
        'negative',
        'negative',
        'negative'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# 2️⃣ Convert text to numbers
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review']).toarray()

le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# -----------------------------
# 3️⃣ Build Neural Network
# -----------------------------
model = Sequential([
    Dense(8, input_dim=X.shape[1], activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------------
# 4️⃣ Train on all data (no split)
# -----------------------------
model.fit(X, y, epochs=500, batch_size=2, verbose=1)

# -----------------------------
# 5️⃣ Test prediction on new review
# -----------------------------
sample_reviews = ['I really enjoyed the movie, it was wonderful!']
sample_X = vectorizer.transform(sample_reviews).toarray()
predictions = model.predict(sample_X)

# -----------------------------
# 6️⃣ Display result
# -----------------------------
print("\n--- Sample Review Prediction ---")
print("Predicted Probability:", predictions[0][0])

# Fix: sentiment label logic
print("Predicted Sentiment:", 'positive' if predictions[0][0] < 0.5 else 'negative')
