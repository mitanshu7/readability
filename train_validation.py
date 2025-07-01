"""
Train Models to classify reading level of a text
"""

from pickle import dump  # Save models

import pandas as pd  # For data manipulation
from sklearn.model_selection import (
    train_test_split,  # Split data into train and test sets
)
from sklearn.svm import SVC  # Classification model

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train SVM
text_svm = SVC()
text_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = text_svm.score(test_df["text_embedding"].tolist(), test_df["level"])
print(f"Text SVM Accuracy: {text_svm_accuracy}")

# Train SVM for paragraphs
paragraph_svm = SVC()
paragraph_svm.fit(train_df["paragraph_embedding"].tolist(), train_df["level"])

# Find accuracy
paragraph_svm_accuracy = paragraph_svm.score(
    test_df["paragraph_embedding"].tolist(), test_df["level"]
)
print(f"Paragraph SVM Accuracy: {paragraph_svm_accuracy}")

# Save the svm models using pickle
with open("models/OneStopEnglist/text_svm_model.pkl", "wb") as f:
    dump(text_svm, f)

with open("models/OneStopEnglist/paragraph_svm_model.pkl", "wb") as f:
    dump(paragraph_svm, f)
