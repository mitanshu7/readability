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

################################################################################

# Train SVM for full text
text_svm = SVC()
text_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = text_svm.score(test_df["text_embedding"].tolist(), test_df["level"])
print(f"Text SVM Accuracy: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/text_svm_model.pkl", "wb") as f:
    dump(text_svm, f)

################################################################################

# Train SVM for paragraphs extracted from text
paragraph_svm = SVC()
paragraph_svm.fit(train_df["paragraph_embedding"].tolist(), train_df["level"])

# Find accuracy
paragraph_svm_accuracy = paragraph_svm.score(
    test_df["paragraph_embedding"].tolist(), test_df["level"]
)
print(f"Paragraph SVM Accuracy: {paragraph_svm_accuracy}")

# Save the svm models using pickle
with open("models/OneStopEnglish/paragraph_svm_model.pkl", "wb") as f:
    dump(paragraph_svm, f)

################################################################################

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_readability.parquet")

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train svm on readability scores of full text
readability_svm = SVC()
readability_svm.fit(train_df["text_readability_scores"].tolist(), train_df["level"])

# Find accuracy
readability_svm_accuracy = readability_svm.score(
    test_df["text_readability_scores"].tolist(), test_df["level"]
)
print(f"Readability SVM Accuracy: {readability_svm_accuracy}")

# Save the svm models using pickle
with open("models/OneStopEnglish/text_readability_svm_model.pkl", "wb") as f:
    dump(readability_svm, f)

################################################################################

# Train svm on readability scores of paragraphs
paragraph_readability_svm = SVC()
paragraph_readability_svm.fit(
    train_df["paragraph_readability_scores"].tolist(), train_df["level"]
)

# Find accuracy
paragraph_readability_svm_accuracy = paragraph_readability_svm.score(
    test_df["paragraph_readability_scores"].tolist(), test_df["level"]
)
print(f"Paragraph Readability SVM Accuracy: {paragraph_readability_svm_accuracy}")

# Save the svm models using pickle
with open("models/OneStopEnglish/paragraph_readability_svm_model.pkl", "wb") as f:
    dump(paragraph_readability_svm, f)
