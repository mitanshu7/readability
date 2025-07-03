from pickle import dump  # Save models

import pandas as pd  # For data manipulation
from sklearn.model_selection import (
    train_test_split,  # Split data into train and test sets
)
from sklearn.svm import SVC  # Classification model

################################################################################

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_embed.parquet")

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

################################################################################

# Train SVM for full text with RBF kernel
embed_svm = SVC(kernel="rbf")
embed_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df["text_embedding"].tolist(), test_df["level"]
)
print(f"Text SVM Accuracy with RBF Kernel: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_rbf.pkl", "wb") as f:
    dump(embed_svm, f)

################################################################################

# Train SVM for full text with Linear Kernel
embed_svm = SVC(kernel="linear")
embed_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df["text_embedding"].tolist(), test_df["level"]
)
print(f"Text SVM Accuracy with Linear Kernel: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_linear.pkl", "wb") as f:
    dump(embed_svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
embed_svm = SVC(kernel="poly")
embed_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df["text_embedding"].tolist(), test_df["level"]
)
print(f"Text SVM Accuracy with poly Kernel: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_poly.pkl", "wb") as f:
    dump(embed_svm, f)

################################################################################

# Train SVM for full text with Linear Kernel
embed_svm = SVC(kernel="sigmoid")
embed_svm.fit(train_df["text_embedding"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df["text_embedding"].tolist(), test_df["level"]
)
print(f"Text SVM Accuracy with sigmoid Kernel: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_sigmoid.pkl", "wb") as f:
    dump(embed_svm, f)
################################################################################

# Chop the vectors in half
train_df["text_embedding_mrl"] = train_df["text_embedding"].apply(
    lambda x: x[: len(x) // 2]
)
test_df["text_embedding_mrl"] = test_df["text_embedding"].apply(
    lambda x: x[: len(x) // 2]
)

# Train SVM for full text with Linear kernel
embed_svm = SVC(kernel="linear")
embed_svm.fit(train_df["text_embedding_mrl"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df["text_embedding_mrl"].tolist(), test_df["level"]
)
print(f"Text SVM Accuracy with Linear Kernel and Half Vectors: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_linear_half.pkl", "wb") as f:
    dump(embed_svm, f)

################################################################################

# Train SVM for full text with Linear Kernel, UMAP - 512

umap_df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_umap.parquet")

# Split data into train and test sets
train_df_umap, test_df_umap = train_test_split(umap_df, test_size=0.2, random_state=42)

embed_svm = SVC(kernel="linear")
embed_svm.fit(train_df_umap["text_embedding_umap"].tolist(), train_df_umap["level"])

# Find accuracy
text_svm_accuracy = embed_svm.score(
    test_df_umap["text_embedding_umap"].tolist(), test_df_umap["level"]
)
print(f"Text SVM Accuracy with Linear Kernel and UMAP: {text_svm_accuracy}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_SVM_linear_umap.pkl", "wb") as f:
    dump(embed_svm, f)
