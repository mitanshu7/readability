from pickle import dump  # Save models

import pandas as pd  # For data manipulation
from sklearn.model_selection import (
    train_test_split,  # Split data into train and test sets
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler # For feature scaling

################################################################################
################################################################################

# Read the embeddings and readability scores
df_embed = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_embed.parquet")
df_readability = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_readability.parquet")

# Merge the embeddings and readability scores
df = pd.merge(df_embed, df_readability, on="filename", suffixes=('', '_right'))

# Drop the columns with _right suffixes
df = df.drop(columns=[col for col in df.columns if '_right' in col])

# Concatenate the embedding vectors and readability scores to one list
readability_score_names = ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index", "automated_readability_index", "coleman_liau_index", "dale_chall_readability_score", "linsear_write_formula", "gunning_fog"]

def merge_features(row: pd.Series) -> list[float]:
    
    readability_scores = row[readability_score_names].tolist()
    
    embedding_vector = row["text_embedding"].tolist()
    
    merged_features = embedding_vector + readability_scores
    
    return merged_features

df["embedding_readability"] = df.apply(merge_features, axis=1)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

################################################################################
print('################################################################################')

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="linear")
svm.fit(train_df["embedding_readability"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability"].tolist(), test_df["level"]
)
print(f"Linear Kernel: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_linear.pkl", "wb") as f:
    dump(svm, f)

################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="rbf")
svm.fit(train_df["embedding_readability"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability"].tolist(), test_df["level"]
)
print(f"RBG Kernel: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_rbf.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="poly")
svm.fit(train_df["embedding_readability"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability"].tolist(), test_df["level"]
)
print(f"Polynomial Kernel: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_poly.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="sigmoid")
svm.fit(train_df["embedding_readability"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability"].tolist(), test_df["level"]
)
print(f"Sigmoid Kernel: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_sigmoid.pkl", "wb") as f:
    dump(svm, f)
################################################################################
################################################################################

# Scale the features
scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df['embedding_readability'].tolist())
test_df_scaled = scaler.transform(test_df['embedding_readability'].tolist())

################################################################################
print('################################################################################')

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="linear")
svm.fit(train_df_scaled, train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df_scaled, test_df["level"]
)
print(f"Linear Kernel, Scaled All: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_linear_scaled.pkl", "wb") as f:
    dump(svm, f)

################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="rbf")
svm.fit(train_df_scaled, train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df_scaled, test_df["level"]
)
print(f"RBF Kernel, Scaled All: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_rbf_scaled.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="poly")
svm.fit(train_df_scaled, train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df_scaled, test_df["level"]
)
print(f"Polynomial Kernel, Scaled All: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_poly_scaled.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="sigmoid")
svm.fit(train_df_scaled, train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df_scaled, test_df["level"]
)
print(f"Sigmoid Kernel, Scaled All: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_sigmoid_scaled.pkl", "wb") as f:
    dump(svm, f)
################################################################################
################################################################################

# Scale only the readability indice
readability_score_names_scaled = [f"{score_name}_scaled" for score_name in readability_score_names]

scaler = StandardScaler()
df[readability_score_names_scaled] = scaler.fit_transform(df[readability_score_names])

def merge_scaled_features(row: pd.Series) -> list[float]:
    
    readability_scores = row[readability_score_names_scaled].tolist()
    
    embedding_vector = row["text_embedding"].tolist()
    
    merged_features = embedding_vector + readability_scores
    
    return merged_features

df["embedding_readability_Rscaled"] = df.apply(merge_scaled_features, axis=1)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

################################################################################
print('################################################################################')

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="linear")
svm.fit(train_df["embedding_readability_Rscaled"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability_Rscaled"].tolist(), test_df["level"]
)
print(f"Linear Kernel, Scaled Readability: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_linear_Rscaled.pkl", "wb") as f:
    dump(svm, f)

################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="rbf")
svm.fit(train_df["embedding_readability_Rscaled"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability_Rscaled"].tolist(), test_df["level"]
)
print(f"RBG Kernel, Scaled Readability: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_rbf_Rscaled.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="poly")
svm.fit(train_df["embedding_readability_Rscaled"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability_Rscaled"].tolist(), test_df["level"]
)
print(f"Polynomial Kernel, Scaled Readability: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_poly_Rscaled.pkl", "wb") as f:
    dump(svm, f)
################################################################################

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="sigmoid")
svm.fit(train_df["embedding_readability_Rscaled"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_readability_Rscaled"].tolist(), test_df["level"]
)
print(f"Sigmoid Kernel, Scaled Readability: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_sigmoid_Rscaled.pkl", "wb") as f:
    dump(svm, f)

################################################################################
################################################################################

def merge_scaled_features_half_vector(row: pd.Series) -> list[float]:
    
    readability_scores = row[readability_score_names_scaled].tolist()
    
    embedding_vector = row["text_embedding"].tolist()
    
    embedding_vector_half = embedding_vector[:len(embedding_vector)//2]
    
    merged_features = embedding_vector_half + readability_scores
    
    return merged_features

df["embedding_half_readability_Rscaled"] = df.apply(merge_scaled_features_half_vector, axis=1)


# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

################################################################################
print('################################################################################')

# Train SVM for full text with Linear Kernel
svm = SVC(kernel="linear")
svm.fit(train_df["embedding_half_readability_Rscaled"].tolist(), train_df["level"])

# Find accuracy
text_svm_accuracy = svm.score(
    test_df["embedding_half_readability_Rscaled"].tolist(), test_df["level"]
)
print(f"Linear Kernel, Scaled Readability, Half vectors: {text_svm_accuracy:.3f}")

# Save the text svm model using pickle
with open("models/OneStopEnglish/OneStopEnglish_embed_half_readability_SVM_linear_Rscaled.pkl", "wb") as f:
    dump(svm, f)