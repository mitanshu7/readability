from pickle import dump  # Save models

import pandas as pd  # For data manipulation
from sklearn.model_selection import (
    train_test_split,  # Split data into train and test sets
)
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.svm import SVC  # Classification model

################################################################################

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_readability.parquet")

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features
readability_scores_list = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "automated_readability_index",
    "coleman_liau_index",
    "dale_chall_readability_score",
    "linsear_write_formula",
    "gunning_fog",
]
train_features = train_df[readability_scores_list]
test_features = test_df[readability_scores_list]

# Extract the labels
train_labels = train_df["level"]
test_labels = test_df["level"]

################################################################################

# Train the model
model = SVC(kernel="linear")
model.fit(train_features, train_labels)

# Evaluate the model
accuracy = model.score(test_features, test_labels)
print(f"Linear Kernel: {accuracy:.3f}")

# Save the model
dump(
    model, open("models/OneStopEnglish/OneStopEnglish_readability_SVM_linear.pkl", "wb")
)

################################################################################

# Train the model
model = SVC(kernel="rbf")
model.fit(train_features, train_labels)

# Evaluate the model
accuracy = model.score(test_features, test_labels)
print(f"RBF Kernel: {accuracy:.3f}")

# Save the model
dump(model, open("models/OneStopEnglish/OneStopEnglish_readability_SVM_rbf.pkl", "wb"))

################################################################################

# Train the model
model = SVC(kernel="poly")
model.fit(train_features, train_labels)

# Evaluate the model
accuracy = model.score(test_features, test_labels)
print(f"Polynomial Kernel: {accuracy:.3f}")

# Save the model
dump(model, open("models/OneStopEnglish/OneStopEnglish_readability_SVM_poly.pkl", "wb"))

################################################################################

# Train the model
model = SVC(kernel="sigmoid")
model.fit(train_features, train_labels)

# Evaluate the model
accuracy = model.score(test_features, test_labels)
print(f"Sigmoid Kernel: {accuracy:.3f}")

# Save the model
dump(
    model,
    open("models/OneStopEnglish/OneStopEnglish_readability_SVM_sigmoid.pkl", "wb"),
)

################################################################################

# Scale the features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

################################################################################

# Train the model
model = SVC(kernel="linear")
model.fit(train_features_scaled, train_labels)

# Evaluate the model
accuracy = model.score(test_features_scaled, test_labels)
print(f"Linear Kernel, Scaled: {accuracy:.3f}")

# Save the model
dump(
    model,
    open(
        "models/OneStopEnglish/OneStopEnglish_readability_SVM_linear_scaled.pkl", "wb"
    ),
)

################################################################################

# Train the model
model = SVC(kernel="rbf")
model.fit(train_features_scaled, train_labels)

# Evaluate the model
accuracy = model.score(test_features_scaled, test_labels)
print(f"RBF Kernel, Scaled: {accuracy:.3f}")

# Save the model
dump(
    model,
    open("models/OneStopEnglish/OneStopEnglish_readability_SVM_rbf_scaled.pkl", "wb"),
)

################################################################################

# Train the model
model = SVC(kernel="poly")
model.fit(train_features_scaled, train_labels)

# Evaluate the model
accuracy = model.score(test_features_scaled, test_labels)
print(f"Polynomial Kernel, Scaled: {accuracy:.3f}")

# Save the model
dump(
    model,
    open("models/OneStopEnglish/OneStopEnglish_readability_SVM_poly_scaled.pkl", "wb"),
)

################################################################################

# Train the model
model = SVC(kernel="sigmoid")
model.fit(train_features_scaled, train_labels)

# Evaluate the model
accuracy = model.score(test_features_scaled, test_labels)
print(f"Sigmoid Kernel, Scaled: {accuracy:.3f}")

# Save the model
dump(
    model,
    open(
        "models/OneStopEnglish/OneStopEnglish_readability_SVM_sigmoid_scaled.pkl", "wb"
    ),
)
