"""
Validate generated text for readability
"""

# Verify the required level of text here

from pickle import load

import nltk
import textstat
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Download the punkt tokenizer model
nltk.download("punkt_tab")

# Load the classifier
svm_model_name = "models/OneStopEnglish/OneStopEnglish_embed_readability_SVM_linear.pkl"
with open(svm_model_name, "rb") as file:
    svm_model = load(file)

# Load the embedding models
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


# Function to create readability scores on the text
# Function to create readability scores on the text
def create_readability_scores(text: str) -> list[float]:
    # Calculate readability scores using textstat library
    scores = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "gunning_fog": textstat.gunning_fog(text),
    }

    return list(scores.values())


# Function to predict the class of input text
def predict_class(text: str) -> str:
    # Embed the text
    text_embedding = embedding_model.encode(text, convert_to_numpy=True)

    # Readability scores
    readability_scores = create_readability_scores(text)

    # Classifier input
    input_features = list(text_embedding) + readability_scores

    # Predict the class
    prediction = svm_model.predict([input_features])

    # Return the predicted class
    return prediction[0]


# Function to check whether the generated text belongs to the same level
def validate_level(text: str, level: str) -> bool:
    # Predict the class
    prediction = predict_class(text)

    # Check if the prediction matches the level
    return prediction == level


# Function to check where the generated text has the required trip words
def validate_trip_words(text: str, trip_words: list) -> bool:
    # Tokenize the text
    words = word_tokenize(text)

    # Make all text lower case and make the list unique
    words = [word.lower() for word in words]
    words = list(set(words))

    # Make all trip words lower case
    trip_words = [trip_word.lower() for trip_word in trip_words]

    return all(word in words for word in trip_words)


# Function to give feedback to llm, in case the generated text is insufficient
def feedback(text: str, level: str, trip_words: list) -> str:
    # Predict the class
    prediction = predict_class(text)

    # Check if the prediction matches the level
    if prediction != level:
        return f"""The generated text does not match the required readability level.
    Detected Level: {prediction}
    Expected Level: {level}

    Please revise the text to match the expected level while preserving the original meaning and including all required Trip Words."""

    # Check if the text has the required trip words
    if not validate_trip_words(text, trip_words):
        return f"""The generated text is missing one or more required Trip Words.

    Required Trip Words: {", ".join(trip_words)}

    Please regenerate the text to ensure all Trip Words are naturally and appropriately included, while maintaining the intended readability level and original meaning."""

    # If the text passes both checks, return a positive feedback message
    return "pass"
