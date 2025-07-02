"""
Validate generated text for readability
"""

# Verify the required level of text here
# send appropriate prompts back in case the levels mis match
#
from pickle import load

import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Download the punkt tokenizer model
nltk.download("punkt_tab")

# Load the classifier
svm_model_name = "models/OneStopEnglish/text_svm_linear_model.pkl"
with open(svm_model_name, "rb") as file:
    svm_model = load(file)

# Load the embedding models
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


# Function to predict the class of input text
def predict_class(text: str) -> str:
    # Embed the text
    text_embedding = embedding_model.encode(text)

    # Predict the class
    prediction = svm_model.predict([text_embedding])

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
    words = word_tokenize(text)
    return all(word in words for word in trip_words)


# Function to give feedback to llm, in case the generated text is insufficient
def feedback(text: str, level: str, trip_words: list) -> str:
    # Predict the class
    prediction = predict_class(text)

    # Check if the prediction matches the level
    if prediction != level:
        return f"""The generated text does not match the required level.
        The level of generated text is {prediction}, it should be {level}.
        Where Adv means Advanced, Int means Intermediate, and Ele means Elementary.
        Please generate new text with appropriate level and trip words."""

    # Check if the text has the required trip words
    if not validate_trip_words(text, trip_words):
        return f"""The generated text does not contain the required trip words. 
        It should contain the following words: {', '.join(trip_words)}.
        Please generate new text with appropriate level and trip words."""

    # If the text passes both checks, return a positive feedback message
    return "pass"
