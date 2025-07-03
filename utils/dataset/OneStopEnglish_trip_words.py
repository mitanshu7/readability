import random  # For random selection of words

import nltk
import pandas as pd  # For data manipulation
from nltk.tokenize import word_tokenize

# Download the punkt tokenizer model
nltk.download("punkt_tab")


# Read the OneStopEnglish dataset
df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")


# Function to randomly select 5 words of a given length or MemoryError
def create_trip_words(row: pd.DataFrame) -> list[str]:
    # Create a dictionary with key as the word length and value as a list of those words
    words = word_tokenize(row["text"])

    # Empty dictionary
    word_dict = {}
    # Iterate over words to populate the dictionary
    for word in words:
        if len(word) not in word_dict:
            word_dict[len(word)] = []
        word_dict[len(word)].append(word)

    # Create tough words list depending on the level
    if row["level"] == "Elementary":
        # Initialize tough_words list
        tough_words = []

        # Iterate over word_dict keys
        for word_length in word_dict:
            # Filter out words with length more than 6.
            # 6 is a special value chosen by analysing the nlp_task.doc, modify accordingly
            if word_length >= 6:
                tough_words.extend(word_dict[word_length])

                # Make the list unique
                tough_words = list(set(tough_words))

        return random.sample(tough_words, 5)

    elif row["level"] == "Intermediate":
        # Initialize tough_words list
        tough_words = []

        # Iterate over word_dict keys
        for word_length in word_dict:
            # Filter out words with length more than 6.
            # 6 is a special value chosen by analysing the nlp_task.doc, modify accordingly
            if word_length >= 7:
                tough_words.extend(word_dict[word_length])

                # Make the list unique
                tough_words = list(set(tough_words))

        return random.sample(tough_words, 5)

    elif row["level"] == "Advanced":
        # Initialize tough_words list
        tough_words = []

        # Iterate over word_dict keys
        for word_length in word_dict:
            # Filter out words with length more than 6.
            # 6 is a special value chosen by analysing the nlp_task.doc, modify accordingly
            if word_length >= 8:
                tough_words.extend(word_dict[word_length])

                # Make the list unique
                tough_words = list(set(tough_words))

        return random.sample(tough_words, 5)


# Create trip words list
df["trip_words"] = df.apply(create_trip_words, axis=1)

# Print information about the dataframe
print(df.info())
print(df.sample())


# Save the dataset
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_trip_words.parquet", index=False)
