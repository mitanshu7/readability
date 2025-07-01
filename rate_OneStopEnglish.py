"""
Embed Text from OneStopEnglish dataset to classify reading level of a text
"""

from glob import glob  # Gather files

import pandas as pd  # For data manipulation
import textstat  # For readability scores

# Gather files from the dataset
OneStopEnglish = glob("datasets/OneStopEnglish/**/*.txt")
print(f"Gathered {len(OneStopEnglish)} files")


# Function to extract Level of text
def extract_level(row: pd.DataFrame) -> str:
    # Get foldername for the level
    foldername = row["filename"].split("/")[-2]

    # Get level name from folder, e.g. Adv, Ele, or Int.
    level = foldername.split("-")[0]

    return level


# Function to extract text from files
def extract_text(row: pd.DataFrame) -> str:
    # Read contents from file name
    filename = row["filename"]
    with open(filename, "r") as f:
        text = f.read()

    # Remove the first line, since it only contains the level of the text
    text = text.split("\n", 1)[1]

    return text


# Function to extract paragraphs from text
def extract_paragraph(text: str) -> list[str]:
    # Split text into paragraphs
    paragraphs = text.split("\n")

    # Remove empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]

    return paragraphs


# Function to create readability scores on the text
def create_readability_scores(text: str) -> list[int]:
    # Calculate readability scores using textstat library
    scores = [
        textstat.flesch_reading_ease(text),
        textstat.flesch_kincaid_grade(text),
        textstat.smog_index(text),
        textstat.automated_readability_index(text),
        textstat.coleman_liau_index(text),
        textstat.dale_chall_readability_score(text),
        textstat.linsear_write_formula(text),
        textstat.gunning_fog(text),
    ]

    return scores


# Generate a pandas dataframe for classification
df = pd.DataFrame({"filename": OneStopEnglish})

# Extract level
df["level"] = df.apply(extract_level, axis=1)

# Extract the text from the file
df["text"] = df.apply(extract_text, axis=1)

# Extract paragraph from text
df["paragraph"] = df["text"].apply(extract_paragraph)

# Create multiple rows from one
df = df.explode("paragraph")

# Reset Index
df = df.reset_index(drop=True)

# Create readability scores for text
df["text_readability_scores"] = df["text"].apply(create_readability_scores)

# Create readability scores for paragraph
df["paragraph_readability_scores"] = df["paragraph"].apply(create_readability_scores)

# Save the dataframe to a parquet file
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_readability.parquet", index=False)
