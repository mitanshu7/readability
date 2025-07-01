"""
Embed Text from OneStopEnglish dataset to classify reading level of a text
"""

from glob import glob  # Gather files

import pandas as pd  # For data manipulation
from sentence_transformers import SentenceTransformer  # Embedding model

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

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Calculate the embeddings for text and paragraphs
df["text_embedding"] = model.encode(
    df["text"].tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True
).tolist()

df["paragraph_embedding"] = model.encode(
    df["paragraph"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
).tolist()

# Save the dataframe
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet", index=False)
