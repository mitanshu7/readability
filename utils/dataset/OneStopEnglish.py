from glob import glob  # Gather files

import pandas as pd  # For data manipulation

# Gather files from the dataset
OneStopEnglish = glob("datasets/OneStopEnglish/**/*.txt")
OneStopEnglish.sort()

print(f"Gathered {len(OneStopEnglish)} files")


# Function to extract Level of text
def extract_level(filename: str) -> str:
    # Get foldername for the level
    foldername = filename.split("/")[-2]

    # Get level name from folder, e.g. Adv, Ele, or Int.
    level = foldername.split("-")[0]

    # Transform the short level names to real names
    level_names = {"Ele": "Elementary", "Int": "Intermediate", "Adv": "Advanced"}
    level = level_names.get(level, level)

    return level


# Function to extract text from files
def extract_text(filename: str) -> str:
    # Read the contents of the file
    with open(filename, "r") as f:
        text = f.read()

    # Remove the first line, since it only contains the level of the text
    text = text.split("\n", 1)[1]

    return text


# Extract title
def extract_title(filename: str) -> str:
    # Extract the title from the filename
    title = filename.split("/")[-1].split("-")[0]

    return title


# Generate a pandas dataframe for classification
df = pd.DataFrame({"filename": OneStopEnglish})

# Extract title
df["title"] = df["filename"].apply(extract_title)

# Extract level
df["level"] = df["filename"].apply(extract_level)

# Extract the text from the file
df["text"] = df["filename"].apply(extract_text)

# Print information about the dataframe
print(df.info())
print(df.sample())

# Save the dataframe to a parquet file
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet", index=False)
