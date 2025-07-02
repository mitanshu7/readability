from glob import glob  # Gather files in a directory
import random # For random selection of words
import pandas as pd  # For data manipulation
from random import sample
import nltk
from nltk.tokenize import word_tokenize

# Download the punkt tokenizer model
nltk.download('punkt_tab')

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


# Generate a pandas dataframe for classification
df = pd.DataFrame({"filename": OneStopEnglish})

# Extract level
df["level"] = df.apply(extract_level, axis=1)

# Extract the text from the file
df["text"] = df.apply(extract_text, axis=1)


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
    if row["level"] == "Ele":
        
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
        
    elif row["level"] == "Int":
        
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
        
    elif  row["level"] == "Adv":
        
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

# Save the dataset
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_trip_words.parquet", index=False)
    