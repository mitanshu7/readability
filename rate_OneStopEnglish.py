"""
Embed Text from OneStopEnglish dataset to classify reading level of a text
"""

import pandas as pd  # For data manipulation
import textstat  # For readability scores

# Read the OneStopEnglish dataset
df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")


# Function to create readability scores on the text
def create_readability_scores(text: str) -> dict[str, int]:
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

    return scores


# Create readability scores for text
df_scores = pd.DataFrame(df["text"].apply(create_readability_scores).tolist())

# Concatenate the original dataframe with the scores dataframe
df = pd.concat([df, df_scores], axis=1)

# Print information about the dataframe
print(df.info())
print(df.sample())


# Save the dataframe to a parquet file
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_readability.parquet", index=False)
