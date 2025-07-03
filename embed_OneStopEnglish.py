"""
Embed Text from OneStopEnglish dataset to classify reading level of a text
"""

import pandas as pd  # For data manipulation
from sentence_transformers import SentenceTransformer  # Embedding model

# Read the OneStopEnglish dataset
df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Calculate the embeddings for text and paragraphs
df["text_embedding"] = model.encode(
    df["text"].tolist(), batch_size=96, show_progress_bar=True, convert_to_numpy=True
).tolist()

# Print information about the dataframe
print(df.info())
print(df.sample())

# Save the dataframe
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_embed.parquet", index=False)
