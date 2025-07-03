import pandas as pd
from umap import UMAP

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")

# Initialize UMAP
reducer = UMAP(metric="euclidean")

# Fit and transform the data using UMAP
df["text_embedding_umap"] = reducer.fit_transform(
    df["text_embedding"].to_list()
).tolist()

# Save the DataFrame with UMAP coordinates
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_umap.parquet", index=False)
