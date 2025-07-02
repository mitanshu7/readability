from umap import UMAP
import pandas as pd

df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish.parquet")

# Drop paragraph related columns and duplicate rows
df.drop(["paragraph", "paragraph_embedding"], axis=1, inplace=True)
df.drop_duplicates(inplace=True, subset=['filename'], keep='first')

print(df.info())

# Initialize UMAP
reducer = UMAP(metric='euclidean')

# Fit and transform the data using UMAP
df["text_embedding_umap"] = reducer.fit_transform(df['text_embedding'].to_list()).tolist()
 
# Save the DataFrame with UMAP coordinates
df.to_parquet("datasets/OneStopEnglish/OneStopEnglish_umap.parquet", index=False)