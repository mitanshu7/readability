Structure of project
1. `main.py` handles the main interface.
2. `generate.py` contains helper functions to generate LLM responses.
3. `validate.py` handles the validation of the LLM generated text

Note: To run `umap_OneStopEnglish.py`, first run `uvx --with umap-learn,pandas,pyarrow --python 3.9 python`, then copy paste the script contents. This avoids having to downgrade python to 3.9 only for umap.
