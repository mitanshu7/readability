import ast
import os

import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm  # Progress bar

from generate import run_inference_api, run_inference_local
from validate import feedback

tqdm.pandas()  # Progress bar for pandas

# Configuration
config = dotenv_values(".env")
LOCAL = ast.literal_eval(config["LOCAL"])
API_URL = config["API_URL"]
API_KEY = config["API_KEY"]
LLM_MODEL = config["LLM_MODEL"]


# Function to generate new text of the same level, but contains trip words
def main(row: pd.DataFrame, history: list[dict] = None, counter: int = 0) -> str:
    # Counter to not exceed recursion limit
    counter += 1
    print(f"Counter: {counter}")

    if counter > 5:
        print("!!!!!!!!!!!!! Recursion limit exceeded !!!!!!!!!!!!!")
        print(row["filename"])
        return None

    # Extract information
    original_text = row["text"]
    level = row["level"]
    trip_words = row["trip_words"]

    # Messages list of converstaion
    messages = []

    # Append task information if history in None
    if history is None:
        # Setup system prompt
        system_prompt = """
        You are a text transformation assistant. Your goal is to generate a rewritten version of a given text. The new version must:

        - Preserve the same general content and meaning as the original.
        - Match the same readability level as specified.
        - Include **all** words from the provided Trip Words list (in any contextually appropriate way).
        - Output **only** the rewritten text, with no extra explanations or commentary.
        """

        # Append system prompt to messages
        messages.append({"role": "system", "content": system_prompt})

        # Setup task prompt
        task_prompt = f"""
        Original Text:
        \"\"\"{original_text}\"\"\"

        Readability Level: {level}

        Trip Words: {", ".join(trip_words)}

        Please rewrite the text accordingly.
        """

        # Append and task prompt to messages
        messages.append({"role": "user", "content": task_prompt})

    # Append history to messages if not None
    if history is not None:
        messages.extend(history)

    print(f"Messages: {messages}")

    # Generate response
    if LOCAL:
        response = run_inference_local(LLM_MODEL, messages, len(original_text))
    else:
        response = run_inference_api(
            API_URL, API_KEY, LLM_MODEL, messages, len(original_text)
        )

    print(f"Generated Response: {response}")

    # Get feedback on generated text
    response_feedback = feedback(response, level, trip_words)
    print(f"Feedback: {response_feedback}")

    # Validate response
    if response_feedback == "pass":
        print("*************Generation Passed****************")
        return response
    else:
        print("#############Generation Failed################")

        # Append generated response and feedback to messages
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Feedback: {response_feedback}"})

        # Call main function recursively with updated messages
        return main(row, messages, counter)


# Read the trip_words dataset
df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_trip_words.parquet")

# Create rewritten dataset
df["rewritten_text"] = df.progress_apply(main, axis=1)

# Save the dataset
folder_name = f"datasets/OneStopEnglish/{LLM_MODEL}"
os.makedirs(folder_name, exist_ok=True)
df.to_parquet(f"{folder_name}/OneStopEnglish.parquet")
