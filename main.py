import pandas as pd
from dotenv import dotenv_values
import ast

from generate import run_inference_api, run_inference_local
from validate import feedback

# Configuration
config = dotenv_values(".env")
LOCAL =  ast.literal_eval(config["LOCAL"])
API_URL = config["OPENROUTER_API_URL"]
API_KEY = config["OPENROUTER_API_KEY"]
LLM_MODEL = config["LLM_MODEL"]

# Read the trip_words dataset
df = pd.read_parquet("datasets/OneStopEnglish/OneStopEnglish_trip_words.parquet")


# Function to generate new text of the same level, but contains trip words
def main(row: pd.DataFrame, history: list[dict] = None) -> str:
    # Counter to not exceed recursion limit
    counter = 0

    # Extract information
    original_text = row["text"].values[0]
    level = row["level"].values[0]
    trip_words = row["trip_words"].values[0]

    # Messages list of converstaion
    messages = []

    # Append task information if history in None
    if history is None:
        # Setup system prompt
        system_prompt = """You are tasked to generate text that has similar content to the one provided below,
            is of the same readability level and must contain all the words from the trip words list.
            Only return the modified text and nothing else."""

        # Append system prompt to messages
        messages.append({"role": "system", "content": system_prompt})

        # Setup task prompt
        task_prompt = (
            f"Original Text: {original_text}\nLevel: {level}\nTrip Words: {trip_words}"
        )

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
        response = run_inference_api(API_URL, API_KEY, LLM_MODEL, messages, len(original_text))

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
        counter += 1
        if counter > 3:
            raise RecursionError("Recursion limit exceeded")
        print(f"Counter: {counter}")
        
        # Append generated response and feedback to messages
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Feedback: {response_feedback}"})

        # Call main function recursively with updated messages
        return main(row, messages)


# Test the main function
print(main(df.sample()))
