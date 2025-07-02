"""
Generate answers with local and API models.
"""

import torch
from openai import OpenAI
from transformers import pipeline

# TODO: ADD Logger


def run_inference_local(
    model_name: str,
    messages: list[dict],
    max_new_tokens: int = 200,
    temperature: float = 0.1,
) -> str:
    """
    Function to run text generation inference locally using a specified model.

    Parameters:
    model_name (str): Name of the model to use
    messages (list[dict]): List of messages to generate text from
    max_new_tokens (int): Maximum length of the generated text (default: 100)
    temperature (float): Creativity of LLM from 0.0-1.0. Higher is less accurate.

    Returns:
    dict: The model response
    """

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    generator = pipeline("text-generation", model=model_name, device=device)

    # Get response
    response = generator(
        messages,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return response[0]["generated_text"]

def run_inference_api(
    url: str,
    api_key: str,
    model_name: str,
    messages: list[dict],
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """
    Function to query OpenAI compatible API with a prompt and return the response.

    Parameters:
    url (str): The OpenAI API endpoint URL
    api_key (str): Your OpenAI API key
    model_name (str): Name of the model to use
    messages (list[dict]): List of messages to send to the model
    max_new_tokens (int):  Maximum length of the generated text (default: 100)
    temperature (float): Creativity of LLM from 0.0-1.0. Higher is less accurate.

    Returns:
    dict: The API response converted from JSON
    """

    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_new_tokens,
        temperature=temperature,
    )

    return completion.choices[0].message.content
