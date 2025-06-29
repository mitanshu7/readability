from dotenv import dotenv_values
from generate import run_inference_api, run_inference_local

config = dotenv_values(".env")

OPENROUTER_API_KEY = config["OPENROUTER_API_KEY"]

output = run_inference_api(
    "https://openrouter.ai/api/v1",
    OPENROUTER_API_KEY,
    "meta-llama/llama-3.2-1b-instruct:free",
    "2+2=?",
    # "Only tell me the result and have nothing else in the output",
)

print(output)

print("*" * 80)

output = run_inference_api(
    "https://openrouter.ai/api/v1",
    OPENROUTER_API_KEY,
    "meta-llama/llama-3.2-1b-instruct:free",
    "2+2=?",
    "Only tell me the result and have nothing else in the output",
)

print(output)

# print("*" * 80)

# output = run_inference_local(
#     "meta-llama/llama-3.2-1b-instruct",
#     "2+2=?",
#     # "Only tell me the result and have nothing else in the output",
# )

# print(output)

# print("*" * 80)

# output = run_inference_local(
#     "meta-llama/llama-3.2-1b-instruct",
#     "2+2=?",
#     "Only tell me the result and have nothing else in the output",
# )

# print(output)
