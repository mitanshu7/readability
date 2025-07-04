import gradio as gr
import pandas as pd
from dotenv import dotenv_values

# Configuration
config = dotenv_values(".env")
LLM_MODEL = config["LLM_MODEL"]

# Read the dataset for demo
df = pd.read_parquet(
    f"datasets/OneStopEnglish/{LLM_MODEL}/OneStopEnglish.parquet"
)


# Function to show a random sample
def gen_sample():
    row = df.sample(1)

    text_output =str(row["text"].values[0])
    level_output = str(row["level"].values[0])
    trip_words_output = str(row["trip_words"].values[0])
    generated_output = str(row["rewritten_text"].values[0])

    return text_output, level_output, trip_words_output, generated_output


# Gradio Interface
with gr.Blocks() as demo:
    # Header
    gr.Markdown("# 📝 Text Generator")
    gr.Markdown("### Keeps the readability of the original text. Preserves Trip words.")

    with gr.Accordion("ℹ️ Dataset Info", open=False):
        gr.Markdown(
            "Uses the [OneStopEnglish](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) dataset for classification and generation guidance."
        )
        gr.Markdown(
            f"The LLM used for this demo is: **[{LLM_MODEL}](https://openrouter.ai/{LLM_MODEL})**"
        )

    # Generate Button
    with gr.Row():
        generate_btn = gr.Button("🔁 Generate Random Sample", scale=1)

    # Text Display Area
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 📄 Original Text")
            text_output = gr.Markdown("*(Original text will appear here)*")

            gr.Markdown("#### 📊 Reading Level")
            level_output = gr.Markdown("*(Reading level will appear here)*")

            gr.Markdown("#### 🧩 Trip Words")
            trip_words_output = gr.Markdown("*(Trip words will appear here)*")

        with gr.Column():
            gr.Markdown("#### ✨ Generated Text")
            generated_output = gr.Markdown("*(Generated text will appear here)*")

    # Define click event for generate button
    generate_btn.click(
        gen_sample,
        inputs=[],
        outputs=[text_output, level_output, trip_words_output, generated_output],
    )

# Launch the Gradio interface
demo.launch(server_port=7895)
