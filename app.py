import gradio as gr
import pandas as pd

df = pd.read_parquet("datasets/OneStopEnglish/meta-llama/Llama-3.2-3B-Instruct/OneStopEnglish.parquet")

def gen_sample():
    
    row = df.sample(1)
    
    text_output = row["text"].values[0]
    level_output = row["level"].values[0]
    trip_words_output = row["trip_words"].values[0]
    generated_output = row["rewritten_text"].values[0]
    
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

    # Generate Button
    with gr.Row():
        generate_btn = gr.Button("🔁 Generate Random Sample", scale=1)

    # Text Display Area
    with gr.Row():
        with gr.Column():
            text_output = gr.Textbox(label="📄 Original Text", lines=6)
            level_output = gr.Textbox(label="📊 Reading Level")
            trip_words_output = gr.Textbox(label="🧩 Trip Words")

        with gr.Column():
            generated_output = gr.Textbox(label="✨ Generated Text", lines=10)

    # Optional spacing or footer
    gr.Markdown("—")

    # Define click event for generate button
    generate_btn.click(
        gen_sample,
        inputs=[],
        outputs=[text_output, level_output, trip_words_output, generated_output],
    )

# Launch the Gradio interface
demo.launch(server_port=7895)