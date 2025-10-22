import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Use a small CPU-friendly model
MODEL_NAME = "distilgpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create a pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU-only
)

def chat_with_model(prompt):
    """
    Generate response from distilgpt2 (CPU-friendly).
    """
    if not prompt.strip():
        return "Please enter a prompt!"
    
    try:
        # Generate text with reasonable limits
        outputs = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
        return outputs[0]['generated_text']
    except Exception as e:
        return f"Error: {e}"

# Gradio interface
iface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(label="Enter your prompt here"),
    outputs=gr.Textbox(label="Model Response"),
    title="CPU-Friendly LLM Chat App",
    description="A lightweight text generation app using distilgpt2, ready for Render deployment."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)

