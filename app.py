import gradio as gr
from transformers import pipeline

# Load a small local model for CPU
# For production, you can pick a small instruction-tuned model like "ehartford/WizardLM-7B-1.0-GPTQ-4bit" if you have GPU,
# or a tiny CPU-friendly model like "google/flan-t5-small"
# Here we use flan-t5-small for CPU usage
summarizer = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def mirror_work_coach(prompt):
    """
    Generates mirror work affirmations or self-love guidance
    """
    if not prompt.strip():
        return "Please enter your prompt or intention!"

    try:
        # Use model to generate guidance
        result = summarizer(f"Give a mirror work affirmation for: {prompt}", max_length=150)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating response: {e}"

# Gradio interface
iface = gr.Interface(
    fn=mirror_work_coach,
    inputs=gr.Textbox(label="Your intention or feeling", placeholder="I feel stressed..."),
    outputs=gr.Textbox(label="Mirror Work Guidance"),
    title="Mirror Work Coach",
    description="Enter a feeling or intention and get a self-love affirmation. Runs entirely on CPU!"
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)

