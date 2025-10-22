import gradio as gr
from transformers import pipeline

# Use a tiny model to stay within Render 512MB
coach = pipeline("text-generation", model="distilgpt2", device=-1)

def mirror_work_coach(prompt):
    if not prompt.strip():
        return "Please enter a feeling or thought to reflect on."
    try:
        res = coach(f"Mirror work affirmation for: {prompt}\nAffirmation:", max_length=80, num_return_sequences=1)
        return res[0]['generated_text']
    except Exception as e:
        return f"Error: {e}"

iface = gr.Interface(
    fn=mirror_work_coach,
    inputs=gr.Textbox(label="What are you feeling today?", placeholder="I feel anxious about my future..."),
    outputs=gr.Textbox(label="Mirror Work Affirmation"),
    title="🪞 Mirror Work Coach",
    description="A gentle mirror work coach that generates self-love affirmations (CPU + free-tier friendly)."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)

