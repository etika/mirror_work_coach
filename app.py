import gradio as gr
from transformers import pipeline

# Load a lightweight local model (free, CPU-based)
coach = pipeline("text-generation", model="distilgpt2", device=-1)

def mirror_coach(prompt):
    """
    Generates a reflective, self-love oriented response using a local model.
    """
    if not prompt.strip():
        return "Please share a few thoughts or emotions you'd like to reflect on 💭"

    # Create a self-love style prompt
    intro = (
        "You are a kind, empathetic mirror work coach. "
        "Respond to the user's reflection with compassion and encouragement.\n\n"
        f"User: {prompt}\nCoach:"
    )

    response = coach(
        intro,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Extract and clean the generated text
    text = response[0]['generated_text']
    # Remove prompt repetition
    text = text.split("Coach:")[-1].strip()
    return text

# Gradio interface
demo = gr.Interface(
    fn=mirror_coach,
    inputs=gr.Textbox(label="✨ What do you see or feel when you look in the mirror?", lines=4),
    outputs=gr.Textbox(label="💖 Mirror Coach's Reflection", lines=6),
    title="🪞 Mirror Work Coach (Offline)",
    description=(
        "A gentle, self-reflective mirror work companion powered by a local model. "
        "No API key needed, 100% free and private."
    ),
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

