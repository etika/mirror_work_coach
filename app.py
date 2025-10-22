import os
import gradio as gr
from openai import OpenAI

# Initialize client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_gpt(prompt):
    """
    Simple function to get response from OpenAI's GPT-3.5/4 API using the new 1.0+ interface
    """
    if not prompt:
        return "Please enter a prompt!"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Create Gradio Interface
demo = gr.Interface(
    fn=chat_with_gpt,
    inputs=gr.Textbox(label="Enter your prompt here"),
    outputs=gr.Textbox(label="GPT Response"),
    title="Simple LLM Chat App",
    description="A minimal OpenAI GPT chat app compatible with the latest API."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)

