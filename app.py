import os
import openai
import gradio as gr

# Get your OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(prompt):
    """
    Simple function to get response from OpenAI's GPT-3.5/4 API
    """
    if not prompt:
        return "Please enter a prompt!"
    
    try:
        response = openai.ChatCompletion.create(
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
    description="A minimal OpenAI GPT chat app ready for Render deployment."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)

