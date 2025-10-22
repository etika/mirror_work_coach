# app.py
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import gradio as gr

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize LLM and memory
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(input_key="input", output_key="output")
conversation = ConversationChain(llm=llm, memory=memory)

# Define respond function
def respond(user_input):
    response = conversation.predict(input=user_input)
    return response

# Build Gradio interface
with gr.Blocks() as demo:
    txt = gr.Textbox(label="Your Message")
    output = gr.Textbox(label="AI Response")
    txt.submit(respond, txt, output)

# Use server_name="0.0.0.0" for Render, share=True for public link
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port, share=True)

