from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import gradio as gr

# Initialize ChatOpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Memory to store conversation
memory = ConversationBufferMemory(input_key="input", output_key="output")

# Conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# Function to respond to user input
def respond(user_input):
    return conversation.run(user_input)

# Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Enter your message")
    txt.submit(respond, inputs=txt, outputs=chatbot)

# Launch Gradio app on Render
demo.launch(
    server_name="0.0.0.0",
    server_port=10000,
    share=True  # Important for Render
)

