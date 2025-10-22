from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import gradio as gr

# Initialize ChatOpenAI model
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"  # change to "gpt-4" if you have access
)

# Memory to maintain conversation context
memory = ConversationBufferMemory(
    input_key="input",
    output_key="output"
)

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Function to handle user input
def respond(user_input):
    return conversation.run(user_input)

# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(
        label="Your Message",
        placeholder="Type something..."
    )
    txt.submit(respond, inputs=txt, outputs=chatbot)

# Launch on Render
demo.launch(
    server_name="0.0.0.0",
    server_port=10000,
    share=True  # required on Render for external access
)

