import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import gradio as gr

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM Setup ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

memory = ConversationBufferMemory(input_key="input", output_key="output")

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# --- Gradio UI ---
def respond(user_input):
    return conversation.run(user_input)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Type your message")
    txt.submit(respond, inputs=txt, outputs=chatbot)
    
    # Optional: Clear chat button
    clear_btn = gr.Button("Clear Chat")
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# --- Launch App ---
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port, share=True)

