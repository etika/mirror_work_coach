import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from gradio import Textbox, Chatbot, Blocks

# Initialize your LLM
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"  # change if needed
)

# Initialize conversation memory
memory = ConversationBufferMemory(input_key="input", output_key="output")

# Dummy respond function
def respond(user_input, chat_history):
    # Replace with actual LangChain logic
    response = f"Echo: {user_input}"
    chat_history.append((user_input, response))
    return "", chat_history

# Gradio UI
with Blocks() as demo:
    txt = Textbox(label="Your Message")
    chatbot = Chatbot(label="Chat History")
    
    # Wire the submit event
    txt.submit(respond, [txt, chatbot], [txt, chatbot])

# Server port for Render
port = int(os.environ.get("PORT", 10000))

# Launch Gradio with shareable link for Render
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=True
)

