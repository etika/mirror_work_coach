import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import gradio as gr

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Conversation memory
memory = ConversationBufferMemory(input_key="input", output_key="output", memory_key="chat_history")

# Simple prompt template
template = """The following is a conversation with an AI assistant.
{chat_history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=template
)

def respond(user_input, chat_history):
    # Generate AI response
    formatted_prompt = prompt.format(chat_history=chat_history or "", input=user_input)
    response = llm(formatted_prompt)
    if chat_history is None:
        chat_history = ""
    chat_history += f"Human: {user_input}\nAI: {response}\n"
    return chat_history, chat_history

# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Your message")
    txt.submit(respond, [txt, chatbot], [txt, chatbot])

# Launch with share=True for Render
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port, share=True)

