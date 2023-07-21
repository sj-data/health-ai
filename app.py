import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

hf_hub_download(repo_id="LLukas22/gpt4all-lora-quantized-ggjt", filename="ggjt-model.bin", local_dir=".")
llm = Llama(model_path="./ggjt-model.bin")

ins = '''### Instruction:
{}
### Response:
'''

fixed_instruction = "You are a healthcare bot designed to give advice for the prevention and treatment of various illnesses."

def respond(message):
    full_instruction = fixed_instruction + " " + message
    formatted_instruction = ins.format(full_instruction)
    bot_message = llm(formatted_instruction, stop=['### Instruction:', '### End'])
    bot_message = bot_message['choices'][0]['text']
    return bot_message

st.title("Healthcare Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = respond(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
