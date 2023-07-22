import streamlit as st
import openai
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangChainOpenAI
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

loader = CSVLoader(file_path='./COMPLETE_DATASET_5_DISEASES 2.csv')
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
chain = RetrievalQA.from_chain_type(llm=LangChainOpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

def respond(message, medicine, routine):
	full_message = f"{message}. The patient is taking {medicine} and their routine involves {routine}."
	response = chain({"question": full_message})
	return response['result']

st.title("Healthcare Bot")

with st.container():
	medicine = st.text_input("Enter any medicines you're currently taking")
	routine = st.text_input("Describe your current daily routine")

# Initialize chat history
if "messages" not in st.session_state:
	st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
        
with st.container():
	# React to user input
	if prompt := st.chat_input("What is your question?"):
		# Display user message in chat message container
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.messages.append({"role": "user", "content": prompt})

		response = respond(prompt, medicine, routine)
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})

