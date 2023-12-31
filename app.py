import streamlit as st
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

# Tool definition
search = SerpAPIWrapper()

df = pd.read_csv('COMPLETE_DATASET_5_DISEASES 2.csv')
loader = CSVLoader(file_path='COMPLETE_DATASET_5_DISEASES 2.csv')
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
csvSearch=RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you cant find the answer inside the csv file, use it only as a last resort"
    ),
    Tool(
        name="DocSearch",
        func=csvSearch.run,
        description="If the user asks a health care question, this tool will search the csv file for the answer"
    )
]

# Memory definition
prefix = "You are a helpful chatbot that helps people with their health problems."
suffix = """Begin!

{chat_history}
{input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

# Agent definition
llm_chain = LLMChain(llm=OpenAI(temperature=0.3), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=False)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)
# chat with the agent

medlist=list(df['Medication'].unique())
routinelist=list(df['Preventive Routine'].unique())
diseaselist=list(df['Disease'].unique())


st.title("Healthcare Bot")

with st.container():

    medicine = st.multiselect(
        'Enter any medicines you are currently taking',
        medlist,
    )

    illness = st.multiselect(
        'What illness have you been diagnosed with?',
        diseaselist
    )
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.container():
    # React to user input
    if question := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Combine the user's question, medicine, and routine into a single string
        combined_input = f"Question: {question}\nMedicine: {medicine}\nIllness: {illness}"

        # Use your agent to generate a response, passing in the combined input.
        response = agent_chain.run({"input": combined_input})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

