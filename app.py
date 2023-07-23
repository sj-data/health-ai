import streamlit as st
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

# Tool definition
search = SerpAPIWrapper()

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

st.title("Healthcare Bot")

with st.container():

    medicine = st.multiselect(
        'Enter any medicines you are currently taking',
        ['Amlodipine', 'Anastrozole', 'Aspirin', 'Atorvastatin', 'Baloxavir', 'Clopidogrel', 'Exemestane', 'Glipizide',
         'Hydrochlorothiazide', 'Insulin', 'Letrozole', 'Lisinopril', 'Metformin', 'Metoprolol', 'Oseltamivir',
         'Peramivir', 'Sitagliptin', 'Tamoxifen', 'Zanamivir'],
    )

    routine = st.multiselect(
        'What routines do you engage in?',
        ['Avoid close contact with sick individuals', 'Cover mouth and nose when coughing/sneezing',
         'Eat a balanced diet with fruits and vegetables', 'Engage in regular physical activity',
         'Follow a heart-healthy diet', 'Follow a low-sodium diet', 'Get vaccinated yearly',
         'Limit alcohol consumption', 'Maintain a healthy diet with low sugar intake',
         'Maintain a healthy weight', 'Monitor blood glucose levels regularly',
         'Preventive Routine', 'Reduce sodium intake to lower blood pressure',
         'Take medications as prescribed', 'Take prescribed medications as directed', 'Wash hands frequently'],
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
        combined_input = f"Question: {question}\nMedicine: {medicine}\nRoutine: {routine}"

        # Use your agent to generate a response, passing in the combined input.
        response = agent_chain.run({"input": combined_input})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

