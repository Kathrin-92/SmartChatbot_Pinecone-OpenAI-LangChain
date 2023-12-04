# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
# Standard library imports
import os

# Third-party library imports
from dotenv import load_dotenv
import streamlit as st

# External module imports
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from vector_embeddings import vectorstore

# ----------------------------------------------------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Access the API key and environment
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ["PINECONE_ENVIRONMENT"]
openai_api_key = os.environ.get("OPEN_API_KEY")


# ----------------------------------------------------------------------------------------------------------------------
# GENERATING PROMPT TO CONNECT TO VECTOR STORAGE
# ----------------------------------------------------------------------------------------------------------------------

# create prompt for the chatbot with context from the vectorstore and the user query
def augmented_prompt(user_query: str):
    results = vectorstore.similarity_search(user_query,
                                            k=3)  # perform similarity search and return X most relevant docs
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f'''Using the contexts below, answer the query. If some information is not provided within
    the contexts below, do not include, and if the query cannot be answered with the below information, 
    say "I don't know".

    Contexts:
    {source_knowledge}

    Query: {user_query}'''
    return augmented_prompt


# ----------------------------------------------------------------------------------------------------------------------
# CREATE STREAMLIT UI
# ----------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Chatbot Demo", page_icon="🤖")
st.header("Chatbot Demo 🤖")
st.divider()

# Initialize Langchain chatbot
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("How can I help you today?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user and response message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Initiate LangChain conversation
    conversation = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi AI, how are you today?"),
        AIMessage(content="I'm great thank you. How can I help you?")
    ]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        user_query = user_input
        with st.spinner("Generating answer..."):
            prompt = HumanMessage(content=augmented_prompt(user_query))
            conversation.append(prompt)
            response = chat(conversation)
        message_placeholder.markdown(response.content)

    # Add response message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})