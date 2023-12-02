# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
import time
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from datasets import load_dataset


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
# LOADING THE DATA
# ----------------------------------------------------------------------------------------------------------------------

data = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
#print(data[0])

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)


# ----------------------------------------------------------------------------------------------------------------------
# GENERATING CHAT
# ----------------------------------------------------------------------------------------------------------------------

chat = ChatOpenAI(model_name="gpt-3.5-turbo")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]

"""prompt = HumanMessage(content="Why do physicist believe it can produce a unified theory?")
messages.append(prompt)
response = chat(messages)
print("------")
print(response.content)
messages.append(response)"""

"""prompt = HumanMessage(content="Can you tell me about the LLMChain in LangChain? If you don't know the answer, tell me so.")
messages.append(prompt)
response = chat(messages)
print("------")
print(response.content)

print(len(messages))"""


