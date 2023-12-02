# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
from dotenv import load_dotenv
import time
import re
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from datasets import load_dataset
from tqdm import tqdm


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
# SETUP PINECONE VECTOR DATABASE
# ----------------------------------------------------------------------------------------------------------------------

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

index_name = "llama-2-rag"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name, dimension=1536, metric="cosine"
    )
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name)
#print(index.describe_index_stats())


# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA / CREATE VECTOR EMBEDDINGS
# ----------------------------------------------------------------------------------------------------------------------

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# load training data and convert to pd DataFrame
# dataset contains chunked extracts (of ~300 tokens) from papers related to (and including) the Llama 2 research paper
# source: https://huggingface.co/datasets/jamescalam/llama-2-arxiv-papers-chunked
data = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
df = data.to_pandas()

# drop unnecessary columns
df = df.drop(columns=['journal_ref', 'references', 'comment', 'id', 'updated'])
df['published'] = pd.to_datetime(df['published'])
df['published_year'] = df['published'].dt.year


# clean chunk text and title


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters and punctuation
    text = re.sub(r"\s{2,}", " ", text) # Remove double white space
    text = text.lower()  # Convert to lowercase
    return text


df['chunk'] = df['chunk'].apply(lambda x: clean_text(x))
df['title'] = df['title'].apply(lambda x: clean_text(x))

# check out the distribution of papers by year
article_count_by_year = df.groupby('published_year')['doi'].count().reset_index(name='count')
article_count_by_year_sorted = article_count_by_year.sort_values(by='published_year')
# print(article_count_by_year_sorted)


batch_size = 100

# split the dataset into batches and add it to the vector
for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i+batch_size, len(df))
    batch = df.iloc[i:i_end]
    ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()] # generate unique id
    texts = [x["chunk"] for _, x in batch.iterrows()] # text to embed
    embeds = embed_model.embed_documents(texts)

    # including metadata raised "ApiValueError: Unable to prepare type ndarray for serialization"
    # couldn't find a solution thus far
    """metadata = [
        {"text": x["chunk"],
         "title": x["title"],
         "authors": x["authors"],
         "categories": x["categories"],
         "published_year": x["published_year"],
         "source": x["source"]} for _, x in batch.iterrows()
    ]"""

    index.upsert(vectors=zip(ids, embeds))

stats = index.describe_index_stats()
print(stats)


# ----------------------------------------------------------------------------------------------------------------------
# GENERATING CHAT
# ----------------------------------------------------------------------------------------------------------------------

"""chat = ChatOpenAI(model_name="gpt-3.5-turbo")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]"""

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


