# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
# Standard library imports
import os
import time

# Third-party library imports
import pandas as pd
from dotenv import load_dotenv
import pinecone
from tqdm import tqdm

# External module imports
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


# ----------------------------------------------------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Access the API key and environment
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ["PINECONE_ENVIRONMENT"]
openai_api_key = os.environ.get("OPENAI_API_KEY")


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


# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA / CREATE VECTOR EMBEDDINGS = BUILDING KNOWLEDGE BASE
# ----------------------------------------------------------------------------------------------------------------------

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# load training data and convert to pd DataFrame
# dataset contains chunked extracts (of ~300 tokens) from papers related to (and including) the Llama 2 research paper
# source: https://huggingface.co/datasets/jamescalam/llama-2-arxiv-papers-chunked
data = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
df = data.to_pandas()

# drop unnecessary columns
df = df.drop(columns=['journal_ref', 'references', 'comment', 'id', 'updated'])
df['published'] = pd.to_datetime(df['published'])
df['published_year'] = df['published'].dt.year

# check out the distribution of papers by year
article_count_by_year = df.groupby('published_year')['doi'].count().reset_index(name='count')
article_count_by_year_sorted = article_count_by_year.sort_values(by='published_year')
# print(article_count_by_year_sorted)

# Check if the index is empty before upserting
index_stats = index.describe_index_stats()
total_vector_count = index_stats['total_vector_count']

if total_vector_count > 0:
    print("Index is not empty. Skipping upsert.")
else:
    # split the dataset into batches and add it to the vector
    batch_size = 100
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i+batch_size, len(df))
        batch = df.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()] # generate unique id
        texts = [x["chunk"] for _, x in batch.iterrows()] # text to embed
        embeds = embed_model.embed_documents(texts)
        metadata = [{ # metadata can't contain arrays; only simple key-value pairs allowed
            "text": x["chunk"],
            "title": x["title"],
            "published_year": x["published_year"],
            "source": x["source"]}
            for _, x in batch.iterrows()]

        index.upsert(vectors=zip(ids, embeds, metadata))

stats = index.describe_index_stats()
print(stats)

text_field = "text"  # metadata field that contains chunk/text
vectorstore = Pinecone(index, embed_model.embed_query, text_field)


