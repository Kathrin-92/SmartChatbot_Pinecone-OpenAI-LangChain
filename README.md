# Creating a ChatBot that learns from Contextual Knowledge 
Building a Chatbot with LangChain, OpenAI, and the Pinecone vector database, using streamlit to build a simple UI.

<img width="669" alt="Bildschirm­foto 2023-12-04 um 20 47 23" src="https://github.com/Kathrin-92/SmartChatbot_Pinecone-OpenAI-LangChain/assets/71875232/2e82c2ce-70db-4309-be78-87ec73e052af">

## General Info

**Project Overview** 

This chatbot project leverages technologies including LangChain, OpenAI, and the Pinecone vector database. The primary objective of this project is to create a simple chatbot that is capable of learning from provided data and adept at answering previously unknown facts. 
This project is based on the DataCamp Codealong: [Building Chatbots with the OpenAI API and Pinecone](https://www.datacamp.com/code-along/building-chatbots-openai-api-pinecone)

The training data is extracted from arxiv research papers and is sourced from the Hugging Face datasets repository (source: https://huggingface.co/datasets/jamescalam/llama-2-arxiv-papers-chunked). The dataset contains chunked extracts of approximately 300 tokens each. This means that the chatbot can provide LLama-specific knowledge.

Pinecone is utilized as the vector database to efficiently store and retrieve the information. With the help of the LangChain framework, the OpenAI model is used as the base for the chatbot. 
A simple and user-friendly interface is built using Streamlit, allowing users to interact seamlessly with the chatbot. 

**Key Skills Learned**

* Gained insights into natural language processing techniques using LangChain and OpenAI's GPT model.
* Learned to integrate contextual knowledge into the gpt-model. 
* Acquired basic skills in leveraging Pinecone as a vector database for streamlined storage and retrieval of information.


## Installation

**Requirements:** 
An OpenAI API key and Pinecone API key are necessary to run the project yourself. 
