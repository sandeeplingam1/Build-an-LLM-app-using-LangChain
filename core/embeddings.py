from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_vector_store(documents, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(documents, embeddings)
