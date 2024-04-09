from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(model_name='gpt-4o', openai_api_key=api_key)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
