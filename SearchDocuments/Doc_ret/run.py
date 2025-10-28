from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import LLMMathChain, OpenAI, SQLDatabase, GoogleSearchAPIWrapper, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from decouple import config
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser

#loading documents
directory = 'use your own path to documents'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

#split the documents atter we load them
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
#print(len(docs))

#embedd text using langchain
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

openai_api_key = 'Use ur own key'

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, api_key=openai_api_key)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

st.title('Ask anything from document')

question = st.text_input("Enter your question:")
          
if st.button("Submit"):
        full_question = f"Question: {question}"
        try:                 
            answer1=retrieval_chain.run(question)
            st.write(f"Answer: {answer1}")

        except Exception as e:
            st.write(f"An error occurred: {e}")






