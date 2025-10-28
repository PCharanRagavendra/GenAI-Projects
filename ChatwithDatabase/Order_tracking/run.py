from langchain import LLMMathChain, OpenAI, SQLDatabase, GoogleSearchAPIWrapper, LLMChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from decouple import config
from langchain.llms import CTransformers
from langchain.llms import CTransformers
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
import time
from datetime import timedelta
import json
import sys
import pyodbc
import pandas as pd

os.environ["GOOGLE_CSE_ID"] = "use ur own Custom Search Engine ID"
os.environ["GOOGLE_API_KEY"] = "use ur own Google API key"


llm = CTransformers(model='use ur own model path', 
                    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",   
                    model_type='mistral',
                    temperature=0.8,
                    gpu_layers=0,
                    max_new_tokens = 6000,context_length = 6000)

search = GoogleSearchAPIWrapper()

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

db = SQLDatabase.from_uri("sqlite:///orders1.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

tools = [
    Tool(
        name="SearchTool",
        func=search.run,
        description="Useful for answering questions about current events. Ask targeted questions."
    ),
    Tool(
        name="MathTool",
        func=llm_math_chain.run,
        description="Useful for answering questions about math."
    ),
    Tool(
        name="Product_Database",
        func=db_chain.run,
        description="Useful for answering questions about products."
    )
]

agent = initialize_agent(tools=tools, llm=llm, agent_type="zero-shot-react-description", verbose=True)

st.title('Order Tracking')
st.write("Enter your order ID and a question regarding your order, and I will help you track it.")

question = st.text_input("Enter your question:")
          
def is_valid_order_id(order_id):
    return order_id.isdigit() and len(order_id) == 6

if st.button("Submit"):
        full_question = f"Question: {question}"
        try:
            s="0"
            for i in question.split(" "):
                res = ''.join(filter(lambda i: i.isdigit(), i))
                if len(res)==6:
                     s=res        
            if len(s)==6:
                 
                output = agent.invoke(full_question, handle_parsing_errors=True)
                st.write(f"Answer: {output['output']}")
            else:
                st.write("Please provide your order id.")
                 
            
        except Exception as e:
            st.write(f"An error occurred: {e}")
