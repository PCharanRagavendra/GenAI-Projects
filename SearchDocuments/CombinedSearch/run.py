import os
from langchain import LLMMathChain, OpenAI, SQLDatabase, GoogleSearchAPIWrapper, LLMChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import streamlit as st


openai_api_key = 'Use ur own key'
os.environ['OPENAI_API_KEY'] = 'Use ur own key'
os.environ["GOOGLE_CSE_ID"] = "Use ur own key"
os.environ["GOOGLE_API_KEY"] = "Use ur own key"

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, api_key=openai_api_key)

db = SQLDatabase.from_uri("sqlite:///orders1.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
dbtool =Tool(name="SearchTool", func=db_chain.run, description="Useful for answering questions about current events. Ask targeted questions.")

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

loader = DirectoryLoader('use your own path to pdfs')
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()
retriever_tool=create_retriever_tool(retriever,"pdf_search", "Search for information in pdf. For any questions, you must use this tool!")

tools=[wiki, dbtool, retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

agent=create_openai_tools_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title('Order Tracking')

question = st.text_input("Enter your question:")

if st.button("Submit"):
        full_question = f"Question: {question}"
        try:
            output = agent_executor.invoke({"input": full_question})
            st.write(f"Answer: {output['output']}")

        except Exception as e:
            st.write(f"An ercdror occurred: {e}")
