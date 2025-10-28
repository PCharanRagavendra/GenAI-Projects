import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.prompts import PromptTemplate

# Database credentials
dc = {
    'host': 'use ur own hostname',
    'user': 'use ur own username',
    'password': 'Use ur own password',
    'database': 'Use ur own database'
}

openai_api_key = 'use ur own OpenAI API key'

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, api_key=openai_api_key)

db = SQLDatabase.from_uri(f"mysql+pymysql://{dc['user']}:{dc['password']}@{dc['host']}/{dc['database']}")

generate_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)

st.title('EV Charging Stations Natural Language to SQL Query Translator')
st.write("Enter a natural language query, and I'll translate it into an SQL query and execute it on the 'stations' table in the 'charge' database.")

question = st.text_input("Enter your question:")

if st.button("Submit"):
    output = chain.invoke({"question": question})
    
    with open("questions.txt", "a") as file:
        file.write(f"Question: {question}\nAnswer: {output}\n\n")

    st.write(f"Answer: {output}")
