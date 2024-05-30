from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

prompt = ChatPromptTemplate.from_messages([
    ("system","you are helpful assiestent, please help"),
    ("user", "Question :{question}")
])

st.title("Langchain Demo With Laama api")
input_text=st.text_input("Search the topic you want to search")

# print("+++++", aviary.get_models())
llm = Ollama(model="llama3")
outputParser = StrOutputParser()
chain=prompt|llm|outputParser

if input_text:
    st.write(chain.invoke({"question":input_text}))
