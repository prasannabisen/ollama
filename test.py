from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os 
from dotenv import load_dotenv

st.title("Hdfc gpt")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader("Vehicle Finance.pdf")
# data = loader.load_and_split()
dataLoad = loader.load()

text_spliter = RecursiveCharacterTextSplitter()
text = text_spliter.split_documents(dataLoad)
# db = FAISS.from_documents(text, OpenAIEmbeddings())
db = Chroma(persist_directory="Rupyy", embedding_function=OpenAIEmbeddings())
llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template('''Answer the following question only on context provided and give detailed answer and also references of the answer
                                          <context>
                                          {context}
                                          </context>
                                          Question:{input}''')
document_chain=create_stuff_documents_chain(llm, prompt)
retriver = db.as_retriever()
retrival_chain = create_retrieval_chain(retriver, document_chain)
input_data = st.text_input("Enter input text")
if input_data:
    result = retrival_chain.invoke({"input": input_data})
    st.write("Result:", result["answer"])