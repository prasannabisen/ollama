from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np

import streamlit as st
import os 
from dotenv import load_dotenv

st.title("Hdfc gpt")

# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]='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'
os.environ["OPENAI_API_KEY"]='sk-proj-RAk5bGLUQFVp25UWdDz8T3BlbkFJSW0zvaG5BfdnDqMGQfaZ'

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
loader = PyPDFLoader("Vehicle Finance.pdf")
# data = loader.load_and_split()
dataLoad = loader.load()

text_spliter = RecursiveCharacterTextSplitter()
text = text_spliter.split_documents(dataLoad)
db = FAISS.from_documents(text, OpenAIEmbeddings())
# db = Chroma(persist_directory="Rupyy", embedding_function=OpenAIEmbeddings())
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