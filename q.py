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

st.title("Quran gpt")

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

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