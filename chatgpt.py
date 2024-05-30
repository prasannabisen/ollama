from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np

import streamlit as st
import os 
from dotenv import load_dotenv

st.title("Hdfc chat gpt llm")

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

db = Chroma(persist_directory="Rupyy", embedding_function=OpenAIEmbeddings())
llm = OpenAI(model_name="gpt-3.5-turbo")

# from langchain_community.llms import OpenAI
# openai = OpenAI(model_name="gpt-3.5-turbo-instruct")


# prompt = ChatPromptTemplate.from_messages([])
prompt = ChatPromptTemplate.from_template('''Answer the following question only on context provided <context>{context}</context> Question:{input}''')
document_chain=create_stuff_documents_chain(llm, prompt)
retriver = db.as_retriever()
retrival_chain = create_retrieval_chain(retriver, document_chain)
input_data = st.text_input("Enter input text")
if input_data:
    result = retrival_chain.invoke({"input": input_data})
    st.write("Result:", result["answer"])