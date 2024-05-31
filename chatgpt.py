import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

st.title("HDFC GPT")

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]='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'
os.environ["OPENAI_API_KEY"]='sk-proj-oANbdKKOM8iwrg3FT4AjT3BlbkFJtTuGMqCZv4CyklnvmbjY'

# Load and split PDF document
loader = PyPDFLoader("Vehicle Finance.pdf")
data_load = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
texts = text_splitter.split_documents(data_load)

# Create FAISS vector store from documents
db = FAISS.from_documents(texts, OpenAIEmbeddings())

# Use OpenAI's ChatGPT Turbo model
llm = OpenAI(model="gpt-3.5-turbo")

# Create prompt template
prompt = ChatPromptTemplate.from_template('''
    Answer the following question only on context provided and give detailed answer and also references of the answer
    <context>
    {context}
    </context>
    Question:{input}
''')

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit input and output
input_data = st.text_input("Enter input text")
if input_data:
    try:
        result = retrieval_chain.run({"input": input_data})
        st.write("Result:", result["output"])
    except Exception as e:
        st.error(f"An error occurred: {e}")
