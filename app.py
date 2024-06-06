import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain #to capture context among docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.retrieval import create_retrieval_chain
import os
import time
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

st.set_page_config(page_title='Q&A demo')

st.title('Gemma Q&A bot')

llm = ChatGroq(groq_api_key=groq_api_key, model='gemma-7b-it')

prompt = ChatPromptTemplate.from_template(
    """
    
    Answer the question based on the context.
    Please provide accurate answers based on the question
    <context>
    {context}
    <context>
    Questions : {input}



    """
)

def vector_embedding():

    if 'vectors' not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        st.session_state.loader = PyPDFDirectoryLoader('./us_census')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

prompt1 = st.text_input('What is your question related to docs')
    
if st.button('Create vector database'):
    vector_embedding()
    st.write('Vector database is ready')



if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")



