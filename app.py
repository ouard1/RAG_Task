import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from MessageTemplate import css, bot_template, user_template



FAISS_INDEX_PATH = "faiss_index"
BOOKS_FOLDER = "books" 

def get_pdf_text_with_metadata(folder_path):
    """
    Extract text from all PDF documents in the folder and associate each chunk with metadata including the book title.
    """
    all_text_chunks = []
    all_metadata = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            book_title = os.path.splitext(filename)[0]
            
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)
            all_text_chunks.extend(chunks)
            all_metadata.extend([{"title": book_title}] * len(chunks))
    
    return all_text_chunks, all_metadata


def get_vectorstore_with_metadata(text_chunks, metadata):
    """
    Create a FAISS vector store with metadata by linking text chunks and metadata manually.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    return vectorstore

def save_vectorstore(vectorstore):
    """
    Save the FAISS vector store locally.
    """
    vectorstore.save_local(FAISS_INDEX_PATH)

def load_vectorstore():
    """
    Load the FAISS vector store from local disk if available.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain using the vector store retriever.
    """
    llm = ChatOpenAI(temperature=0.6)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "\n\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    
    return rag_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"input": user_question})
    answer = response['answer']
    context = response['context']
    
    
    st.write(f"**Answer:** {answer}")
    
    
    unique_sources = set()  
    for doc in context:
        source = doc.metadata.get('title', 'No source available')  
        unique_sources.add(source)  
    
   
    st.write("**Sources Used:**")
    for source in unique_sources:
        st.write(f"- Source: {source}")


def main():
    
    load_dotenv()
    
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    if st.session_state.conversation is None:
        with st.spinner("Loading books and preparing the conversation chain..."):
            vectorstore = load_vectorstore()

            if vectorstore is None:
                raw_text_chunks, metadata = get_pdf_text_with_metadata(BOOKS_FOLDER)
                vectorstore = get_vectorstore_with_metadata(raw_text_chunks, metadata)
                save_vectorstore(vectorstore)

            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
