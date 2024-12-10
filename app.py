import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter



from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from MessageTemplate import css, bot_template, user_template

# Path to save the FAISS index locally
FAISS_INDEX_PATH = "faiss_index"

def get_pdf_text_with_metadata(pdf_docs):
    """
    Extract text from PDF documents and associate each chunk with metadata including the book title.
    """
    all_text_chunks = []
    all_metadata = []
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Extract file name as the book title
        book_title = os.path.splitext(os.path.basename(pdf.name))[0]
        
        # Split text into chunks and assign metadata
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        all_text_chunks.extend(chunks)
        all_metadata.extend([{"title": book_title}] * len(chunks))
    
    return all_text_chunks, all_metadata

def get_vectorstore_with_metadata(text_chunks, metadata):
    """
    Create a FAISS vector store with metadata by linking text chunks and metadata manually.
    """
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    
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
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        return None

def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain using the vector store retriever.
    """
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # Check if FAISS index exists, load it. Otherwise, process the PDFs and create the vectorstore
                vectorstore = load_vectorstore()

                if vectorstore is None:
                    # Get the raw text and metadata from PDFs
                    raw_text_chunks, metadata = get_pdf_text_with_metadata(pdf_docs)
                    # Create the vector store
                    vectorstore = get_vectorstore_with_metadata(raw_text_chunks, metadata)
                    # Save the vectorstore for future use
                    save_vectorstore(vectorstore)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
