import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from MessageTemplate import css , user_template,bot_template
import re



FAISS_INDEX_PATH = "faiss_index"
BOOKS_FOLDER = "books" 


def clean_text(text):
    """
    Cleans the input text by removing tabs, newlines, and extra spaces.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned version of the text.
    """
    cleaned_text = text.replace("\t", " ").replace("\n", " ").strip()
    return cleaned_text



def get_pdf_text_with_metadata(folder_path):
    """
    Extracts text from all PDF documents in the specified folder and associates 
    each chunk with metadata including the book title.

    Args:
        folder_path (str): Path to the folder containing the PDF documents.

    Returns:
        tuple: A tuple containing a list of text chunks and corresponding metadata.
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
            

            
            book_title = os.path.splitext(filename)[0].replace("-", " ").title()
            cleaned_text = clean_text(text)
            text_splitter = RecursiveCharacterTextSplitter(
                  chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(cleaned_text)
            all_text_chunks.extend(chunks)
            all_metadata.extend([{"title": book_title}] * len(chunks))
    
    return all_text_chunks, all_metadata

def get_vectorstore_with_metadata(text_chunks, metadata):
    """
    Creates a FAISS vector store by embedding the provided text chunks and associating them with metadata.

    Args:
        text_chunks (list): A list of text chunks to be embedded.
        metadata (list): A list of metadata corresponding to the text chunks.

    Returns:
        FAISS: A FAISS vector store containing the embedded text chunks.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    return vectorstore

def save_vectorstore(vectorstore):
    """
    Saves the FAISS vector store to the local disk for later use.

    Args:
        vectorstore (FAISS): The FAISS vector store to be saved.
    """
    vectorstore.save_local(FAISS_INDEX_PATH)

def load_vectorstore():
    """
    Loads the FAISS vector store from the local disk if it exists.

    Returns:
        FAISS or None: The loaded FAISS vector store, or None if it doesn't exist.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def filter_passages_with_llm(query, passages, metadata, threshold=0.98):
    """
    Filters passages based on their relevance to the given query using an LLM model.
    Passages below the specified threshold relevance score are discarded.

    Args:
        query (str): The user query to be matched against the passages.
        passages (list): A list of text passages to be scored.
        metadata (list): A list of metadata corresponding to the passages.
        threshold (float): The relevance threshold for keeping passages (default is 0.98).

    Returns:
        list: A list of relevant passages that passed the threshold.
    """
    llm = ChatOpenAI(temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([ 
        ("system", "Score the relevance of the following passage to the query:"), 
        ("user", "Query: {query}\nPassage: {passage}\nReturn a score from 0 to 1, where 0 is completely irrelevant and 1 is fully relevant.")
    ])
    
    relevant_passages = []

    
    for idx, passage in enumerate(passages):
       
        messages = [
            ChatMessage(role="system", content="Score the relevance of the passage to the query."),
            ChatMessage(role="user", content=f"Query: {query}\nPassage: {passage}")
        ]
        
        
        
        response = llm.invoke(messages)
        

       
        response_content = response.content.strip()

       
        match = re.search(r'(\d+)(?:/|\s*)?(\d*)', response_content)
        if match:
            numerator = int(match.group(1))  
            denominator = int(match.group(2)) if match.group(2) else 5  
            relevance_score = numerator / denominator  
        else:
            relevance_score = 0.0  

        
        if relevance_score >= threshold:
            relevant_passages.append({
                "passage": passage, 
                "metadata": metadata[idx],  
                "score": relevance_score,
                "index": idx
            })
    
    return relevant_passages






def get_conversation_chain(vectorstore):
    """
    Creates a conversational retrieval chain using the provided vector store for document retrieval.

    Args:
        vectorstore (FAISS): The FAISS vector store used for retrieval.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain.
    """
    llm = ChatOpenAI(temperature=0.5)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        
        "the question If a question pertains to a specific book, prioritize documents from that book.."
         " If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
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
    """
    Handles user input by processing the question and returning a response based on the retrieved documents.
    
    Args:
        user_question (str): The question input by the user.
    """
    vectorstore = st.session_state.get("vectorstore")  
    if vectorstore:
        retrieved_passages = vectorstore.similarity_search(user_question, k=10)
        
        passages_with_metadata = [
            {"passage": doc.page_content, "metadata": doc.metadata} for doc in retrieved_passages
        ]
        
        filtered_passages = filter_passages_with_llm(
            user_question,
            [doc["passage"] for doc in passages_with_metadata],
            [doc["metadata"] for doc in passages_with_metadata]
        )
        
        

        
        relevant_documents = []
        for doc in filtered_passages:
            relevant_documents.append({
                "passage": doc["passage"],
                "metadata": passages_with_metadata[filtered_passages.index(doc)]["metadata"]
            })
        
        if relevant_documents:
            context = "\n".join([f"Passage: {doc['passage']}\nMetadata: {doc['metadata']}" for doc in relevant_documents])
        else:
            context = "\n".join([f"Passage: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_passages])
        
        conversation_chain = st.session_state.get("conversation")
        if conversation_chain:
            response = conversation_chain.invoke({"input": user_question, "context": context})
            answer = response['answer']
            
            unique_sources = set()
            if relevant_documents:
                for doc in relevant_documents:
                    source = doc["metadata"].get('title', 'No source available')
                    unique_sources.add(source)
            else:
                for doc in retrieved_passages:
                    source = doc.metadata.get('title', 'No source available')
                    unique_sources.add(source)

            sources_text = "\n\n**Sources Used:**\n" + "\n".join([f"- {source}" for source in unique_sources])
            answer_with_sources = f"{answer}{sources_text}"

            st.session_state.chat_history.append({"sender": "user", "message": user_question})
            st.session_state.chat_history.append({"sender": "bot", "message": answer_with_sources})
        else:
            st.write("Error: Conversation chain is not available.")



def render_chat():
    """
    Render the chat history using the bot and user templates.
    """
    for chat in reversed(st.session_state.chat_history):
        if chat["sender"] == "user":
            st.markdown(user_template.replace("{{MSG}}", chat["message"]), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", chat["message"]), unsafe_allow_html=True)



def main():

    """
    Main function to initialize and run the Streamlit app for chat-based question answering.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
        render_chat()

    if st.session_state.conversation is None:
        with st.spinner("Loading books and preparing the conversation chain..."):
            vectorstore = load_vectorstore()

            if vectorstore is None:
                raw_text_chunks, metadata = get_pdf_text_with_metadata(BOOKS_FOLDER)
                vectorstore = get_vectorstore_with_metadata(raw_text_chunks, metadata)
                save_vectorstore(vectorstore)

            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.vectorstore = vectorstore 

if __name__ == '__main__':
    main()