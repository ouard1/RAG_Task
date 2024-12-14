

```markdown
# Chat with Multiple PDFs

This project allows you to interact with multiple PDF documents through a conversational AI. Using a combination of **LangChain**, **OpenAI embeddings**, and **FAISS vector stores**, users can query documents and retrieve relevant passages. The system utilizes a conversational model to answer questions based on the contents of these documents.

## Features

- **PDF Text Extraction**: Extracts and processes text from multiple PDF documents.
- **Text Chunking and Embedding**: Breaks down text into chunks and embeds them into vector space using OpenAI embeddings.
- **FAISS Vector Store**: Utilizes FAISS to store the text chunks with their embeddings, enabling efficient similarity search.
- **Conversational Retrieval Chain**: Retrieves relevant passages and constructs concise answers based on them.
- **Automatic Document Loading**: Loads documents automatically from the specified folder for query processing.
- **Chat Interface**: An interactive chat interface built with Streamlit where users can ask questions about the content of the documents.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ouard1/RAG_Task.git
 
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file:

   ```ini
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Place your PDF documents in the `books` folder.
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open your browser and interact with the application by entering questions about the documents.

## Code Overview

- **Text Extraction**: The `get_pdf_text_with_metadata` function extracts text from PDFs, cleans it, and splits it into chunks.
- **FAISS Vector Store**: The `get_vectorstore_with_metadata` function creates a vector store from the extracted text, embedding the chunks.
- **Conversational Retrieval**: The `get_conversation_chain` function creates a retrieval chain that allows for conversational question answering by retrieving the most relevant documents from the vector store.
- **User Interaction**: The `handle_userinput` function handles user input, retrieves relevant documents, and passes them to the conversational chain to generate an answer.
- **Web Interface**: Streamlit is used to render the chat interface, allowing users to ask questions and view answers.

## Future Enhancements

### Adaptive Retrieval Augmentation
 to implement an adaptive retrieval mechanism that improves the system's ability to answer questions by dynamically adjusting to different types of queries. Currently, the system retrieves relevant documents from stored PDFs using a FAISS vector store, but in the future, if a relevant response is not found, the system will automatically query external sources (e.g., the web or a specific API) to augment the response. This adaptive retrieval system will allow the model to fetch live data, enhancing its ability to handle a broader range of questions, especially those requiring up-to-date information or topics not covered in the stored documents.

This functionality will rely on an intelligent fallback mechanism that determines when to search the web or other external sources, ensuring the AI remains useful even when document coverage is incomplete.

### Automatic Database Updates with New Documents
 plan to introduce a feature that automatically updates the document database when new PDFs are added to the system. The vector store will be re-indexed, and new documents will be processed and embedded, ensuring that the knowledge base remains current without requiring manual intervention. This will involve:
- Automatic text extraction from newly uploaded PDFs.
- Re-running the vector store creation and embedding steps.
- Updating the FAISS index with new content, which will be seamlessly integrated into the existing retrieval pipeline.

This will ensure that the system continuously adapts to new information and provides more accurate and relevant answers as the database grows.
### Reducing Latency
## Sources

- [Optimizing RAG for Production: Leveraging Filtering and Reranking for Better Performance](https://bhavyabarri.medium.com/optimizing-rag-for-production-leveraging-filtering-and-reranking-for-better-performance-da8e16a20453)
- [A Guide on 12 Tuning Strategies for Production-Ready RAG Applications](https://towardsdatascience.com/a-guide-on-12-tuning-strategies-for-production-ready-rag-applications-7ca646833439)
- [Arxiv Paper: Towards Better RAG](https://arxiv.org/html/2406.18740v1#S3)
- [LangChain Official Website](https://www.langchain.com/)
- GitHub Repository: [RAG Task](git@github.com:ouard1/RAG_Task.git)

# Example of Use


The app will provide answers using the extracted content from the PDFs. This is how the interaction looks:

![image](https://github.com/user-attachments/assets/ac2668ab-d3a5-46ca-862e-99cf70a33873)
![image](https://github.com/user-attachments/assets/65fc0728-55c3-43d0-9b38-9cc9d975141b)




