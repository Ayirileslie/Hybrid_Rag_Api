import os
import tempfile
import pickle
import tempfile
import re
import string
import nltk
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.schema import Document

os.environ["GOOGLE_API_KEY"] = "AIzaSyDcxLifISEcpyiW6KJPJzasZlzLJv81OrQ"

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits (optional depending on your case)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords and apply lemmatization
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]# Reconstruct the text from cleaned words
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def save_retriever(retriever):
    # No longer needed – Chroma handles persistence
    pass

def load_retriever():
    if not os.path.exists(CHROMA_DIR):
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # optional: load BM25 again if needed
    # if not, just use dense_retriever
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever],
        weights=[1.0]
    )
    return ensemble



async def ingest_pdf(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    cleaned_docs = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        ) for doc in docs
    ]

    os.remove(tmp_path)
    return cleaned_docs

# 2. Embed and build ensemble retriever (BM25 + Chroma)
def build_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings,persist_directory=CHROMA_DIR)
    vectorstore.persist()

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 3

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble

# 3. Create QA chain with Gemini
def ask_question(retriever, question: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain(question)
    return {"answer": result}
    

