import os
import re
import nltk
import string
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.schema import Document

# Set up your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Use a safe, writable directory instead of tempfile
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)  # Ensure the directory exists

# ✅ Download required NLTK resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

# NLP cleaning setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

import re
def basic_tokenizer(text):
    return re.findall(r'\b\w+\b', text)


def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace('* ', '- ')
    text = text.replace('\n', ' ')
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    words = basic_tokenizer(text)
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Function to save retriever (no-op since Chroma handles persistence)
def save_retriever(retriever):
    pass

# Load retriever from persisted Chroma
def load_retriever():
    if not os.path.exists(CHROMA_DIR):
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever],
        weights=[1.0]
    )
    return ensemble

# Ingest uploaded PDF
async def ingest_pdf(file: UploadFile):
    import tempfile
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

# Build vectorstore and retriever
def build_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 3

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble

# Ask a question using Gemini and retrieval
def ask_question(retriever, question: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain(question)
    plain_result = clean_markdown(result["result"])
    return {"query": question, "answer": plain_result}
