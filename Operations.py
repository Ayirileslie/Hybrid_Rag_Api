import os
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 1. Load and split PDF
async def ingest_pdf(file: UploadFile):
    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load PDF content as LangChain Documents
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)  # Clean up file

    return docs

# 2. Embed and build ensemble retriever (BM25 + Chroma)
def build_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

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
    