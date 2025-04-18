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
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from Operations import ingest_pdf, build_retriever, ask_question

app = FastAPI()

@app.post("/query-doc")
async def query_doc(pdf: UploadFile = File(...), question: str = Form(...)):
    docs = await ingest_pdf(pdf)
    retriever = build_retriever(docs)
    answer = ask_question(retriever, question)
    return JSONResponse(content={"answer": answer})



