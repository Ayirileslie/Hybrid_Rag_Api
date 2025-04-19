import os
import tempfile
import shutil
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
from fastapi.middleware.cors import CORSMiddleware
from Operations import ingest_pdf , build_retriever, ask_question, save_retriever, load_retriever
from Operations import CHROMA_DIR 

app = FastAPI()
# Chroma DB location


# Cleanup Chroma DB on startup
@app.on_event("startup")
def cleanup_chroma_on_startup():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"🧹 Chroma DB cleaned on startup: {CHROMA_DIR}")


# Allow your Vercel frontend domain here
origins = [
    "https://your-vercel-app.vercel.app",  # Replace with your actual Vercel frontend URL
    "http://localhost",                    # Useful during local testing
    "http://localhost:3000"               # Common for local frontend dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],                  # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


@app.post("/upload-doc")
async def upload_doc(pdf: UploadFile = File(...)):
    docs = await ingest_pdf(pdf)
    retriever = build_retriever(docs)
    save_retriever(retriever)
    return {"message": "PDF processed and retriever saved to temp file"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    retriever = load_retriever()
    if retriever is None:
        return JSONResponse(content={"error": "No document uploaded yet."}, status_code=400)
    answer = ask_question(retriever, question)
    return JSONResponse(content=answer)