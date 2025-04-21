import os
import tempfile
import shutil
from fastapi import HTTPException
import traceback
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

# Allow your Vercel frontend domain here
origins = [
    "https://query-verse-m1sh.vercel.app",  # Replace with your actual Vercel frontend URL
    "http://localhost",                    # Useful during local testing
    "http://localhost:3000"               # Common for local frontend dev
]


# Cleanup Chroma DB on startup
@app.on_event("startup")
def cleanup_chroma_on_startup():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"🧹 Chroma DB cleaned on startup: {CHROMA_DIR}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],                  # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


@app.post("/upload-doc")
async def upload_doc(pdf: UploadFile = File(...)):
    try:
        # Log file name and size
        contents = await pdf.read()
        size_mb = len(contents) / (1024 * 1024)
        print(f"📄 Received file: {pdf.filename}, size: {size_mb:.2f} MB")
        await pdf.seek(0)  # Reset the file pointer

        docs = await ingest_pdf(pdf)
        retriever = build_retriever(docs)
        save_retriever(retriever)
        return {"message": "✅ PDF processed and retriever saved"}
    except Exception as e:
        print("❌ Upload error:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Upload failed on server. Check logs.")

@app.post("/ask")
async def ask(question: str = Form(...)):
    retriever = load_retriever()
    if retriever is None:
        return JSONResponse(content={"error": "No document uploaded yet."}, status_code=400)
    answer = ask_question(retriever, question)
    return JSONResponse(content=answer)

@app.post("/refresh-backend")
def refresh_backend():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"🔁 Backend reset: Chroma DB cleared at {CHROMA_DIR}")
        return {"message": "✅ Backend refreshed successfully."}
    return {"message": "ℹ️ Chroma DB was already empty."}
