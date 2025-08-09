[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)


# QueryVerse Backend API 📚🤖

A powerful AI-powered document question-answering API that allows you to upload PDF documents and ask intelligent questions about their content. Built with FastAPI and powered by Google's Gemini AI with advanced retrieval techniques.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF documents for intelligent querying
- **AI-Powered Q&A**: Ask natural language questions about your documents using Google Gemini
- **Hybrid Search**: Combines dense vector search (Chroma) and sparse search (BM25) for optimal retrieval
- **Real-time Processing**: Fast document ingestion and question answering
- **RESTful API**: Clean, well-documented endpoints
- **CORS Enabled**: Ready for frontend integration

## 🛠️ Tech Stack

- **FastAPI**: High-performance Python web framework
- **Google Gemini AI**: Advanced language model for question answering
- **LangChain**: Framework for building AI applications
- **Chroma DB**: Vector database for document embeddings
- **BM25**: Sparse retrieval for keyword-based search
- **NLTK**: Natural language processing toolkit
- **PyPDF**: PDF document processing

## 🌐 Frontend Interface

Access the frontend interface at: **[https://query-verse-m1sh.vercel.app](https://query-verse-m1sh.vercel.app)**

## 📹 Video Documentation

**API Tutorial**: [Backend Setup Guide](https://your-tutorial-link-here.com) - Step-by-step setup and usage guide

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini AI)

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd queryverse-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn python-multipart
   pip install langchain langchain-community langchain-chroma
   pip install langchain-google-genai
   pip install nltk pypdf chromadb
   ```

4. **Set environment variables**
   ```bash
   export GOOGLE_API_KEY="your-google-api-key-here"
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### POST `/upload-doc`
Upload a PDF document for processing.

**Content-Type**: `multipart/form-data`

**Parameters:**
- `pdf`: PDF file

**Response:**
```json
{
  "message": "✅ PDF processed and retriever saved"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/upload-doc" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "pdf=@document.pdf"
```

### POST `/ask`
Ask a question about the uploaded document.

**Content-Type**: `application/x-www-form-urlencoded`

**Parameters:**
- `question`: Question string

**Response:**
```json
{
  "query": "What is the main topic?",
  "answer": "The main topic of the document is..."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What is the main topic of the document?"
```

### POST `/refresh-backend`
Clear the document database and reset the backend.

**Response:**
```json
{
  "message": "✅ Backend refreshed successfully."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/refresh-backend"
```

## 🔍 How It Works

1. **Document Upload**: PDF files are processed using PyPDFLoader and split into manageable chunks
2. **Text Cleaning**: Advanced NLP preprocessing including lemmatization and stopword removal
3. **Embedding Generation**: Text chunks are converted to vector embeddings using Google's embedding model
4. **Hybrid Retrieval**: Combines semantic search (Chroma) with keyword search (BM25) for comprehensive retrieval
5. **Question Answering**: Uses Google's Gemini 2.0 Flash model to generate accurate answers
6. **Response Processing**: Markdown formatting is cleaned for optimal presentation

## 🏗️ Architecture

```
PDF Upload
    ↓
Text Extraction & Cleaning
    ↓
Chunk Creation
    ↓
┌─────────────────┐    ┌─────────────────┐
│   Chroma DB     │    │   BM25 Index    │
│ (Vector Store)  │    │ (Sparse Search) │
└─────────────────┘    └─────────────────┘
    ↓                       ↓
    └───────────┬───────────┘
                ↓
      Ensemble Retriever
                ↓
        Google Gemini AI
                ↓
          Final Answer
```

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── Operations.py           # Core document processing logic
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔐 Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🚀 Deployment

### Recommended Platforms

- **Railway**: `railway up` - Easy Python deployment
- **Heroku**: Classic PaaS platform with Procfile
- **DigitalOcean App Platform**: Simple cloud deployment
- **AWS Lambda**: Serverless option with Mangum adapter
- **Google Cloud Run**: Container-based deployment

### Docker Deployment

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 API Documentation

Once running, access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Configuration

### CORS Settings
The API is pre-configured to work with the frontend at:
- `https://query-verse-m1sh.vercel.app`
- `http://localhost:3000` (for local development)

To modify CORS settings, update the `origins` list in `main.py`.

### Storage
- Documents are temporarily stored in the system's temp directory
- Chroma DB persists in a temporary directory that's cleaned on startup
- For production, consider using persistent storage

## 🧪 Testing

Test the API endpoints using the provided curl examples or tools like Postman.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Google Gemini AI for powerful language understanding
- LangChain for the excellent framework
- Chroma DB for efficient vector storage

---

**QueryVerse Backend API - Making documents searchable and intelligent** 🚀
