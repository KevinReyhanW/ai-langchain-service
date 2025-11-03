from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
import logging

# -----------------------------
# App & Environment Setup
# -----------------------------
app = FastAPI(title="AI LangChain Service")

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("Loaded OPENAI_API_KEY prefix:", OPENAI_API_KEY[:20])

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY — please add it to your environment or .env file")

# -----------------------------
# CORS (allow your frontend)
# -----------------------------
origins = [
    "http://localhost:3000",  # local Next.js frontend
    "https://ai-langchain-frontend.vercel.app",  # deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize LLM
# -----------------------------
try:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
except Exception as e:
    llm = None
    logging.warning("⚠️ Could not initialize ChatOpenAI: %s", e)

# -----------------------------
# Load document and split chunks
# -----------------------------
try:
    with open("sample.txt", "r", encoding="utf-8") as f:
        document = f.read()
except FileNotFoundError:
    document = "This is a placeholder document since sample.txt was not found."

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

chunks = split_text_into_chunks(document)

# -----------------------------
# Vectorstore (FAISS) Setup
# -----------------------------
vectorstore = None
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
except Exception as e:
    logging.warning("⚠️ Could not create FAISS vectorstore: %s", e)
    vectorstore = None

# -----------------------------
# Local fallback retriever
# -----------------------------
def _normalize(text: str):
    return [t for t in ''.join(c.lower() if c.isalnum() else ' ' for c in text).split() if t]

def get_top_chunks(question: str, k: int = 3):
    q_tokens = _normalize(question)
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    scored = []
    for chunk in chunks:
        c_tokens = _normalize(chunk)
        score = len(q_set.intersection(c_tokens))
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:k] if s > 0]

# -----------------------------
# Request Models
# -----------------------------
class AskRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    prompt: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "✅ AI LangChain Service is running. Use POST /ask or POST /generate."}

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        # Try FAISS retrieval first
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(req.question, k=3)
                chunks_out = [getattr(d, "page_content", str(d)) for d in docs if d]
                if chunks_out:
                    return {"method": "vectorstore", "chunks": chunks_out}
            except Exception as e:
                logging.warning("⚠️ FAISS similarity search failed: %s", e)

        # Fallback local search
        top = get_top_chunks(req.question)
        if not top:
            return {"method": "fallback", "answer": "No relevant context found."}
        return {"method": "fallback", "chunks": top}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not initialized (check API key or quota).")
    try:
        response = llm.invoke(req.prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
