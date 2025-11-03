from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# ✅ Load environment variable for your OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Basic text generation model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# ✅ Load and prepare local document
with open("sample.txt", "r", encoding="utf-8") as f:
    document = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(document)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ✅ Pydantic models
class AskRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    prompt: str

# ✅ Routes
@app.get("/")
def root():
    return {"message": "AI LangChain Service is running"}

@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        result = qa_chain.run(req.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(req: GenerateRequest):
    try:
        response = llm.invoke(req.prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
