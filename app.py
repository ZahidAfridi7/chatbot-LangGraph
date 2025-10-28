import os
import subprocess
import threading
import asyncio
import time
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from prompt import SYSTEM_PROMPT  # ✅ Import your company prompt

# ========= CONFIG =========
load_dotenv()
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ========= FastAPI Setup =========
app = FastAPI(title="Dexterz Technologies Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Initialize RAG Components =========
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ Use GPT-4.1-nano for faster response
llm = ChatOpenAI(model="gpt-4.1-nano", streaming=True, temperature=0.2)

# ✅ Combine SYSTEM_PROMPT with dynamic question/context
QA_TEMPLATE = """
{system_prompt}

Context:
{context}

Question:
{question}
"""

qa_prompt = PromptTemplate(
    input_variables=["system_prompt", "context", "question"],
    template=QA_TEMPLATE,
)

# ✅ Pre-bind your SYSTEM_PROMPT into the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": qa_prompt.partial(system_prompt=SYSTEM_PROMPT)
    },
)

# ========= Async Document Upload =========
@app.post("/upload/")
async def upload_file(file: UploadFile):
    """Upload and embed a document (PDF, DOCX, CSV)."""
    try:
        path = os.path.join(DATA_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.filename.endswith(".csv"):
            loader = CSVLoader(path)
        else:
            return {"error": "Unsupported file type"}

        docs = await asyncio.to_thread(loader.load)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = await asyncio.to_thread(splitter.split_documents, docs)
        await asyncio.to_thread(db.add_documents, chunks)

        return {"status": "File embedded successfully!"}
    except Exception as e:
        return {"error": str(e)}

# ========= Async Streaming Endpoint =========
@app.post("/query/")
async def query(question: str = Form(...), thread_id: str = Form("1")):
    print("\n🟢 New Query Received")

    async def event_stream():
        t0 = time.time()
        print("🤖 Thinking...")

        # Phase 1: Retrieval
        retrieved_docs = await asyncio.to_thread(retriever.invoke, question)
        t1 = time.time()
        print(f"📚 Retrieval time: {t1 - t0:.2f}s | {len(retrieved_docs)} docs")

        # Phase 2: Stream generation
        stream_start = time.time()
        async for chunk in qa_chain.astream({"query": question}):
            text_piece = chunk.get("result", "") if isinstance(chunk, dict) else str(chunk)
            if text_piece.strip():
                yield text_piece
            await asyncio.sleep(0)
        print(f"✅ Streamed in {time.time() - stream_start:.2f}s")
        print(f"⏱️ Total time: {time.time() - t0:.2f}s")

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")

# ========= Streamlit Launcher =========
def run_streamlit():
    subprocess.run(["streamlit", "run", "frontend.py", "--server.port", "8501"])

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=run_streamlit, daemon=True).start()
    print("🚀 Streamlit running on http://localhost:8501")

@app.get("/")
async def root():
    return {"message": "Dexterz Technologies Chatbot is running!"}
