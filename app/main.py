from fastapi import FastAPI, HTTPException
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from app.models import ContextRequest
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

llm = OllamaLLM(model="llama3", base_url="http://ollama:11434")

prompt = PromptTemplate.from_template("""
Use the following context to answer the question as clearly and professionally as possible.

Context:
{context}

Question:
{question}
""")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/fullcontext")
def ask_full_context(req: ContextRequest):
    if not req.question or not req.content:
        raise HTTPException(status_code=400, detail="Missing fields.")
    
    response = (prompt | llm).invoke({
        "question": req.question,
        "context": req.content
    })

    return {"response": response}
