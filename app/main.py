from fastapi import FastAPI
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

app = FastAPI()

# Connect to host's Ollama instance (adjust if needed)
ollama = OllamaLLM(
    model="llama3",
    base_url="http://ollama:11434"
)

template = template = PromptTemplate.from_template("{question}")


chain = template | ollama

@app.get("/ask")
def ask_question(question: str):
    response = chain.invoke({"question": question})
    return {"response": response}