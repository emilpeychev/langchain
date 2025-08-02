from fastapi import FastAPI
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os

app = FastAPI()

# --------------------------
# Setup
# --------------------------

# Connect to local Ollama (adjust if not docker)
llm = OllamaLLM(model="llama3", base_url="http://ollama:11434")
embedding = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")

# Use PromptTemplate to combine question + context
prompt = PromptTemplate.from_template("""
Use the following context to answer the question as clearly and professionally as possible.

Context:
{context}

Question:
{question}
""")

# Store FAISS in memory at startup
vectorstore = None

@app.on_event("startup")
def load_docs():
    global vectorstore

    # Read all .py, .sh, .yaml files from current directory recursively
    extensions = (".py", ".sh", ".yaml", ".yml")
    docs = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(extensions):
                path = os.path.join(root, file)
                try:
                    content = open(path).read()
                    docs.append(Document(page_content=content, metadata={"path": path}))
                except Exception as e:
                    print(f"Failed to read {path}: {e}")

    # Split into ~1000 character chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create in-memory FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding)

# --------------------------
# Query Endpoint
# --------------------------

@app.get("/ask")
def ask_question(question: str):
    global vectorstore

    if not vectorstore:
        return {"response": "Vector index not loaded"}

    # Find relevant chunks
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Invoke LLM
    final_chain = prompt | llm
    response = final_chain.invoke({"question": question, "context": context})

    return {"response": response}