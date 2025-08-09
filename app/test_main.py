import pytest
from fastapi.testclient import TestClient
from app.main import app, llm

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_fullcontext_success(monkeypatch):
    # Accept any extra arguments
    monkeypatch.setattr(type(llm), "invoke", lambda self, inputs, *args, **kwargs: "Mocked answer")

    payload = {
        "question": "What is LangChain?",
        "content": "LangChain is a framework for developing applications powered by language models."
    }
    response = client.post("/fullcontext", json=payload)
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked answer"}

def test_fullcontext_missing_fields():
    payload = {
        "question": "",
        "content": ""
    }
    response = client.post("/fullcontext", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing fields."

