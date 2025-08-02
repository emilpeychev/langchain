# Dockerfile for a LangChain

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade langchain
# Install curl and create user
RUN apt-get update && \
    apt-get install -y curl && \
    groupadd -r appuser && useradd -r -g appuser appuser && \
    rm -rf /var/lib/apt/lists/*
COPY --chown=appuser:appuser app/ ./app
USER appuser
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081", "--reload", "--log-level", "info"]