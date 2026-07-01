# SHL Assessment Recommendation Agent

## Overview

The SHL Assessment Recommendation Agent is a conversational AI system that helps hiring managers identify suitable SHL Individual Test Solutions based on job roles, required skills, and hiring requirements.

The application is built using FastAPI and follows the assignment specification by exposing the required REST API endpoints:

- GET `/health`
- POST `/chat`

The agent supports multi-turn conversations, asks clarification questions when necessary, recommends relevant assessments from the SHL catalog, compares assessments, and refuses requests that are outside its intended scope.

---

# Features

- Multi-turn conversational assessment recommendation
- Clarification questions for incomplete hiring requirements
- Recommendation of relevant SHL Individual Test Solutions
- Assessment comparison
- Scope-aware refusal for unrelated requests
- REST API using FastAPI
- Swagger API documentation
- Stateless conversation handling

---

# Project Structure

```
app/
│── main.py              # FastAPI application
│── dialogue.py          # Conversation orchestration
│── retrieval.py         # Assessment retrieval
│── catalog.py           # Catalog loading
│── llm.py               # LLM integration
│── schemas.py           # Request/Response models

data/
│── catalog.json         # SHL assessment catalog

scripts/
│── scrape_catalog.py    # Catalog scraper

tests/
│── test_traces.py       # Evaluation script

Dockerfile
requirements.txt
README.md
.env.example
```

---

# System Workflow

```
User Request
      │
      ▼
FastAPI (/chat)
      │
      ▼
Dialogue Manager
      │
      ├── Clarification
      ├── Recommendation
      ├── Comparison
      └── Refusal
      │
      ▼
LLM Controller
      │
      ▼
Retrieval Engine
      │
      ▼
SHL Assessment Catalog
      │
      ▼
JSON Response
```

---

# Conversation Flow

The agent supports the following conversation types:

### 1. Clarification

If the user provides incomplete hiring requirements, the agent asks relevant follow-up questions before searching for assessments.

Example:

User:

```
I am hiring a Data Analyst.
```

Assistant:

```
What technical skills or competencies are required for this role?
```

---

### 2. Recommendation

Once sufficient information is available, the agent searches the SHL catalog and recommends the most relevant assessments.

The response contains:

- Assessment name
- SHL URL
- Test type

---

### 3. Refinement

Users can refine their requirements during the conversation.

Example:

```
Add communication skills.
```

The recommendations are updated accordingly.

---

### 4. Comparison

The agent can compare multiple SHL assessments.

Example:

```
Compare SQL (New) and Python (New)
```

The response highlights their differences and intended use cases.

---

### 5. Refusal

Requests outside the assignment scope are politely declined.

Example:

```
Write a Python program.
```

The agent informs the user that it only assists with SHL assessment recommendations.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd shl-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Add your API key inside `.env`.

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Health Check:

```
GET /health
```

Chat Endpoint:

```
POST /chat
```

---

# Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a Data Analyst."
    }
  ]
}
```

---

# Example Response

```json
{
  "reply": "What technical skills are required for this role?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

# Technologies Used

- Python
- FastAPI
- Pydantic
- Anthropic Claude API
- TF-IDF Retrieval
- Docker

---

# API Endpoints

## GET /health

Returns the application status.

Example Response:

```json
{
  "status": "ok"
}
```

---

## POST /chat

Processes a conversation and returns assessment recommendations.

---

# Design Highlights

- Stateless API design
- Multi-turn conversation support
- Catalog-grounded recommendations
- JSON-based request and response format
- Assessment comparison support
- Scope-aware refusal handling
- FastAPI Swagger documentation

---

# Future Improvements

- Embedding-based semantic retrieval
- Hybrid keyword and vector search
- Ranking improvements
- Better conversation memory
- Automatic catalog updates

---

# Author

Shreya Som

M.Tech – Machine Learning and Computing

Indian Institute of Space Science and Technology (IIST)