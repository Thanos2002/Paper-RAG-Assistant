# GEMINI CLI SYSTEM INSTRUCTIONS

## 🛠 TECH STACK CONTEXT
- **Framework**: FastAPI (Python 3.9+)
- **Primary LLM**: Google Gemini (generative-ai SDK)
- **Vector DB**: ChromaDB / FAISS
- **Tracking**: MLflow
- **Database**: PostgreSQL with pgvector / SQLite
- **Architecture**: RAG (Retrieval-Augmented Generation)

## ⚠️ OPERATIONAL PROTOCOL (MANDATORY)
1. **NO SILENT WRITES**: You are strictly prohibited from modifying, creating, or deleting files in this repository without explicit user authorization.
2. **PROPOSE BEFORE ACTING**: For every code change, you must first:
    - Display a `diff` or a summary of the intended changes.
    - Ask the user: "Should I apply these changes? (y/n)"
3. **EXECUTION**: Only execute the write command if the user responds with "y", "yes", or a similar affirmative.

