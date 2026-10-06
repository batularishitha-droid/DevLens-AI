# AI E-Library Chatbot

A complete Streamlit AI E-Library project with:

- Student signup/login
- Admin login
- Book catalog and search
- Book details
- Borrow/return
- Favorites
- My Books
- AI Librarian
- PDF Study Materials
- RAG document question answering
- ChromaDB knowledge base
- Ollama integration
- AI recommendations
- Chat history
- Admin dashboard

## 1. Requirements

- Python 3.10+
- VS Code
- Optional: Ollama for local LLM answers

## 2. Install

Windows PowerShell:

```powershell
cd AI_E_LIBRARY_CHATBOT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## 3. Ollama (optional but recommended)

Install Ollama from its official website, then pull a model such as:

```powershell
ollama pull llama3.2
```

Make sure Ollama is running before using AI-generated answers.

The app defaults to:

- URL: http://localhost:11434
- Model: llama3.2

You can change these with environment variables or a `.env` file.

## 4. Run

```powershell
streamlit run app.py
```

Open the local address shown by Streamlit.

## 5. Demo accounts

Student:

- Username: `student`
- Password: `student123`

Admin:

- Username: `admin`
- Password: `admin123`

Change these before using the project in a real environment.

## 6. RAG workflow

Admin opens Admin Dashboard -> Documents:

1. Upload a PDF.
2. The application extracts PDF text.
3. Text is split into chunks.
4. Chunks are indexed into ChromaDB.
5. Student opens Ask My Documents.
6. The question retrieves relevant chunks.
7. Ollama uses the retrieved context to generate an answer.
8. Sources are displayed.

If ChromaDB is unavailable, the app has a lightweight keyword-retrieval fallback so the document feature remains usable.

## 7. GitHub

Do not commit:

- `.env`
- `data/library.db`
- uploaded private PDFs
- `chroma_db/`

The included `.gitignore` excludes these.

A good first commit is:

```powershell
git init
git add .
git commit -m "Initial AI E-Library project"
```

Then create a GitHub repository and push the project.

## 8. Project structure

```text
AI_E_LIBRARY_CHATBOT/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
├── uploads/
├── chroma_db/
├── assets/
└── sample_materials/
```

The app intentionally uses one main `app.py` so it is easy to copy, run, and upload to GitHub. It can later be split into modules once the project is stable.
