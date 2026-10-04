import os
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "./chroma_db"

COLLECTION_NAME = "devlens_documents"


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(text, chunk_size=800):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document(
    file_path,
    document_name
):

    text = extract_pdf_text(
        file_path
    )

    chunks = chunk_text(text)

    if not chunks:

        return 0


    embeddings = embedding_model.encode(
        chunks
    ).tolist()


    ids = []

    metadatas = []


    for index in range(
        len(chunks)
    ):

        ids.append(
            f"{document_name}_{index}"
        )

        metadatas.append({

            "source": document_name,

            "chunk": index

        })


    collection.add(

        documents=chunks,

        embeddings=embeddings,

        ids=ids,

        metadatas=metadatas

    )


    return len(chunks)


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    number_of_results=4
):

    question_embedding = (
        embedding_model
        .encode([question])
        .tolist()
    )


    results = collection.query(

        query_embeddings=question_embedding,

        n_results=number_of_results

    )


    documents = results.get(
        "documents",
        [[]]
    )


    if documents:

        return documents[0]


    return []