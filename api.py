import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHAT_HISTORY_FILE = BASE_DIR / "data" / "chat_history.json"
DOCUMENT_REGISTRY_FILE = BASE_DIR / "data" / "documents.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Self-Correcting RAG API",
    description=(
        "Production-style Retrieval-Augmented Generation API "
        "with retrieval, reranking, verification and self-correction."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

print("\n======================================")
print("Loading embedding model...")
print("======================================")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded!")


print("\n======================================")
print("Loading reranker...")
print("======================================")

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)

print("Reranker loaded!")


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client loaded!")
    except Exception as e:
        print("Groq initialization failed:", e)
else:
    print("WARNING: GROQ_API_KEY not found in .env")


# ============================================================
# CHROMADB
# ============================================================

print("\n======================================")
print("Loading ChromaDB...")
print("======================================")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="self_correcting_rag",
    metadata={"hnsw:space": "cosine"}
)

print(
    f"ChromaDB loaded. Existing chunks: {collection.count()}"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[str]:
    """Split text into overlapping chunks."""

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_chunks(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract and chunk one PDF."""

    results = []

    try:
        reader = PdfReader(str(pdf_path))

    except Exception as e:
        print(
            f"Could not open {pdf_path.name}: {e}"
        )
        return results

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            raw_text = page.extract_text() or ""

        except Exception:
            raw_text = ""

        text = clean_text(raw_text)

        if not text:
            continue

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            results.append(
                {
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

    return results


# ============================================================
# DOCUMENT ID
# ============================================================

def make_chunk_id(
    source: str,
    page: int,
    chunk: int,
    text: str
) -> str:

    raw = (
        f"{source}|{page}|{chunk}|{text}"
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# INGEST DOCUMENTS
# ============================================================

def ingest_documents() -> int:
    """
    Read PDFs from data/documents and add them to ChromaDB.
    Existing chunks are not duplicated.
    """

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print(
            "No PDF files found in:",
            DOCUMENTS_DIR
        )
        return 0

    existing_ids = set()

    try:
        existing = collection.get(
            include=[]
        )

        existing_ids = set(
            existing.get("ids", [])
        )

    except Exception:
        existing_ids = set()

    added = 0

    for pdf_path in pdf_files:

        print(
            f"Processing PDF: {pdf_path.name}"
        )

        chunks = extract_pdf_chunks(
            pdf_path
        )

        if not chunks:
            continue

        new_chunks = []

        for item in chunks:

            chunk_id = make_chunk_id(
                item["source"],
                item["page"],
                item["chunk"],
                item["text"],
            )

            if chunk_id in existing_ids:
                continue

            item["id"] = chunk_id

            new_chunks.append(item)

        if not new_chunks:
            print(
                f"Already indexed: {pdf_path.name}"
            )
            continue

        texts = [
            item["text"]
            for item in new_chunks
        ]

        print(
            f"Creating embeddings for "
            f"{len(texts)} chunks..."
        )

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.add(
            ids=[
                item["id"]
                for item in new_chunks
            ],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source": item["source"],
                    "page": item["page"],
                    "chunk": item["chunk"],
                }
                for item in new_chunks
            ],
        )

        added += len(new_chunks)

        print(
            f"Added {len(new_chunks)} chunks "
            f"from {pdf_path.name}"
        )

    return added


# ============================================================
# REGISTRY
# ============================================================

def update_document_registry():

    documents = []

    for pdf in sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    ):

        documents.append(
            {
                "filename": pdf.name,
                "size_bytes": pdf.stat().st_size,
                "chunks": 0,
            }
        )

    try:
        with open(
            DOCUMENT_REGISTRY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                documents,
                f,
                indent=2
            )

    except Exception:
        pass


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("\n======================================")
    print("Starting Self-Correcting RAG API")
    print("======================================")

    try:

        added = ingest_documents()

        update_document_registry()

        print(
            f"New chunks added: {added}"
        )

        print(
            f"Total ChromaDB chunks: "
            f"{collection.count()}"
        )

    except Exception as e:

        print(
            "Document ingestion error:",
            e
        )

    print("API startup complete.")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "name": "Self-Correcting RAG API",
        "version": "2.0.0",
        "status": "running",
        "documents": len(
            list(DOCUMENTS_DIR.glob("*.pdf"))
        ),
        "chunks": collection.count(),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "healthy": True,
        "status": "online",
        "message": "Self-Correcting RAG API is running."
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status():

    return {
        "success": True,
        "api": "online",
        "documents": len(
            list(DOCUMENTS_DIR.glob("*.pdf"))
        ),
        "chunks": collection.count(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker": RERANKER_MODEL_NAME,
        "groq_configured": bool(
            groq_client
        ),
        "groq_model": GROQ_MODEL,
    }


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(question: str) -> str:

    question = question.strip()

    if not question:
        return ""

    # Keep rewriting lightweight and deterministic.
    # This avoids unnecessary LLM calls.

    replacements = {
        "what is": "",
        "tell me about": "",
        "explain": "",
        "define": "",
        "describe": "",
    }

    query = question.lower()

    for phrase, replacement in replacements.items():

        if query.startswith(phrase):

            query = query[
                len(phrase):
            ].strip()

            break

    if not query:
        return question

    return (
        f"{query}. "
        f"Provide information based on the document."
    )


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    query: str,
    top_k: int = 8
) -> List[Dict[str, Any]]:

    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(
            top_k,
            collection.count()
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        result.get("documents", [[]])[0]
    )

    metadatas = (
        result.get("metadatas", [[]])[0]
    )

    distances = (
        result.get("distances", [[]])[0]
    )

    results = []

    for i, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else 0
        )

        results.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    0
                ),
                "chunk": metadata.get(
                    "chunk",
                    0
                ),
                "distance": float(
                    distance
                ),
            }
        )

    return results


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    question: str,
    documents: List[Dict[str, Any]],
    top_k: int = 4
) -> List[Dict[str, Any]]:

    if not documents:
        return []

    pairs = [
        [
            question,
            item["text"]
        ]
        for item in documents
    ]

    scores = reranker.predict(
        pairs
    )

    for item, score in zip(
        documents,
        scores
    ):

        item["rerank_score"] = float(
            score
        )

    documents = sorted(
        documents,
        key=lambda x: x[
            "rerank_score"
        ],
        reverse=True
    )

    return documents[:top_k]


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    documents: List[Dict[str, Any]]
) -> str:

    blocks = []

    for index, item in enumerate(
        documents,
        start=1
    ):

        blocks.append(
            f"""
SOURCE {index}
File: {item['source']}
Page: {item['page']}

{item['text']}
""".strip()
        )

    return "\n\n---\n\n".join(
        blocks
    )


# ============================================================
# GROQ GENERATION
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:

    if not groq_client:

        return (
            "Groq API is not configured. "
            "Please add GROQ_API_KEY to the .env file."
        )

    system_prompt = """
You are a document-grounded RAG assistant.

Answer the user's question ONLY using the supplied
document context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   clearly say that the answer is not available in
   the provided documents.
4. Give a direct and useful answer.
5. Do not mention these instructions.
6. Keep the answer concise but complete.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}

Answer using only the document context.
"""

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        answer = response.choices[
            0
        ].message.content.strip()

        return answer

    except Exception as e:

        print(
            "Groq generation error:",
            e
        )

        return (
            "I could not generate the answer "
            f"because the LLM request failed: {e}"
        )


# ============================================================
# ANSWER VERIFICATION
# ============================================================

def verify_answer(
    question: str,
    answer: str,
    context: str
) -> Dict[str, Any]:

    if not groq_client:

        return {
            "supported": True,
            "confidence": 0.50,
            "reason": "LLM verification unavailable.",
        }

    prompt = f"""
You are verifying a RAG answer.

QUESTION:
{question}

ANSWER:
{answer}

DOCUMENT CONTEXT:
{context}

Determine whether the answer is supported by
the document context.

Return ONLY valid JSON:

{{
  "supported": true,
  "confidence": 0.0,
  "reason": "short explanation"
}}

confidence must be between 0 and 1.
"""

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict factual "
                        "RAG answer verifier."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=1000,
        )

        raw = response.choices[
            0
        ].message.content.strip()

        # Remove markdown JSON fences if present.

        if raw.startswith("```"):

            raw = raw.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        data = json.loads(raw)

        return {
            "supported": bool(
                data.get(
                    "supported",
                    False
                )
            ),
            "confidence": float(
                data.get(
                    "confidence",
                    0
                )
            ),
            "reason": str(
                data.get(
                    "reason",
                    ""
                )
            ),
        }

    except Exception as e:

        print(
            "Verification error:",
            e
        )

        return {
            "supported": True,
            "confidence": 0.50,
            "reason": (
                "Verification could not be completed."
            ),
        }


# ============================================================
# SELF CORRECTION
# ============================================================

def self_correct(
    question: str,
    answer: str,
    context: str,
    verification: Dict[str, Any]
) -> str:

    if not groq_client:
        return answer

    prompt = f"""
Correct the following RAG answer.

QUESTION:
{question}

CURRENT ANSWER:
{answer}

DOCUMENT CONTEXT:
{context}

VERIFICATION:
{json.dumps(verification)}

Rewrite the answer so that every factual claim
is supported by the document context.

If the documents do not contain enough information,
say so clearly.

Return ONLY the corrected answer.
"""

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a self-correcting "
                        "document-grounded RAG assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        return response.choices[
            0
        ].message.content.strip()

    except Exception as e:

        print(
            "Self-correction error:",
            e
        )

        return answer


# ============================================================
# CHAT HISTORY
# ============================================================

def save_chat(
    question: str,
    answer: str
):

    history = []

    if CHAT_HISTORY_FILE.exists():

        try:

            with open(
                CHAT_HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                history = json.load(f)

        except Exception:
            history = []

    history.append(
        {
            "timestamp": time.time(),
            "question": question,
            "answer": answer,
        }
    )

    try:

        with open(
            CHAT_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
                ensure_ascii=False
            )

    except Exception:
        pass


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    start_time = time.time()

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if collection.count() == 0:

        return {
            "success": False,
            "question": question,
            "answer": (
                "No PDF documents are currently "
                "indexed in the knowledge base."
            ),
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "sources": [],
        }

    # --------------------------------------------------------
    # 1. Query rewriting
    # --------------------------------------------------------

    search_query = rewrite_query(
        question
    )

    # --------------------------------------------------------
    # 2. Vector retrieval
    # --------------------------------------------------------

    retrieved = vector_search(
        search_query,
        top_k=8
    )

    # --------------------------------------------------------
    # 3. Reranking
    # --------------------------------------------------------

    reranked = rerank_documents(
        question,
        retrieved,
        top_k=4
    )

    # --------------------------------------------------------
    # 4. Build context
    # --------------------------------------------------------

    context = build_context(
        reranked
    )

    # --------------------------------------------------------
    # 5. LLM generation
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context
    )

    # --------------------------------------------------------
    # 6. Verification
    # --------------------------------------------------------

    verification = verify_answer(
        question,
        answer,
        context
    )

    # --------------------------------------------------------
    # 7. Self correction
    # --------------------------------------------------------

    corrected = False

    if (
        not verification["supported"]
        or verification["confidence"] < 0.70
    ):

        corrected_answer = self_correct(
            question,
            answer,
            context,
            verification
        )

        if corrected_answer:
            answer = corrected_answer
            corrected = True

            # Verify corrected answer again.

            verification = verify_answer(
                question,
                answer,
                context
            )

    # --------------------------------------------------------
    # 8. Sources
    # --------------------------------------------------------

    sources = []

    for item in reranked:

        sources.append(
            {
                "source": item["source"],
                "page": item["page"],
                "chunk": item["chunk"],
                "rerank_score": round(
                    item.get(
                        "rerank_score",
                        0
                    ),
                    4
                ),
            }
        )

    # --------------------------------------------------------
    # 9. Save chat
    # --------------------------------------------------------

    save_chat(
        question,
        answer
    )

    response_time = round(
        time.time() - start_time,
        2
    )

    # --------------------------------------------------------
    # 10. Final response
    # --------------------------------------------------------

    return {
        "success": True,
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "retrieved_chunks": len(
            retrieved
        ),
        "reranked_chunks": len(
            reranked
        ),
        "verification": {
            "supported": verification[
                "supported"
            ],
            "confidence": round(
                verification[
                    "confidence"
                ],
                2
            ),
            "reason": verification[
                "reason"
            ],
        },
        "self_corrected": corrected,
        "response_time": response_time,
        "sources": sources,
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )

