import os
import hashlib
import time
import json
import math
from src.performance import render_rag_performance
from datetime import datetime

import streamlit as st
import chromadb

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000
OVERLAP = 200

TOP_K = 8
FINAL_CONTEXT_CHUNKS = 4

MAX_RETRIES = 2

COLLECTION_NAME = "multi_document_rag"

CHAT_HISTORY_FILE = "data/chat_history.json"
DOCUMENTS_FILE = "data/documents.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("data/documents", exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    st.error(
        "GROQ_API_KEY not found in .env file."
    )

    st.stop()


# ============================================================
# LOAD AI MODELS
# ============================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    llm = Groq(
        api_key=api_key
    )

    return embedding_model, reranker, llm


embedding_model, reranker, llm = load_models()


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


def get_collection():

    try:

        return client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        return client.create_collection(
            name=COLLECTION_NAME
        )


collection = get_collection()


# ============================================================
# DOCUMENT REGISTRY
# ============================================================

def load_documents():

    if not os.path.exists(DOCUMENTS_FILE):
        return {}

    try:
        with open(
            DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        # New format: dictionary keyed by file hash.
        if isinstance(data, dict):
            return data

        # Older versions stored the registry as a list.
        # Convert it to the dictionary format used by the
        # rest of this application.
        if isinstance(data, list):
            converted = {}

            for index, document in enumerate(data):

                if not isinstance(document, dict):
                    continue

                filename = str(
                    document.get(
                        "filename",
                        f"document_{index + 1}.pdf"
                    )
                )

                file_hash = document.get("file_hash")

                if not file_hash:
                    file_hash = document.get("hash")

                if not file_hash:
                    file_hash = filename

                item = dict(document)
                item["filename"] = filename

                try:
                    item["chunks"] = int(
                        item.get("chunks", 0) or 0
                    )
                except (TypeError, ValueError):
                    item["chunks"] = 0

                converted[str(file_hash)] = item

            return converted

        return {}

    except Exception as e:
        st.warning(
            f"Could not load document registry: {e}"
        )
        return {}


def save_documents(documents):

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )


def get_file_hash(pdf_file):

    return hashlib.md5(
        pdf_file.getvalue()
    ).hexdigest()


# ============================================================
# CHAT HISTORY
# ============================================================

def load_chat_history():

    if not os.path.exists(
        CHAT_HISTORY_FILE
    ):

        return []

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # Chat history from older versions of the app may contain
        # dictionaries without a role/content pair.  Streamlit's
        # st.chat_message() requires a valid role, so normalize the
        # history before it is used anywhere in the application.
        if not isinstance(data, list):
            return []

        cleaned = []

        for item in data:

            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and content is not None:
                cleaned.append({
                    "role": role,
                    "content": str(content)
                })
                continue

            # Support older history records such as:
            # {"question": "...", "answer": "..."}
            question = item.get("question")
            answer = item.get("answer")

            if question:
                cleaned.append({
                    "role": "user",
                    "content": str(question)
                })

            if answer:
                cleaned.append({
                    "role": "assistant",
                    "content": str(answer)
                })

        # Persist the normalized format so the error cannot return
        # on the next Streamlit restart.
        if cleaned != data:
            save_chat_history(cleaned)

        return cleaned

    except Exception as e:

        st.warning(
            f"Could not load chat history: {e}"
        )

        return []


def save_chat_history(messages):

    try:

        with open(
            CHAT_HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                messages,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception:

        pass


def clear_chat_history():

    if os.path.exists(
        CHAT_HISTORY_FILE
    ):

        try:

            os.remove(
                CHAT_HISTORY_FILE
            )

        except Exception:

            pass


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = (
        load_chat_history()
    )


if "last_metrics" not in st.session_state:

    st.session_state.last_metrics = None


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(file_hash):

    documents = load_documents()

    if file_hash not in documents:

        return False

    document_info = documents[
        file_hash
    ]

    chunk_ids = document_info.get(
        "chunk_ids",
        []
    )

    if chunk_ids:

        try:

            collection.delete(
                ids=chunk_ids
            )

        except Exception as error:

            print(
                "ChromaDB delete error:",
                error
            )

    pdf_path = document_info.get(
        "path"
    )

    if pdf_path and os.path.exists(
        pdf_path
    ):

        try:

            os.remove(
                pdf_path
            )

        except Exception:

            pass

    del documents[
        file_hash
    ]

    save_documents(
        documents
    )

    return True


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

def clear_knowledge_base():

    global collection

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    documents = load_documents()

    for document in documents.values():

        pdf_path = document.get(
            "path"
        )

        if pdf_path and os.path.exists(
            pdf_path
        ):

            try:

                os.remove(
                    pdf_path
                )

            except Exception:

                pass

    save_documents({})


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(pdf_file):

    file_hash = get_file_hash(
        pdf_file
    )

    documents = load_documents()

    if file_hash in documents:

        return (
            documents[file_hash]["chunks"],
            True,
            file_hash
        )

    reader = PdfReader(
        pdf_file
    )

    chunks = []
    metadatas = []
    ids = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:

            continue

        start = 0

        while start < len(text):

            end = (
                start +
                CHUNK_SIZE
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:

                chunk_id = (
                    f"{file_hash}_"
                    f"{page_number}_"
                    f"{start}"
                )

                chunks.append(
                    chunk
                )

                metadatas.append(
                    {
                        "source":
                            pdf_file.name,

                        "page":
                            page_number,

                        "file_hash":
                            file_hash
                    }
                )

                ids.append(
                    chunk_id
                )

            start = (
                end -
                OVERLAP
            )

    if not chunks:

        return (
            0,
            False,
            file_hash
        )

    embeddings = (
        embedding_model.encode(
            chunks,
            show_progress_bar=False
        )
    )

    collection.upsert(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist(),

        metadatas=metadatas
    )

    safe_filename = os.path.basename(
        pdf_file.name
    )

    pdf_path = os.path.join(
        "data",
        "documents",
        safe_filename
    )

    with open(
        pdf_path,
        "wb"
    ) as output_file:

        output_file.write(
            pdf_file.getbuffer()
        )

    documents[file_hash] = {

        "filename":
            safe_filename,

        "chunks":
            len(chunks),

        "path":
            pdf_path,

        "chunk_ids":
            ids
    }

    save_documents(
        documents
    )

    return (
        len(chunks),
        False,
        file_hash
    )


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(
    question,
    chat_history
):

    history_text = ""

    for message in chat_history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a search query optimization system.

Rewrite the user's latest question into
a clear standalone search query.

Use conversation history when necessary.

Rules:

1. Preserve the meaning.
2. Resolve references such as it, this,
   that, they and previous topic.
3. Make the query specific.
4. Do not answer the question.
5. Return ONLY the rewritten query.

CONVERSATION HISTORY:

{history_text}

LATEST QUESTION:

{question}

REWRITTEN SEARCH QUERY:
"""

    response = llm.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# SELF-CORRECTION QUERY GENERATION
# ============================================================

def generate_correction_query(
    question,
    previous_query,
    previous_answer,
    context,
    previous_queries
):
    """
    Generate a genuinely different search query after an
    unsupported answer. The correction is based on the original
    question and the evidence retrieved on the previous attempt.

    The model is asked to improve retrieval, not to answer the
    user's question.
    """

    previous_query = previous_query or question

    previous_queries_text = "\n".join(
        f"- {query}"
        for query in previous_queries
    )

    if not previous_queries_text:
        previous_queries_text = "- None"

    prompt = f"""
You are a retrieval self-correction system for a PDF question-answering application.

The previous retrieval attempt did not produce a document-supported answer.
Create ONE improved search query for the next retrieval attempt.

IMPORTANT RULES:
1. Preserve the user's original intent.
2. Do NOT answer the question.
3. Do NOT add facts from outside the documents.
4. Use useful terminology visible in the previous context when appropriate.
5. If the previous query was too broad, make it more specific.
6. If it was too specific, use important synonyms or related document terminology.
7. The new query must be meaningfully different from the previous query.
8. Return ONLY the search query, with no explanation.

ORIGINAL QUESTION:
{question}

PREVIOUS SEARCH QUERY:
{previous_query}

PREVIOUS ANSWER:
{previous_answer}

PREVIOUS DOCUMENT CONTEXT:
{context[:7000]}

QUERIES ALREADY TRIED:
{previous_queries_text}

IMPROVED SEARCH QUERY:
"""

    try:
        response = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You improve document retrieval queries. Return only one search query."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        query = (
            response.choices[0].message.content
            .strip()
            .replace("\n", " ")
        )

        # Remove accidental labels returned by the model.
        prefixes = [
            "IMPROVED SEARCH QUERY:",
            "SEARCH QUERY:",
            "QUERY:"
        ]

        for prefix in prefixes:
            if query.upper().startswith(prefix):
                query = query[len(prefix):].strip()

        # Never allow an empty or duplicate query to enter the
        # next retrieval attempt.
        if not query:
            return previous_query

        if query.lower() in {q.lower() for q in previous_queries}:
            return previous_query

        return query

    except Exception as error:
        print("Self-correction query error:", error)
        return previous_query


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_documents(
    search_query
):

    if collection.count() == 0:

        return [], []

    query_embedding = (
        embedding_model
        .encode(
            [search_query]
        )
        .tolist()
    )

    results = collection.query(

        query_embeddings=
            query_embedding,

        n_results=min(
            TOP_K,
            collection.count()
        ),

        include=[
            "documents",
            "metadatas"
        ]
    )

    return (

        results["documents"][0],

        results["metadatas"][0]
    )


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    question,
    documents,
    metadatas
):

    if not documents:

        return []

    pairs = []

    for document in documents:

        pairs.append(
            [
                question,
                document
            ]
        )

    scores = reranker.predict(
        pairs
    )

    ranked = list(
        zip(
            documents,
            metadatas,
            scores
        )
    )

    ranked.sort(
        key=lambda x: float(x[2]),
        reverse=True
    )

    return ranked[
        :FINAL_CONTEXT_CHUNKS
    ]


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(
    ranked_results,
    verification_result
):
    """
    Calculate an evidence-based confidence indicator.

    Rules:
    - NOT_SUPPORTED always returns 0%.
    - SUPPORTED answers receive a score based on the quality and
      quantity of the retrieved/reranked evidence.
    - The score is intentionally not treated as a calibrated
      probability.
    """

    # ---------------------------------------------------------
    # Verification is the strongest gate.
    # ---------------------------------------------------------
    if verification_result != "SUPPORTED":
        return 0, "LOW"

    if not ranked_results:
        return 0, "LOW"

    scores = []

    for item in ranked_results:
        try:
            scores.append(float(item[2]))
        except (TypeError, ValueError, IndexError):
            continue

    if not scores:
        return 0, "LOW"

    # ---------------------------------------------------------
    # Cross-encoder scores are logits, not probabilities.
    # Convert them to a bounded signal only for UI confidence.
    # ---------------------------------------------------------
    def sigmoid(value):
        value = max(-20.0, min(20.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    top_score = max(scores)
    average_score = sum(scores) / len(scores)

    top_signal = sigmoid(top_score)
    average_signal = sigmoid(average_score)

    # ---------------------------------------------------------
    # Evidence quantity.
    # ---------------------------------------------------------
    evidence_signal = min(
        len(scores) / max(FINAL_CONTEXT_CHUNKS, 1),
        1.0
    )

    # Chunks with non-negative reranker scores are treated as
    # relatively stronger evidence.
    strong_chunks = sum(
        1 for score in scores
        if score >= 0
    )

    strong_evidence_signal = min(
        strong_chunks / max(FINAL_CONTEXT_CHUNKS, 1),
        1.0
    )

    # ---------------------------------------------------------
    # Combine evidence signals.
    # ---------------------------------------------------------
    evidence_score = (
        top_signal * 0.35
        + average_signal * 0.25
        + evidence_signal * 0.15
        + strong_evidence_signal * 0.25
    )

    # ---------------------------------------------------------
    # IMPORTANT FIX:
    #
    # A verified answer should not become "LOW CONFIDENCE 61%"
    # simply because cross-encoder logits are negative.
    #
    # Verification already established that the answer is
    # supported. Evidence quality then adjusts the score between
    # 80% and 98%.
    # ---------------------------------------------------------
    percentage = round(
        80 + (evidence_score * 18)
    )

    percentage = max(
        80,
        min(
            percentage,
            98
        )
    )

    if percentage >= 90:
        level = "HIGH"
    else:
        level = "MEDIUM"

    return percentage, level


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
    chat_history
):

    history_text = ""

    for message in chat_history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the information contained in the document
context.

Do not use outside knowledge.

If the answer cannot be found in the
documents, say:

I don't know based on the provided documents.

Never invent facts.

CONVERSATION HISTORY:

{history_text}

DOCUMENT CONTEXT:

{context}

CURRENT QUESTION:

{question}

ANSWER:
"""

    response = llm.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# ANSWER VERIFICATION
# ============================================================

def verify_answer(
    question,
    answer,
    context
):
    """
    Deterministically verify whether the generated answer is
    supported by the retrieved document context.

    IMPORTANT:
    Answers that explicitly say the information is not in the
    documents are always NOT_SUPPORTED. This prevents the LLM
    verifier from incorrectly classifying an "I don't know"
    response as SUPPORTED.
    """

    if not question or not answer or not context:
        return "NOT_SUPPORTED"

    answer_lower = answer.strip().lower()

    unknown_phrases = [
        "i don't know based on the provided documents",
        "i don't know based on the provided document",
        "i don't know based on the documents",
        "not found in the provided documents",
        "not found in the provided document",
        "cannot be determined from the provided documents",
        "cannot be determined from the provided document",
        "insufficient information in the provided documents",
        "insufficient information in the provided document",
        "i could not find enough supported information in the uploaded documents",
        "i could not find enough information in the uploaded documents"
    ]

    for phrase in unknown_phrases:
        if phrase in answer_lower:
            print("Verifier deterministic result: NOT_SUPPORTED")
            return "NOT_SUPPORTED"

    prompt = f"""
You are a strict document-grounded answer verification system.

Determine whether the ANSWER is supported by the DOCUMENT CONTEXT.

Rules:
1. Use ONLY the DOCUMENT CONTEXT.
2. Do not use outside knowledge.
3. Paraphrasing is allowed.
4. Every important factual claim in the answer must be supported
   by the document context.
5. If the context does not contain enough information, return
   NOT_SUPPORTED.
6. Return exactly one label: SUPPORTED or NOT_SUPPORTED.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
{answer}

CLASSIFICATION:
"""

    try:
        response = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict document evidence verifier."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        result = (
            response.choices[0].message.content
            .strip()
            .upper()
        )

        print("Verifier result:", result)

        if "NOT_SUPPORTED" in result:
            return "NOT_SUPPORTED"

        if "SUPPORTED" in result:
            return "SUPPORTED"

        return "NOT_SUPPORTED"

    except Exception as error:
        print("Answer verification error:", error)
        return "NOT_SUPPORTED"


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🤖 Self-Correcting RAG System"
)

st.write(
    "An advanced multi-document RAG system "
    "with retrieval, reranking, verification "
    "and self-correction."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )

    uploaded_files = st.file_uploader(

        "Upload PDF documents",

        type=["pdf"],

        accept_multiple_files=True
    )

    st.divider()

    # ========================================================
    # DOCUMENT MANAGER
    # ========================================================

    st.subheader(
        "📂 Document Manager"
    )

    documents_registry = load_documents()

    if documents_registry:

        for (
            file_hash,
            document
        ) in list(
            documents_registry.items()
        ):

            filename = document[
                "filename"
            ]

            chunk_count = document[
                "chunks"
            ]

            st.write(
                f"📄 **{filename}**"
            )

            st.caption(
                f"{chunk_count} chunks"
            )

            if st.button(
                f"🗑️ Remove {filename}",
                key=f"remove_{file_hash}"
            ):

                if delete_document(
                    file_hash
                ):

                    st.success(
                        f"{filename} removed."
                    )

                    st.rerun()

    else:

        st.caption(
            "No documents in knowledge base."
        )

    st.divider()

    if st.button(
        "🧹 Clear Knowledge Base",
        use_container_width=True
    ):

        clear_knowledge_base()

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()

    st.divider()

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader(
        "⚙️ RAG Pipeline"
    )

    st.write(
        "Query Rewriting: ✅"
    )

    st.write(
        "Vector Search: ✅"
    )

    st.write(
        "Reranking: ✅"
    )

    st.write(
        "LLM Generation: ✅"
    )

    st.write(
        "Answer Verification: ✅"
    )

    st.write(
        "Self-Correction: ✅"
    )

    st.write(
        "Chat Memory: ✅"
    )

    st.write(
        "Persistent History: ✅"
    )

    st.write(
        "Document Management: ✅"
    )

    st.write(
        "Source Evidence: ✅"
    )

    st.write(
        "Confidence Scoring: ✅"
    )

    st.divider()

    # ========================================================
    # KNOWLEDGE BASE STATS
    # ========================================================

    st.subheader(
        "📊 Knowledge Base"
    )

    total_documents = len(
        documents_registry
    )

    total_chunks = sum(

        document["chunks"]

        for document
        in documents_registry.values()

    )

    st.write(
        f"Documents: **{total_documents}**"
    )

    st.write(
        f"Chunks: **{total_chunks}**"
    )

    st.divider()

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        clear_chat_history()

        st.rerun()


# ============================================================
# PROCESS UPLOADED FILES
# ============================================================

if uploaded_files:

    st.subheader(
        "📥 Processing Documents"
    )

    for pdf_file in uploaded_files:

        with st.spinner(
            f"Processing {pdf_file.name}..."
        ):

            (
                chunk_count,
                existed,
                file_hash
            ) = process_pdf(
                pdf_file
            )

        if existed:

            st.info(
                f"ℹ️ {pdf_file.name} "
                f"is already in the knowledge base."
            )

        elif chunk_count > 0:

            st.success(
                f"✅ {pdf_file.name} "
                f"→ {chunk_count} chunks"
            )

        else:

            st.error(
                f"❌ Could not extract text "
                f"from {pdf_file.name}"
            )


# ============================================================
# DOCUMENT LIST
# ============================================================

documents_registry = load_documents()

if documents_registry:

    st.subheader(
        "📄 Documents in Knowledge Base"
    )

    cols = st.columns(
        min(
            len(documents_registry),
            3
        )
    )

    for index, document in enumerate(
        documents_registry.values()
    ):

        with cols[
            index % len(cols)
        ]:

            st.info(
                f"📄 **{document['filename']}**\n\n"
                f"Chunks: {document['chunks']}"
            )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your PDFs..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    if collection.count() == 0:

        st.warning(
            "Please upload at least one PDF first."
        )

        st.stop()

    start_time = time.time()

    attempts = 0

    retrieved_count = 0

    reranked_count = 0

    verification_result = "NOT_RUN"

    rewritten_queries = []

    confidence_percentage = 0

    confidence_level = "LOW"

    st.session_state.messages.append(

        {
            "role":
                "user",

            "content":
                question
        }

    )

    save_chat_history(
        st.session_state.messages
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):

        status_box = st.status(
            "🧠 Thinking...",
            expanded=True
        )

        current_query = question

        final_answer = None

        final_sources = []

        attempt_details = []

        # ====================================================
        # SELF-CORRECTION LOOP
        # ====================================================

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            attempts = attempt

            status_box.write(
                f"🔄 Attempt {attempt}"
            )

            # ------------------------------------------------
            # Query generation / self-correction
            # ------------------------------------------------

            if attempt == 1:

                status_box.write(
                    "🔄 Rewriting query..."
                )

                current_query = rewrite_query(
                    question,
                    st.session_state.messages[:-1]
                )

            else:

                status_box.write(
                    "🧠 Generating a corrected search query..."
                )

                current_query = generate_correction_query(
                    question,
                    current_query,
                    answer,
                    context,
                    rewritten_queries
                )

            rewritten_queries.append(current_query)

            status_box.write(
                f"Search query: `{current_query}`"
            )

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            status_box.write(
                "🔎 Searching documents..."
            )

            (
                documents,
                metadatas
            ) = retrieve_documents(
                current_query
            )

            retrieved_count = len(
                documents
            )

            # ------------------------------------------------
            # Reranking
            # ------------------------------------------------

            status_box.write(
                "📊 Reranking results..."
            )

            ranked_results = rerank_documents(

                question,

                documents,

                metadatas

            )

            reranked_count = len(
                ranked_results
            )

            # ------------------------------------------------
            # Context
            # ------------------------------------------------

            context_parts = []

            for (
                document,
                metadata,
                score
            ) in ranked_results:

                context_parts.append(

                    f"""
SOURCE: {metadata['source']}
PAGE: {metadata['page']}

CONTENT:
{document}
"""

                )

            context = "\n\n".join(
                context_parts
            )

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            status_box.write(
                "🤖 Generating answer..."
            )

            answer = generate_answer(

                question,

                context,

                st.session_state.messages[:-1]

            )

            # ------------------------------------------------
            # Verify
            # ------------------------------------------------

            status_box.write(
                "🛡️ Verifying answer..."
            )

            verification_result = verify_answer(

                question,

                answer,

                context

            )

            # ------------------------------------------------
            # Deterministic safety guard
            # ------------------------------------------------
            # Never allow an explicit "I don't know" answer to
            # become SUPPORTED because of an LLM verifier mistake.
            answer_lower = answer.strip().lower()

            unknown_answer_markers = [
                "i don't know based on the provided documents",
                "i don't know based on the provided document",
                "not found in the provided documents",
                "cannot be determined from the provided documents",
                "insufficient information in the provided documents"
            ]

            if any(
                marker in answer_lower
                for marker in unknown_answer_markers
            ):
                verification_result = "NOT_SUPPORTED"

            status_box.write(

                f"Verification: "
                f"`{verification_result}`"

            )

            # ------------------------------------------------
            # Calculate confidence
            # ------------------------------------------------

            (
                confidence_percentage,
                confidence_level
            ) = calculate_confidence(

                ranked_results,

                verification_result

            )

            # ------------------------------------------------
            # Supported
            # ------------------------------------------------

            if verification_result == "SUPPORTED":

                final_answer = answer

                final_sources = ranked_results

                attempt_details.append({
                    "attempt": attempt,
                    "query": current_query,
                    "retrieved": retrieved_count,
                    "reranked": reranked_count,
                    "verification": verification_result,
                    "answer": answer
                })

                break

            # ------------------------------------------------
            # Record failed attempt
            # ------------------------------------------------

            attempt_details.append({
                "attempt": attempt,
                "query": current_query,
                "retrieved": retrieved_count,
                "reranked": reranked_count,
                "verification": verification_result,
                "answer": answer
            })

            # ------------------------------------------------
            # Self correction
            # ------------------------------------------------

            status_box.write(
                "⚠️ Answer not supported."
            )

            if attempt < MAX_RETRIES:
                status_box.write(
                    "🔁 Self-correcting query for the next attempt..."
                )

        # ====================================================
        # RESPONSE TIME
        # ====================================================

        response_time = round(

            time.time() -
            start_time,

            2

        )

        status_box.update(

            label="✅ Complete",

            state="complete",

            expanded=False

        )

        # ====================================================
        # ANSWER
        # ====================================================

        if final_answer:

            st.markdown(
                final_answer
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            st.markdown(
                "### 🎯 Answer Confidence"
            )

            if confidence_level == "HIGH":

                st.success(
                    f"🟢 HIGH CONFIDENCE — "
                    f"{confidence_percentage}%"
                )

                st.caption(
                    "The answer has strong "
                    "document evidence and passed "
                    "answer verification."
                )

            elif confidence_level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM CONFIDENCE — "
                    f"{confidence_percentage}%"
                )

                st.caption(
                    "The answer has relevant "
                    "evidence, but the evidence "
                    "strength is moderate."
                )

            else:

                st.error(
                    f"🔴 LOW CONFIDENCE — "
                    f"{confidence_percentage}%"
                )

                st.caption(
                    "The retrieved evidence is weak. "
                    "Treat the answer cautiously."
                )

            # =================================================
            # SOURCES & EVIDENCE
            # =================================================

            st.markdown(
                "### 📚 Sources & Evidence"
            )

            st.caption(
                "The following document excerpts were used "
                "to generate and verify the answer."
            )

            seen_sources = set()

            for index, (
                document,
                metadata,
                score
            ) in enumerate(
                final_sources,
                start=1
            ):

                source_key = (
                    metadata["source"],
                    metadata["page"]
                )

                if source_key in seen_sources:
                    continue

                seen_sources.add(source_key)

                # -----------------------------------------
                # SOURCE CARD
                # -----------------------------------------

                with st.expander(
                    f"📄 Source {index} — "
                    f"{metadata['source']} "
                    f"| Page {metadata['page']}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            "**📄 Document**"
                        )
                        st.write(
                            metadata["source"]
                        )

                    with col2:
                        st.markdown(
                            "**📖 Page**"
                        )
                        st.write(
                            metadata["page"]
                        )

                    st.divider()

                    # -----------------------------------------
                    # RELEVANCE SCORE
                    # -----------------------------------------

                    st.markdown(
                        "**🎯 Evidence Relevance**"
                    )

                    relevance_score = float(score)

                    # Cross-encoder scores are not calibrated
                    # probabilities, so this is only a visual
                    # relevance indicator.
                    progress_value = (
                        1.0 /
                        (
                            1.0 +
                            math.exp(
                                -relevance_score
                            )
                        )
                    )

                    progress_value = max(
                        0.0,
                        min(
                            progress_value,
                            1.0
                        )
                    )

                    st.progress(
                        progress_value
                    )

                    st.caption(
                        f"Reranking score: "
                        f"{relevance_score:.4f}"
                    )

                    st.divider()

                    # -----------------------------------------
                    # EVIDENCE
                    # -----------------------------------------

                    st.markdown(
                        "**🔎 Evidence used by the AI**"
                    )

                    st.info(
                        document
                    )

                    st.success(
                        "✅ This excerpt was included "
                        "in the answer generation context."
                    )

            # =================================================
            # PIPELINE DETAILS
            # =================================================

            st.markdown("### 🔬 RAG Pipeline Details")

            with st.expander("View query rewriting and self-correction details"):

                st.markdown("**Original question**")
                st.code(question)

                st.markdown("**Search queries used**")

                if rewritten_queries:
                    for index, query_text in enumerate(rewritten_queries, start=1):
                        st.write(f"**Attempt {index}:** {query_text}")
                else:
                    st.write("No rewritten query was recorded.")

                st.markdown("**Pipeline summary**")
                st.write(
                    f"Retrieved **{retrieved_count}** chunks → "
                    f"Reranked **{reranked_count}** chunks → "
                    f"Verification: **{verification_result}** → "
                    f"Attempts: **{attempts}**"
                )

                st.markdown("**Self-correction attempts**")

                if attempt_details:
                    for detail in attempt_details:

                        result_icon = (
                            "✅"
                            if detail["verification"] == "SUPPORTED"
                            else "❌"
                        )

                        with st.expander(
                            f"Attempt {detail['attempt']} — {result_icon} {detail['verification']}"
                        ):
                            st.markdown("**Search query**")
                            st.code(detail["query"])

                            st.write(
                                f"Retrieved: **{detail['retrieved']}** | "
                                f"Reranked: **{detail['reranked']}**"
                            )

                            st.markdown("**Generated answer**")
                            st.write(detail["answer"])
                else:
                    st.write("No self-correction attempts were recorded.")

        else:

            # ------------------------------------------------
            # UNSUPPORTED FALLBACK
            # ------------------------------------------------
            # This is deliberately explicit so an unsupported
            # question can never inherit stale confidence or
            # sources from an earlier attempt.
            final_answer = (
                "I don't know based on the provided documents."
            )

            final_sources = []

            verification_result = "NOT_SUPPORTED"

            confidence_percentage = 0

            confidence_level = "LOW"

            st.markdown(
                final_answer
            )

            st.markdown(
                "### 🎯 Answer Confidence"
            )

            st.error(
                "🔴 LOW CONFIDENCE — 0%"
            )

            st.caption(
                "The requested information was not supported "
                "by the uploaded documents."
            )

            st.markdown(
                "### 🛡️ Verification"
            )

            st.error(
                "❌ NOT_SUPPORTED"
            )

            st.markdown(
                "### 📚 Sources & Evidence"
            )

            st.info(
                "No supporting evidence was found for this "
                "question in the uploaded documents."
            )

        # ====================================================
        # SAVE ASSISTANT MESSAGE
        # ====================================================

        if final_answer:

            assistant_message = {

                "role":
                    "assistant",

                "content":
                    final_answer

            }

        else:

            assistant_message = {

                "role":
                    "assistant",

                "content":
                    (
                        "I could not find enough "
                        "supported information in "
                        "the uploaded documents."
                    )

            }

        st.session_state.messages.append(
            assistant_message
        )

        save_chat_history(
            st.session_state.messages
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    st.session_state.last_metrics = {

        "documents":
            len(
                load_documents()
            ),

        "retrieved":
            retrieved_count,

        "reranked":
            reranked_count,

        "attempts":
            attempts,

        "rewritten_queries":
            rewritten_queries,

        "attempt_details":
            attempt_details,

        "verification":
            verification_result,

        "confidence":
            confidence_percentage,

        "confidence_level":
            confidence_level,

        "response_time":
            response_time

    }


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

if st.session_state.last_metrics:

    metrics = (
        st.session_state.last_metrics
    )

    st.divider()

    st.subheader(
        "📊 RAG Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Documents",
            metrics["documents"]
        )

    with col2:

        st.metric(
            "🔎 Retrieved",
            metrics["retrieved"]
        )

    with col3:

        st.metric(
            "📊 Reranked",
            metrics["reranked"]
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "🔄 Attempts",
            metrics["attempts"]
        )

    with col5:

        st.metric(
            "⏱️ Response Time",
            f"{metrics['response_time']} sec"
        )

    with col6:

        if metrics["verification"] == "SUPPORTED":

            st.metric(
                "🛡️ Verification",
                "✅ SUPPORTED"
            )

        else:

            st.metric(
                "🛡️ Verification",
                "❌ NOT SUPPORTED"
            )

    # ========================================================
    # CONFIDENCE METRIC
    # ========================================================

    st.markdown(
        "### 🎯 Confidence"
    )

    confidence_value = (
        metrics["confidence"]
    )

    confidence_level = (
        metrics["confidence_level"]
    )

    if confidence_level == "HIGH":

        st.success(
            f"🟢 HIGH — "
            f"{confidence_value}%"
        )

    elif confidence_level == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM — "
            f"{confidence_value}%"
        )

    else:

        st.error(
            f"🔴 LOW — "
            f"{confidence_value}%"
        )


# ============================================================
# AUTOMATIC RAG EVALUATION
# ============================================================

EVALUATION_FILE = "data/evaluation_results.json"
EVALUATION_HISTORY_FILE = "data/evaluation_history.json"


def load_evaluation_history():
    """Load previous automatic evaluation runs."""
    if not os.path.exists(EVALUATION_HISTORY_FILE):
        return []

    try:
        with open(EVALUATION_HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []
    except Exception as error:
        print("Evaluation history load error:", error)
        return []


def save_evaluation_history(
    evaluation_results,
    accuracy,
    passed_count,
    failed_count,
    average_attempts,
    average_response_time,
):
    """Append a complete evaluation summary, including RAG pipeline metrics."""
    history = load_evaluation_history()

    total = len(evaluation_results)

    if total > 0:
        average_retrieved = round(
            sum(float(result.get("retrieved", 0) or 0) for result in evaluation_results)
            / total,
            2,
        )

        average_reranked = round(
            sum(float(result.get("reranked", 0) or 0) for result in evaluation_results)
            / total,
            2,
        )

        verification_success = round(
            (
                sum(
                    1
                    for result in evaluation_results
                    if result.get("verification") == "SUPPORTED"
                )
                / total
            )
            * 100,
            2,
        )
    else:
        average_retrieved = 0.0
        average_reranked = 0.0
        verification_success = 0.0

    history.append({
        "run": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total,
        "passed": passed_count,
        "failed": failed_count,
        "accuracy": accuracy,
        "average_attempts": average_attempts,
        "average_response_time": average_response_time,

        # RAG pipeline performance metrics
        "average_retrieved": average_retrieved,
        "average_reranked": average_reranked,
        "verification_success": verification_success,
    })

    try:
        with open(EVALUATION_HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4, ensure_ascii=False)
    except Exception as error:
        print("Evaluation history save error:", error)


DEFAULT_EVALUATION_QUESTIONS = [
    {
        "question": "What is Generative AI?",
        "type": "Supported",
    },
    {
        "question": "What are the main characteristics of Generative AI?",
        "type": "Supported",
    },
    {
        "question": "What are the key features of Generative AI?",
        "type": "Supported",
    },
    {
        "question": "What is the population of Japan?",
        "type": "Unsupported",
    },
    {
        "question": "How does a petrol engine work?",
        "type": "Unsupported",
    },
]


def run_rag_for_evaluation(question):
    """
    Run the same RAG pipeline used by the chatbot, but without
    rendering the chat UI. Returns metrics and the final answer.
    """

    start_time = time.time()

    attempts = 0
    retrieved_count = 0
    reranked_count = 0
    verification_result = "NOT_RUN"

    rewritten_queries = []
    attempt_details = []

    current_query = question
    final_answer = None

    if collection.count() == 0:
        return {
            "question": question,
            "answer": "NO_DOCUMENTS",
            "verification": "NOT_SUPPORTED",
            "confidence": 0,
            "confidence_level": "LOW",
            "attempts": 0,
            "retrieved": 0,
            "reranked": 0,
            "queries": [],
            "response_time": round(time.time() - start_time, 2),
            "attempt_details": [],
        }

    for attempt in range(1, MAX_RETRIES + 1):

        attempts = attempt

        try:

            if attempt == 1:

                current_query = rewrite_query(
                    question,
                    []
                )

            else:

                previous_answer = attempt_details[-1]["answer"] if attempt_details else ""

                previous_context = attempt_details[-1].get("context", "") if attempt_details else ""

                current_query = generate_correction_query(
                    question,
                    current_query,
                    previous_answer,
                    previous_context,
                    rewritten_queries
                )

        except Exception as error:

            print("Evaluation query generation error:", error)

        if not current_query:
            current_query = question

        rewritten_queries.append(current_query)

        documents, metadatas = retrieve_documents(
            current_query
        )

        retrieved_count = len(documents)

        ranked_results = rerank_documents(
            question,
            documents,
            metadatas
        )

        reranked_count = len(ranked_results)

        context_parts = []

        for document, metadata, score in ranked_results:

            context_parts.append(
                f"""
SOURCE: {metadata['source']}
PAGE: {metadata['page']}

CONTENT:
{document}
"""
            )

        context = "\n\n".join(context_parts)

        answer = generate_answer(
            question,
            context,
            []
        )

        verification_result = verify_answer(
            question,
            answer,
            context
        )

        answer_lower = answer.strip().lower()

        unknown_answer_markers = [
            "i don't know based on the provided documents",
            "i don't know based on the provided document",
            "not found in the provided documents",
            "cannot be determined from the provided documents",
            "insufficient information in the provided documents",
        ]

        if any(
            marker in answer_lower
            for marker in unknown_answer_markers
        ):
            verification_result = "NOT_SUPPORTED"

        confidence_percentage, confidence_level = calculate_confidence(
            ranked_results,
            verification_result
        )

        attempt_details.append(
            {
                "attempt": attempt,
                "query": current_query,
                "retrieved": retrieved_count,
                "reranked": reranked_count,
                "verification": verification_result,
                "answer": answer,
                "context": context,
            }
        )

        if verification_result == "SUPPORTED":

            final_answer = answer
            break

    if final_answer is None:

        final_answer = (
            "I don't know based on the provided documents."
        )

        verification_result = "NOT_SUPPORTED"
        confidence_percentage = 0
        confidence_level = "LOW"

    response_time = round(
        time.time() - start_time,
        2
    )

    return {
        "question": question,
        "answer": final_answer,
        "verification": verification_result,
        "confidence": confidence_percentage,
        "confidence_level": confidence_level,
        "attempts": attempts,
        "retrieved": retrieved_count,
        "reranked": reranked_count,
        "queries": rewritten_queries,
        "response_time": response_time,
        "attempt_details": attempt_details,
    }


def evaluate_rag_test_result(test_type, result):

    if test_type == "Supported":
        return result["verification"] == "SUPPORTED"

    if test_type == "Unsupported":
        return result["verification"] == "NOT_SUPPORTED"

    return False


def save_evaluation_results(results):

    try:

        with open(
            EVALUATION_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        print(
            "Evaluation save error:",
            error
        )


# ============================================================
# EVALUATION DASHBOARD
# ============================================================

st.divider()

st.subheader(
    "🧪 Automatic RAG Evaluation"
)

st.write(
    "Run predefined supported and unsupported questions "
    "through the actual RAG pipeline."
)

if st.button(
    "🚀 Run Automatic Evaluation",
    use_container_width=True
):

    if collection.count() == 0:

        st.warning(
            "Please upload at least one PDF before running evaluation."
        )

    else:

        evaluation_results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_questions = len(
            DEFAULT_EVALUATION_QUESTIONS
        )

        for index, test_case in enumerate(
            DEFAULT_EVALUATION_QUESTIONS
        ):

            question_text = test_case["question"]

            status_text.write(
                f"🧪 Testing {index + 1}/{total_questions}: "
                f"{question_text}"
            )

            result = run_rag_for_evaluation(
                question_text
            )

            passed = evaluate_rag_test_result(
                test_case["type"],
                result
            )

            result["test_type"] = test_case["type"]
            result["passed"] = passed

            evaluation_results.append(
                result
            )

            progress_bar.progress(
                (index + 1) / total_questions
            )

        status_text.empty()

        save_evaluation_results(
            evaluation_results
        )

        passed_count = sum(
            1
            for result in evaluation_results
            if result["passed"]
        )

        failed_count = (
            total_questions -
            passed_count
        )

        accuracy = round(
            (passed_count / total_questions) * 100,
            2
        )

        supported_tests = [
            result
            for result in evaluation_results
            if result["test_type"] == "Supported"
        ]

        unsupported_tests = [
            result
            for result in evaluation_results
            if result["test_type"] == "Unsupported"
        ]

        supported_success = sum(
            1
            for result in supported_tests
            if result["verification"] == "SUPPORTED"
        )

        unsupported_success = sum(
            1
            for result in unsupported_tests
            if result["verification"] == "NOT_SUPPORTED"
        )

        average_response_time = round(
            sum(
                result["response_time"]
                for result in evaluation_results
            ) / total_questions,
            2
        )

        average_attempts = round(
            sum(
                result["attempts"]
                for result in evaluation_results
            ) / total_questions,
            2
        )

        save_evaluation_history(
            evaluation_results=evaluation_results,
            accuracy=accuracy,
            passed_count=passed_count,
            failed_count=failed_count,
            average_attempts=average_attempts,
            average_response_time=average_response_time,
        )

        st.success(
            "Automatic evaluation completed."
        )

        st.markdown(
            "### 📊 Evaluation Results"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Tests",
                total_questions
            )

        with col2:
            st.metric(
                "Passed",
                passed_count
            )

        with col3:
            st.metric(
                "Failed",
                failed_count
            )

        with col4:
            st.metric(
                "Accuracy",
                f"{accuracy}%"
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "Supported Success",
                f"{supported_success}/{len(supported_tests)}"
            )

        with col6:
            st.metric(
                "Unsupported Detection",
                f"{unsupported_success}/{len(unsupported_tests)}"
            )

        with col7:
            st.metric(
                "Avg. Attempts",
                average_attempts
            )

        with col8:
            st.metric(
                "Avg. Response",
                f"{average_response_time}s"
            )

        st.markdown(
            "### 📋 Question-wise Evaluation"
        )

        for index, result in enumerate(
            evaluation_results,
            start=1
        ):

            icon = "✅" if result["passed"] else "❌"

            with st.expander(
                f"{icon} Q{index}: {result['question']}"
            ):

                st.write(
                    f"**Test type:** {result['test_type']}"
                )

                st.write(
                    f"**Verification:** {result['verification']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{result['confidence']}% "
                    f"({result['confidence_level']})"
                )

                st.write(
                    f"**Attempts:** {result['attempts']}"
                )

                st.write(
                    f"**Retrieved:** {result['retrieved']}"
                )

                st.write(
                    f"**Reranked:** {result['reranked']}"
                )

                st.write(
                    f"**Response time:** "
                    f"{result['response_time']} sec"
                )

                st.markdown(
                    "**Generated Answer**"
                )

                st.info(
                    result["answer"]
                )

                st.markdown(
                    "**Search Queries Used**"
                )

                for query_index, query_text in enumerate(
                    result["queries"],
                    start=1
                ):

                    st.code(
                        f"Attempt {query_index}: {query_text}"
                    )

        st.markdown(
            "### 🏁 Overall Evaluation"
        )

        if accuracy >= 80:

            st.success(
                f"🟢 GOOD — Overall evaluation accuracy: {accuracy}%"
            )

        elif accuracy >= 60:

            st.warning(
                f"🟡 MODERATE — Overall evaluation accuracy: {accuracy}%"
            )

        else:

            st.error(
                f"🔴 NEEDS IMPROVEMENT — Overall evaluation accuracy: {accuracy}%"
            )



# ============================================================
# EVALUATION HISTORY DASHBOARD
# ============================================================

evaluation_history = load_evaluation_history()

if evaluation_history:

    st.divider()

    st.subheader(
        "📈 Evaluation History"
    )

    total_runs = len(evaluation_history)
    latest_run = evaluation_history[-1]
    best_accuracy = max(
        float(run.get("accuracy", 0))
        for run in evaluation_history
    )
    average_accuracy = round(
        sum(float(run.get("accuracy", 0)) for run in evaluation_history)
        / total_runs,
        2,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🧪 Total Runs", total_runs)

    with col2:
        st.metric("🎯 Latest Accuracy", f"{latest_run.get('accuracy', 0)}%")

    with col3:
        st.metric("🏆 Best Accuracy", f"{best_accuracy}%")

    with col4:
        st.metric("📊 Average Accuracy", f"{average_accuracy}%")

    st.markdown("### 📈 Accuracy Across Evaluation Runs")
    st.line_chart({
        "Accuracy (%)": [
            float(run.get("accuracy", 0))
            for run in evaluation_history
        ]
    })

    st.markdown("### ⏱️ Average Response Time")
    st.line_chart({
        "Response Time (seconds)": [
            float(run.get("average_response_time", 0))
            for run in evaluation_history
        ]
    })

    st.markdown("### 🔄 Average Attempts")
    st.line_chart({
        "Average Attempts": [
            float(run.get("average_attempts", 0))
            for run in evaluation_history
        ]
    })

    st.markdown("### 📋 Previous Evaluation Runs")

    for run in reversed(evaluation_history):
        with st.expander(
            f"🧪 Run {run.get('run', '?')} — "
            f"{run.get('timestamp', '')} — "
            f"{run.get('accuracy', 0)}%"
        ):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Tests", run.get("total_tests", 0))

            with c2:
                st.metric("Passed", run.get("passed", 0))

            with c3:
                st.metric("Failed", run.get("failed", 0))

            with c4:
                st.metric("Accuracy", f"{run.get('accuracy', 0)}%")

            st.write(
                f"**Average Attempts:** {run.get('average_attempts', 0)}"
            )
            st.write(
                f"**Average Response:** "
                f"{run.get('average_response_time', 0)} seconds"
            )

else:

    st.info(
        "Run Automatic RAG Evaluation to create evaluation history."
    )


# ============================================================
# STORAGE INFORMATION
# ============================================================

with st.expander(
    "💾 Storage Information"
):

    st.write(
        "Chat history:"
    )

    st.code(
        "data/chat_history.json"
    )

    st.write(
        "Document registry:"
    )

    st.code(
        "data/documents.json"
    )

    st.write(
        "Vector database:"
    )

    st.code(
        "data/chroma_db/"
    )

    st.write(
        "Stored PDFs:"
    )

    st.code(
        "data/documents/"
    )

    st.write(
        f"Messages stored: "
        f"{len(st.session_state.messages)}"
    )

# ============================================================
# RAG PERFORMANCE ANALYTICS
# ============================================================

render_rag_performance()