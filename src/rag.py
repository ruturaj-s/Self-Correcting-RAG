"""
RAG Pipeline

Complete Retrieval-Augmented Generation pipeline with:

1. Query rewriting
2. Vector retrieval
3. Cross-encoder reranking
4. LLM answer generation
5. Answer verification
6. Self-correction
7. Retry mechanism
8. Source evidence
9. Confidence scoring
"""

import time
from typing import Any

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.self_correction import self_correct_query


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 2

RETRIEVAL_K = 8

RERANK_K = 4


# ============================================================
# MODEL INITIALIZATION
# ============================================================

_embedding_model = None

_reranker_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _embedding_model


def get_reranker_model():

    global _reranker_model

    if _reranker_model is None:

        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker_model


# ============================================================
# SAFE IMPORT HELPERS
# ============================================================

def _load_query_rewriter():

    try:

        from src.query_rewriter import rewrite_query

        return rewrite_query

    except Exception:

        return None


def _load_retriever():

    try:

        from src.retriever import retrieve_documents

        return retrieve_documents

    except Exception:

        return None


def _load_reranker():

    try:

        from src.reranker import rerank_documents

        return rerank_documents

    except Exception:

        return None


def _load_verifier():

    try:

        from src.verifier import verify_answer

        return verify_answer

    except Exception:

        return None


# ============================================================
# LLM
# ============================================================

def get_llm():

    """
    Load the LLM used by the project.

    Uses the existing LLM object from the project
    whenever available.
    """

    try:

        from src.rag import llm

        return llm

    except Exception:

        pass

    return None


# ============================================================
# TEXT EXTRACTION
# ============================================================

def _extract_text(document: Any) -> str:

    if document is None:
        return ""

    if isinstance(document, str):
        return document

    if isinstance(document, dict):

        for key in [
            "page_content",
            "content",
            "text",
            "document",
        ]:

            if key in document:

                value = document[key]

                if value is not None:
                    return str(value)

        return str(document)

    if hasattr(
        document,
        "page_content"
    ):

        return str(
            document.page_content
        )

    return str(document)


# ============================================================
# EXTRACT METADATA
# ============================================================

def _extract_metadata(document: Any) -> dict:

    if isinstance(document, dict):

        metadata = document.get(
            "metadata",
            {}
        )

        if isinstance(metadata, dict):
            return metadata

    if hasattr(
        document,
        "metadata"
    ):

        metadata = document.metadata

        if isinstance(metadata, dict):
            return metadata

    return {}


# ============================================================
# GENERATE ANSWER
# ============================================================

def _generate_answer(
    question: str,
    context: str,
    llm: Any,
) -> str:

    if llm is None:

        return (
            "I don't know based on the provided documents."
        )

    prompt = f"""
You are a document-grounded RAG assistant.

Answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the context,
   say exactly:

I don't know based on the provided documents.

4. Give a clear and concise answer.
5. Use information from the provided context only.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
"""

    try:

        response = llm.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict "
                        "document-grounded assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        return answer

    except Exception as error:

        print(
            "Answer generation error:",
            error
        )

        return (
            "I don't know based on the provided documents."
        )


# ============================================================
# BUILD CONTEXT
# ============================================================

def _build_context(
    documents: list[Any]
) -> str:

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        text = _extract_text(
            document
        )

        if not text.strip():
            continue

        context_parts.append(
            f"""
--- DOCUMENT CHUNK {index} ---

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# FALLBACK RETRIEVAL
# ============================================================

def _fallback_retrieve(
    query: str,
    collection: Any,
    embedding_model: Any,
    top_k: int,
):

    if collection is None:
        return []

    try:

        query_embedding = (
            embedding_model
            .encode(query)
            .tolist()
        )

        result = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
        )

        documents = result.get(
            "documents",
            [[]]
        )

        metadatas = result.get(
            "metadatas",
            [[]]
        )

        if not documents:
            return []

        documents = documents[0]

        if metadatas:
            metadatas = metadatas[0]
        else:
            metadatas = []

        output = []

        for index, text in enumerate(
            documents
        ):

            metadata = {}

            if index < len(metadatas):

                if isinstance(
                    metadatas[index],
                    dict
                ):

                    metadata = metadatas[index]

            output.append(
                {
                    "page_content": text,
                    "metadata": metadata,
                }
            )

        return output

    except Exception as error:

        print(
            "Fallback retrieval error:",
            error
        )

        return []


# ============================================================
# FALLBACK RERANKING
# ============================================================

def _fallback_rerank(
    query: str,
    documents: list[Any],
    top_k: int,
):

    if not documents:
        return []

    try:

        model = get_reranker_model()

        pairs = []

        for document in documents:

            text = _extract_text(
                document
            )

            pairs.append(
                [
                    query,
                    text,
                ]
            )

        scores = model.predict(
            pairs
        )

        ranked = sorted(
            zip(
                documents,
                scores
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        output = []

        for document, score in ranked[
            :top_k
        ]:

            if isinstance(
                document,
                dict
            ):

                item = dict(
                    document
                )

            else:

                item = {
                    "page_content":
                        _extract_text(
                            document
                        ),
                    "metadata":
                        _extract_metadata(
                            document
                        ),
                }

            item["rerank_score"] = float(
                score
            )

            output.append(
                item
            )

        return output

    except Exception as error:

        print(
            "Fallback reranking error:",
            error
        )

        return documents[
            :top_k
        ]


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def _retrieve(
    query: str,
    collection: Any,
    top_k: int,
):

    retriever = _load_retriever()

    if retriever is not None:

        try:

            result = retriever(
                query,
                collection,
                top_k=top_k,
            )

            if result is not None:

                return result

        except TypeError:

            try:

                result = retriever(
                    query,
                    collection,
                )

                if result is not None:
                    return result

            except Exception:
                pass

        except Exception:
            pass

    return _fallback_retrieve(
        query,
        collection,
        get_embedding_model(),
        top_k,
    )


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def _rerank(
    query: str,
    documents: list[Any],
    top_k: int,
):

    if not documents:
        return []

    reranker = _load_reranker()

    if reranker is not None:

        try:

            result = reranker(
                query,
                documents,
                top_k=top_k,
            )

            if result is not None:
                return result

        except TypeError:

            try:

                result = reranker(
                    query,
                    documents,
                )

                if result is not None:

                    return result[
                        :top_k
                    ]

            except Exception:
                pass

        except Exception:
            pass

    return _fallback_rerank(
        query,
        documents,
        top_k,
    )


# ============================================================
# VERIFY ANSWER
# ============================================================

def _verify_answer(
    question: str,
    answer: str,
    context: str,
):

    verifier = _load_verifier()

    if verifier is None:

        return "NOT_SUPPORTED"

    try:

        result = verifier(
            question,
            answer,
            context,
        )

        if isinstance(
            result,
            str
        ):

            return result.upper()

        if isinstance(
            result,
            dict
        ):

            return str(
                result.get(
                    "verification",
                    result.get(
                        "status",
                        "NOT_SUPPORTED",
                    ),
                )
            ).upper()

        if isinstance(
            result,
            bool
        ):

            return (
                "SUPPORTED"
                if result
                else "NOT_SUPPORTED"
            )

    except Exception as error:

        print(
            "Verification error:",
            error
        )

    return "NOT_SUPPORTED"


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_rag(
    question: str,
    collection: Any,
    status_box: Any = None,
) -> dict[str, Any]:

    start_time = time.time()

    if not question.strip():

        return {
            "answer": "",
            "verification": "NOT_SUPPORTED",
            "confidence": 0,
            "attempts": 0,
            "retrieved": 0,
            "reranked": 0,
            "sources": [],
            "response_time": 0,
            "queries": [],
            "corrected": False,
        }

    llm = get_llm()

    current_query = question

    queries_used = []

    attempt_details = []

    final_answer = ""

    final_sources = []

    final_verification = "NOT_SUPPORTED"

    total_retrieved = 0

    total_reranked = 0

    corrected = False

    # ========================================================
    # RETRY LOOP
    # ========================================================

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        if status_box is not None:

            status_box.write(
                f"🔄 Attempt {attempt}/{MAX_RETRIES}"
            )

        # ----------------------------------------------------
        # QUERY REWRITING
        # ----------------------------------------------------

        if attempt == 1:

            rewritten_query = current_query

        else:

            corrected = True

            try:

                corrected_query = self_correct_query(
                    question=question,
                    previous_query=current_query,
                    answer=final_answer,
                    context=_build_context(
                        final_sources
                    ),
                    llm=llm,
                )

                if (
                    corrected_query
                    and
                    corrected_query.strip()
                ):

                    rewritten_query = (
                        corrected_query.strip()
                    )

                else:

                    rewritten_query = current_query

            except Exception as error:

                print(
                    "Self-correction error:",
                    error
                )

                rewritten_query = current_query

        current_query = rewritten_query

        queries_used.append(
            current_query
        )

        if status_box is not None:

            status_box.write(
                f"🔎 Searching: `{current_query}`"
            )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieved_documents = _retrieve(
            current_query,
            collection,
            RETRIEVAL_K,
        )

        if retrieved_documents is None:

            retrieved_documents = []

        retrieved_count = len(
            retrieved_documents
        )

        total_retrieved += (
            retrieved_count
        )

        if status_box is not None:

            status_box.write(
                f"📚 Retrieved: "
                f"{retrieved_count} chunks"
            )

        # ----------------------------------------------------
        # RERANKING
        # ----------------------------------------------------

        reranked_documents = _rerank(
            current_query,
            retrieved_documents,
            RERANK_K,
        )

        if reranked_documents is None:

            reranked_documents = []

        reranked_count = len(
            reranked_documents
        )

        total_reranked += (
            reranked_count
        )

        if status_box is not None:

            status_box.write(
                f"📊 Reranked: "
                f"{reranked_count} chunks"
            )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context = _build_context(
            reranked_documents
        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        if status_box is not None:

            status_box.write(
                "🤖 Generating answer..."
            )

        answer = _generate_answer(
            question,
            context,
            llm,
        )

        final_answer = answer

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        if status_box is not None:

            status_box.write(
                "🛡️ Verifying answer..."
            )

        verification = _verify_answer(
            question,
            answer,
            context,
        )

        final_verification = (
            verification
        )

        # ----------------------------------------------------
        # SAVE ATTEMPT DETAILS
        # ----------------------------------------------------

        attempt_details.append(
            {
                "attempt": attempt,
                "query": current_query,
                "retrieved": retrieved_count,
                "reranked": reranked_count,
                "verification": verification,
            }
        )

        # ----------------------------------------------------
        # VERIFIED
        # ----------------------------------------------------

        if verification == "SUPPORTED":

            if status_box is not None:

                status_box.success(
                    "✅ Answer verified."
                )

            final_sources = (
                reranked_documents
            )

            break

        # ----------------------------------------------------
        # NOT SUPPORTED
        # ----------------------------------------------------

        if status_box is not None:

            status_box.warning(
                "⚠️ Answer not supported."
            )

        # ----------------------------------------------------
        # SELF CORRECTION
        # ----------------------------------------------------

        if attempt < MAX_RETRIES:

            if status_box is not None:

                status_box.write(
                    "🔁 Self-correcting "
                    "retrieval query..."
                )

            try:

                corrected_query = (
                    self_correct_query(
                        question=question,
                        previous_query=current_query,
                        answer=answer,
                        context=context,
                        llm=llm,
                    )
                )

                if (
                    corrected_query
                    and
                    corrected_query.strip()
                    and
                    corrected_query.strip().lower()
                    != current_query.strip().lower()
                ):

                    current_query = (
                        corrected_query.strip()
                    )

                    if status_box is not None:

                        status_box.write(
                            "🧠 Corrected query:"
                        )

                        status_box.code(
                            current_query
                        )

                else:

                    if status_box is not None:

                        status_box.write(
                            "ℹ️ No better query "
                            "was generated."
                        )

            except Exception as error:

                print(
                    "Self-correction error:",
                    error
                )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if final_verification == "SUPPORTED":

        confidence = 95

        confidence_level = "HIGH"

    elif total_reranked > 0:

        confidence = 30

        confidence_level = "LOW"

    else:

        confidence = 0

        confidence_level = "LOW"

    # ========================================================
    # RESPONSE TIME
    # ========================================================

    response_time = round(
        time.time() - start_time,
        2
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": final_answer,

        "verification": final_verification,

        "confidence": confidence,

        "confidence_level":
            confidence_level,

        "attempts": len(
            attempt_details
        ),

        "retrieved": (
            total_retrieved /
            max(
                len(attempt_details),
                1
            )
        ),

        "reranked": (
            total_reranked /
            max(
                len(attempt_details),
                1
            )
        ),

        "sources": final_sources,

        "response_time":
            response_time,

        "queries":
            queries_used,

        "attempt_details":
            attempt_details,

        "corrected":
            corrected,
    }