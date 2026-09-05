import streamlit as st
import requests
import time

# ==============================
# CONFIGURATION
# ==============================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🤖",
    layout="wide"
)


# ==============================
# API CONNECTION
# ==============================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def ask_api(question):
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=180
        )

        if response.status_code == 200:
            return response.json()

        return {
            "success": False,
            "answer": f"API returned status code {response.status_code}"
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "answer": f"Could not connect to API: {e}"
        }


def upload_pdf(uploaded_file):
    """
    Send PDF to FastAPI backend.
    """

    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=300
        )

        if response.status_code == 200:
            return response.json()

        return {
            "success": False,
            "message": f"Upload failed. API status: {response.status_code}"
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Upload error: {e}"
        }


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.title("📚 Knowledge Base")

    # --------------------------------
    # API STATUS
    # --------------------------------

    api_online = check_api()

    if api_online:
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")

    st.divider()

    # --------------------------------
    # PDF UPLOAD
    # --------------------------------

    st.subheader("📤 Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "⬆️ Add to Knowledge Base",
            use_container_width=True
        ):

            if not api_online:
                st.error(
                    "FastAPI is offline. Start the API first."
                )

            else:

                for uploaded_file in uploaded_files:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        result = upload_pdf(uploaded_file)

                    if result.get("success"):

                        st.success(
                            f"✅ {uploaded_file.name} added!"
                        )

                    else:

                        st.error(
                            result.get(
                                "message",
                                f"Failed to upload {uploaded_file.name}"
                            )
                        )

    st.divider()

    # --------------------------------
    # KNOWLEDGE BASE
    # --------------------------------

    st.subheader("📖 Knowledge Base")

    st.write(
        "Upload PDF documents above to add them "
        "to the RAG knowledge base."
    )

    st.divider()

    # --------------------------------
    # RAG PIPELINE
    # --------------------------------

    st.subheader("⚙️ RAG Pipeline")

    st.write("🔄 Query Rewriting")
    st.write("🔎 Vector Search")
    st.write("📊 Reranking")
    st.write("🧠 LLM Generation")
    st.write("🛡️ Answer Verification")
    st.write("🔧 Self-Correction")
    st.write("📚 Source Evidence")

    st.divider()

    # --------------------------------
    # API
    # --------------------------------

    st.subheader("🔗 API")

    st.code(API_URL)


# ==============================
# MAIN PAGE
# ==============================

st.title("🤖 Self-Correcting RAG System")

st.write(
    "Ask questions about your PDF knowledge base "
    "using Retrieval-Augmented Generation."
)

st.divider()


# ==============================
# ASK QUESTION
# ==============================

st.header("💬 Ask a Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is Retrieval-Augmented Generation?",
    height=100
)


if st.button(
    "🔍 Ask AI",
    use_container_width=True,
    type="primary"
):

    if not question.strip():

        st.warning("Please enter a question.")

    elif not check_api():

        st.error(
            "FastAPI is not reachable. "
            "Make sure the API server is running on port 8000."
        )

    else:

        with st.spinner(
            "🔄 Searching documents and generating answer..."
        ):

            result = ask_api(question)

        if result.get("success"):

            st.success(
                "✅ Answer generated successfully!"
            )

            st.subheader("🤖 AI Response")

            st.write(
                result.get(
                    "answer",
                    "No answer returned."
                )
            )

            # ==============================
            # RAG PERFORMANCE
            # ==============================

            st.divider()

            st.subheader("📊 RAG Performance")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Retrieved Chunks",
                    result.get(
                        "retrieved_chunks",
                        0
                    )
                )

            with col2:

                st.metric(
                    "Reranked Chunks",
                    result.get(
                        "reranked_chunks",
                        0
                    )
                )

            with col3:

                st.metric(
                    "Attempts",
                    result.get(
                        "attempts",
                        1
                    )
                )

            # ==============================
            # VERIFICATION
            # ==============================

            verification = result.get(
                "verification",
                result.get(
                    "verification_status",
                    "SUPPORTED"
                )
            )

            st.subheader("🛡️ Verification")

            if isinstance(
                verification,
                dict
            ):

                supported = verification.get(
                    "supported",
                    False
                )

                confidence = verification.get(
                    "confidence",
                    0
                )

                reason = verification.get(
                    "reason",
                    ""
                )

                if supported:

                    st.success(
                        f"✅ Supported "
                        f"(Confidence: {confidence:.0%})"
                    )

                else:

                    st.warning(
                        f"⚠️ Verification failed "
                        f"(Confidence: {confidence:.0%})"
                    )

                if reason:
                    st.caption(
                        f"Reason: {reason}"
                    )

            else:

                if str(verification).upper() in [
                    "SUPPORTED",
                    "TRUE",
                    "PASS",
                    "VERIFIED"
                ]:

                    st.success(
                        "✅ Answer Supported"
                    )

                else:

                    st.warning(
                        f"⚠️ Verification: {verification}"
                    )

            # ==============================
            # SOURCES
            # ==============================

            sources = result.get(
                "sources",
                []
            )

            if sources:

                st.divider()

                st.subheader(
                    "📚 Source Evidence"
                )

                for i, source in enumerate(
                    sources,
                    1
                ):

                    with st.expander(
                        f"📄 Source {i}"
                    ):

                        if isinstance(
                            source,
                            dict
                        ):

                            st.write(
                                f"**Document:** "
                                f"{source.get('source', 'Unknown')}"
                            )

                            if "page" in source:

                                st.write(
                                    f"**Page:** "
                                    f"{source.get('page')}"
                                )

                            if "score" in source:

                                st.write(
                                    f"**Score:** "
                                    f"{source.get('score')}"
                                )

                            if "rerank_score" in source:

                                st.write(
                                    f"**Rerank Score:** "
                                    f"{source.get('rerank_score')}"
                                )

                            if "text" in source:

                                st.write(
                                    source.get("text")
                                )

                        else:

                            st.write(source)

        else:

            st.error(
                result.get(
                    "answer",
                    result.get(
                        "message",
                        "Something went wrong."
                    )
                )
            )


# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "Self-Correcting RAG • FastAPI + ChromaDB + "
    "Sentence Transformers + Groq"
)