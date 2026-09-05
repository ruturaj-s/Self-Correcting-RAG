"""
Automatic RAG Evaluation

Runs predefined supported and unsupported questions
through the actual RAG pipeline and stores evaluation
history with complete RAG performance metrics.
"""

import json
import os
from datetime import datetime
from typing import Any, Callable

import streamlit as st


EVALUATION_FILE = "data/evaluation_results.json"
HISTORY_FILE = "data/evaluation_history.json"


# ============================================================
# FILE HELPERS
# ============================================================

def _ensure_data_directory():
    os.makedirs("data", exist_ok=True)


def load_evaluation_history() -> list[dict[str, Any]]:
    _ensure_data_directory()

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        return []

    except Exception as error:
        print("Evaluation history load error:", error)
        return []


def save_evaluation_results(results: list[dict[str, Any]]):
    _ensure_data_directory()

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
        print("Evaluation results save error:", error)


def save_evaluation_history(
    evaluation_results: list[dict[str, Any]],
    accuracy: float,
    passed_count: int,
    failed_count: int,
    average_attempts: float,
    average_response_time: float,
    average_retrieved: float,
    average_reranked: float,
    verification_success: float,
):
    """
    Save a complete evaluation run.

    IMPORTANT:
    Retrieval, reranking and verification metrics are
    stored here so performance.py can display them.
    """

    _ensure_data_directory()

    history = load_evaluation_history()

    supported_tests = [
        result
        for result in evaluation_results
        if result.get("test_type") == "Supported"
    ]

    unsupported_tests = [
        result
        for result in evaluation_results
        if result.get("test_type") == "Unsupported"
    ]

    supported_success = sum(
        1
        for result in supported_tests
        if result.get("verification") == "SUPPORTED"
    )

    unsupported_success = sum(
        1
        for result in unsupported_tests
        if result.get("verification") == "NOT_SUPPORTED"
    )

    history.append(
        {
            "run": len(history) + 1,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "total_tests": len(evaluation_results),
            "passed": passed_count,
            "failed": failed_count,

            "accuracy": accuracy,

            "supported_success": supported_success,
            "supported_total": len(supported_tests),

            "unsupported_success": unsupported_success,
            "unsupported_total": len(unsupported_tests),

            "average_attempts": average_attempts,
            "average_response_time": average_response_time,

            # IMPORTANT PERFORMANCE METRICS
            "average_retrieved": average_retrieved,
            "average_reranked": average_reranked,
            "verification_success": verification_success,
        }
    )

    try:
        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:
        print("Evaluation history save error:", error)


# ============================================================
# DEFAULT TEST QUESTIONS
# ============================================================

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


# ============================================================
# TEST RESULT
# ============================================================

def evaluate_test_result(
    test_type: str,
    result: dict[str, Any]
) -> bool:

    verification = result.get(
        "verification",
        "NOT_SUPPORTED"
    )

    if test_type == "Supported":
        return verification == "SUPPORTED"

    if test_type == "Unsupported":
        return verification == "NOT_SUPPORTED"

    return False


# ============================================================
# AUTOMATIC EVALUATION UI
# ============================================================

def render_automatic_evaluation(
    run_rag_for_evaluation: Callable[[str], dict[str, Any]],
    collection_count: int,
):
    """
    Render the complete automatic evaluation section.

    run_rag_for_evaluation must be the same RAG pipeline
    used by the main chatbot.
    """

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

        if collection_count == 0:

            st.warning(
                "Please upload at least one PDF before "
                "running evaluation."
            )

            return

        evaluation_results = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_questions = len(
            DEFAULT_EVALUATION_QUESTIONS
        )

        # ====================================================
        # RUN TESTS
        # ====================================================

        for index, test_case in enumerate(
            DEFAULT_EVALUATION_QUESTIONS
        ):

            question = test_case["question"]

            status_text.write(
                f"🧪 Testing "
                f"{index + 1}/{total_questions}: "
                f"{question}"
            )

            try:

                result = run_rag_for_evaluation(
                    question
                )

            except Exception as error:

                result = {
                    "question": question,
                    "answer": (
                        "Evaluation error."
                    ),
                    "verification": "NOT_SUPPORTED",
                    "confidence": 0,
                    "confidence_level": "LOW",
                    "attempts": 0,
                    "retrieved": 0,
                    "reranked": 0,
                    "queries": [],
                    "response_time": 0,
                    "attempt_details": [],
                    "error": str(error),
                }

            passed = evaluate_test_result(
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

        # ====================================================
        # SAVE RAW RESULTS
        # ====================================================

        save_evaluation_results(
            evaluation_results
        )

        # ====================================================
        # BASIC METRICS
        # ====================================================

        passed_count = sum(
            1
            for result in evaluation_results
            if result.get("passed")
        )

        failed_count = (
            total_questions -
            passed_count
        )

        accuracy = round(
            (
                passed_count /
                total_questions
            ) * 100,
            2
        )

        # ====================================================
        # SUPPORTED / UNSUPPORTED
        # ====================================================

        supported_tests = [
            result
            for result in evaluation_results
            if result.get("test_type") == "Supported"
        ]

        unsupported_tests = [
            result
            for result in evaluation_results
            if result.get("test_type") == "Unsupported"
        ]

        supported_success = sum(
            1
            for result in supported_tests
            if result.get("verification")
            == "SUPPORTED"
        )

        unsupported_success = sum(
            1
            for result in unsupported_tests
            if result.get("verification")
            == "NOT_SUPPORTED"
        )

        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        average_response_time = round(
            sum(
                float(
                    result.get(
                        "response_time",
                        0
                    )
                )
                for result in evaluation_results
            )
            / total_questions,
            2
        )

        average_attempts = round(
            sum(
                float(
                    result.get(
                        "attempts",
                        0
                    )
                )
                for result in evaluation_results
            )
            / total_questions,
            2
        )

        average_retrieved = round(
            sum(
                float(
                    result.get(
                        "retrieved",
                        0
                    )
                )
                for result in evaluation_results
            )
            / total_questions,
            2
        )

        average_reranked = round(
            sum(
                float(
                    result.get(
                        "reranked",
                        0
                    )
                )
                for result in evaluation_results
            )
            / total_questions,
            2
        )

        # ====================================================
        # VERIFICATION SUCCESS
        # ====================================================
        #
        # Verification is successful when the verifier
        # correctly handles BOTH:
        #
        # Supported question:
        #     verification == "SUPPORTED"
        #
        # Unsupported question:
        #     verification == "NOT_SUPPORTED"
        #
        # Therefore, both correct classifications count
        # toward verification success.
        # ====================================================

        verification_correct = sum(
            1
            for result in evaluation_results
            if (
                (
                    result.get("test_type") == "Supported"
                    and result.get("verification")
                    == "SUPPORTED"
                )
                or
                (
                    result.get("test_type") == "Unsupported"
                    and result.get("verification")
                    == "NOT_SUPPORTED"
                )
            )
        )

        verification_success = round(
            (
                verification_correct /
                total_questions
            )
            * 100,
            2
        )

        # ====================================================
        # SAVE COMPLETE HISTORY
        # ====================================================

        save_evaluation_history(
            evaluation_results=evaluation_results,
            accuracy=accuracy,
            passed_count=passed_count,
            failed_count=failed_count,
            average_attempts=average_attempts,
            average_response_time=average_response_time,
            average_retrieved=average_retrieved,
            average_reranked=average_reranked,
            verification_success=verification_success,
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "✅ Automatic evaluation completed."
        )

        # ====================================================
        # SUMMARY
        # ====================================================

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
                f"{supported_success}/"
                f"{len(supported_tests)}"
            )

        with col6:
            st.metric(
                "Unsupported Detection",
                f"{unsupported_success}/"
                f"{len(unsupported_tests)}"
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

        # ====================================================
        # PIPELINE METRICS
        # ====================================================

        st.markdown(
            "### 🔬 RAG Pipeline Metrics"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Average Retrieved",
                f"{average_retrieved:.2f}"
            )

        with c2:
            st.metric(
                "Average Reranked",
                f"{average_reranked:.2f}"
            )

        with c3:
            st.metric(
                "Verification Success",
                f"{verification_success:.1f}%"
            )

        # ====================================================
        # QUESTION-WISE RESULTS
        # ====================================================

        st.markdown(
            "### 📋 Question-wise Evaluation"
        )

        for index, result in enumerate(
            evaluation_results,
            start=1
        ):

            icon = (
                "✅"
                if result.get("passed")
                else "❌"
            )

            with st.expander(
                f"{icon} Q{index}: "
                f"{result.get('question', '')}"
            ):

                st.write(
                    f"**Test type:** "
                    f"{result.get('test_type', '')}"
                )

                st.write(
                    f"**Verification:** "
                    f"{result.get('verification', '')}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{result.get('confidence', 0)}% "
                    f"({result.get('confidence_level', 'LOW')})"
                )

                st.write(
                    f"**Attempts:** "
                    f"{result.get('attempts', 0)}"
                )

                st.write(
                    f"**Retrieved:** "
                    f"{result.get('retrieved', 0)}"
                )

                st.write(
                    f"**Reranked:** "
                    f"{result.get('reranked', 0)}"
                )

                st.write(
                    f"**Response time:** "
                    f"{result.get('response_time', 0)} sec"
                )

                st.markdown(
                    "**Generated Answer**"
                )

                st.info(
                    result.get(
                        "answer",
                        ""
                    )
                )

                st.markdown(
                    "**Search Queries Used**"
                )

                queries = result.get(
                    "queries",
                    []
                )

                for query_index, query in enumerate(
                    queries,
                    start=1
                ):

                    st.code(
                        f"Attempt {query_index}: "
                        f"{query}"
                    )

        # ====================================================
        # OVERALL EVALUATION
        # ====================================================

        st.markdown(
            "### 🏁 Overall Evaluation"
        )

        if accuracy >= 80:

            st.success(
                f"🟢 GOOD — "
                f"Overall evaluation accuracy: "
                f"{accuracy}%"
            )

        elif accuracy >= 60:

            st.warning(
                f"🟡 MODERATE — "
                f"Overall evaluation accuracy: "
                f"{accuracy}%"
            )

        else:

            st.error(
                f"🔴 NEEDS IMPROVEMENT — "
                f"Overall evaluation accuracy: "
                f"{accuracy}%"
            )


# ============================================================
# EVALUATION HISTORY DASHBOARD
# ============================================================

def render_evaluation_history():

    history = load_evaluation_history()

    if not history:

        st.info(
            "Run Automatic RAG Evaluation "
            "to create evaluation history."
        )

        return

    st.divider()

    st.subheader(
        "📈 Evaluation History"
    )

    total_runs = len(history)

    latest = history[-1]

    accuracies = [
        float(
            item.get(
                "accuracy",
                0
            )
        )
        for item in history
    ]

    best_accuracy = max(
        accuracies
    )

    average_accuracy = round(
        sum(accuracies) /
        len(accuracies),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🧪 Total Runs",
            total_runs
        )

    with c2:
        st.metric(
            "🎯 Latest Accuracy",
            f"{latest.get('accuracy', 0)}%"
        )

    with c3:
        st.metric(
            "🏆 Best Accuracy",
            f"{best_accuracy}%"
        )

    with c4:
        st.metric(
            "📊 Average Accuracy",
            f"{average_accuracy}%"
        )

    # ========================================================
    # ACCURACY
    # ========================================================

    st.markdown(
        "### 📈 Accuracy Across Evaluation Runs"
    )

    st.line_chart(
        {
            "Accuracy (%)": accuracies
        }
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    st.markdown(
        "### ⏱️ Average Response Time"
    )

    response_times = [
        float(
            item.get(
                "average_response_time",
                0
            )
        )
        for item in history
    ]

    st.line_chart(
        {
            "Response Time (seconds)": response_times
        }
    )

    # ========================================================
    # ATTEMPTS
    # ========================================================

    st.markdown(
        "### 🔄 Average Attempts"
    )

    attempts = [
        float(
            item.get(
                "average_attempts",
                0
            )
        )
        for item in history
    ]

    st.line_chart(
        {
            "Average Attempts": attempts
        }
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    st.markdown(
        "### 🔎 Average Retrieved Chunks"
    )

    retrieved = [
        float(
            item.get(
                "average_retrieved",
                0
            )
        )
        for item in history
    ]

    st.line_chart(
        {
            "Retrieved Chunks": retrieved
        }
    )

    # ========================================================
    # RERANKING
    # ========================================================

    st.markdown(
        "### 📊 Average Reranked Chunks"
    )

    reranked = [
        float(
            item.get(
                "average_reranked",
                0
            )
        )
        for item in history
    ]

    st.line_chart(
        {
            "Reranked Chunks": reranked
        }
    )

    # ========================================================
    # VERIFICATION
    # ========================================================

    st.markdown(
        "### 🛡️ Verification Success"
    )

    verification = [
        float(
            item.get(
                "verification_success",
                0
            )
        )
        for item in history
    ]

    st.line_chart(
        {
            "Verification (%)": verification
        }
    )

    # ========================================================
    # PREVIOUS RUNS
    # ========================================================

    st.markdown(
        "### 📋 Previous Evaluation Runs"
    )

    for item in reversed(history):

        with st.expander(
            f"🧪 Run "
            f"{item.get('run', '?')} — "
            f"{item.get('timestamp', '')} — "
            f"{item.get('accuracy', 0)}%"
        ):

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Tests",
                    item.get(
                        "total_tests",
                        0
                    )
                )

            with c2:
                st.metric(
                    "Passed",
                    item.get(
                        "passed",
                        0
                    )
                )

            with c3:
                st.metric(
                    "Failed",
                    item.get(
                        "failed",
                        0
                    )
                )

            with c4:
                st.metric(
                    "Accuracy",
                    f"{item.get('accuracy', 0)}%"
                )

            st.write(
                f"**Average Attempts:** "
                f"{item.get('average_attempts', 0)}"
            )

            st.write(
                f"**Average Response:** "
                f"{item.get('average_response_time', 0)} sec"
            )

            st.write(
                f"**Average Retrieved:** "
                f"{item.get('average_retrieved', 0)}"
            )

            st.write(
                f"**Average Reranked:** "
                f"{item.get('average_reranked', 0)}"
            )

            st.write(
                f"**Verification Success:** "
                f"{item.get('verification_success', 0)}%"
            )