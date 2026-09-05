"""
RAG Performance Analytics

Reads data/evaluation_history.json created by app.py.

This module:
- Displays historical RAG performance
- Shows accuracy, response time and attempts
- Shows retrieval and reranking metrics
- Shows verification correctness
- Shows system health
- Handles incomplete/old evaluation records safely
- Prevents NaN/int conversion errors
"""

import json
import os
from typing import Any

import pandas as pd
import streamlit as st


HISTORY_FILE = "data/evaluation_history.json"


# ============================================================
# LOAD EVALUATION HISTORY
# ============================================================

def _load_history() -> list[dict[str, Any]]:
    """
    Load evaluation history safely.
    """

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, list):
            return []

        return [
            row
            for row in data
            if isinstance(row, dict)
        ]

    except Exception as error:

        st.warning(
            f"Could not read evaluation history: {error}"
        )

        return []


# ============================================================
# VALID RUN FILTER
# ============================================================

def _valid_runs(
    history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Keep evaluation runs containing the required
    RAG performance metrics.

    Older runs may not contain all metrics.
    """

    valid = []

    for row in history:

        required = (
            "accuracy",
            "average_response_time",
            "average_attempts",
            "average_retrieved",
            "average_reranked",
        )

        if all(
            key in row
            for key in required
        ):
            valid.append(row)

    return valid


# ============================================================
# SAFE NUMBER
# ============================================================

def _safe_number(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Convert a value safely to float.

    Prevents:
        NaN
        None
        empty strings
        invalid values

    from breaking the dashboard.
    """

    try:

        number = float(value)

        if pd.isna(number):
            return default

        return number

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# SAFE INTEGER
# ============================================================

def _safe_integer(
    value: Any,
    default: int = 0
) -> int:
    """
    Convert a value safely to integer.
    """

    try:

        number = float(value)

        if pd.isna(number):
            return default

        return int(number)

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# RENDER PERFORMANCE DASHBOARD
# ============================================================

def render_rag_performance():

    st.divider()

    st.header(
        "📊 RAG Performance Analytics"
    )

    # --------------------------------------------------------
    # Load history
    # --------------------------------------------------------

    history = _load_history()

    if not history:

        st.info(
            "Run Automatic RAG Evaluation once "
            "to generate performance data."
        )

        return

    # --------------------------------------------------------
    # Filter valid runs
    # --------------------------------------------------------

    runs = _valid_runs(history)

    if not runs:

        st.warning(
            "No complete evaluation history is available. "
            "Run Automatic RAG Evaluation again."
        )

        return

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(runs)

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "run",
        "accuracy",
        "average_response_time",
        "average_attempts",
        "average_retrieved",
        "average_reranked",
        "verification_success",
        "supported_success",
        "supported_total",
        "unsupported_success",
        "unsupported_total",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].fillna(0)

    # --------------------------------------------------------
    # Ensure run number is always valid
    # --------------------------------------------------------

    if "run" in df.columns:

        df["run"] = [
            _safe_integer(
                value,
                index + 1
            )
            for index, value
            in enumerate(df["run"])
        ]

    else:

        df["run"] = list(
            range(
                1,
                len(df) + 1
            )
        )

    # --------------------------------------------------------
    # Latest run
    # --------------------------------------------------------

    latest = df.iloc[-1]

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    st.subheader(
        "🎯 Overall Performance"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Historical Avg Accuracy",
            f"{df['accuracy'].mean():.1f}%"
        )

    with c2:

        st.metric(
            "Historical Avg Response",
            f"{df['average_response_time'].mean():.2f}s"
        )

    with c3:

        st.metric(
            "Historical Avg Attempts",
            f"{df['average_attempts'].mean():.2f}"
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        st.metric(
            "Avg Retrieved Chunks",
            f"{df['average_retrieved'].mean():.2f}"
        )

    with c5:

        st.metric(
            "Avg Reranked Chunks",
            f"{df['average_reranked'].mean():.2f}"
        )

    with c6:

        # IMPORTANT:
        # This is the actual correctness of the
        # evaluation/verification decision.
        verification_accuracy = (
            df["accuracy"].mean()
        )

        st.metric(
            "Verification Accuracy",
            f"{verification_accuracy:.1f}%"
        )

    # ========================================================
    # LATEST RUN
    # ========================================================

    st.subheader(
        "📌 Latest Evaluation Run"
    )

    latest_accuracy = _safe_number(
        latest.get("accuracy", 0)
    )

    latest_response = _safe_number(
        latest.get(
            "average_response_time",
            0
        )
    )

    latest_attempts = _safe_number(
        latest.get(
            "average_attempts",
            0
        )
    )

    latest_retrieved = _safe_number(
        latest.get(
            "average_retrieved",
            0
        )
    )

    latest_reranked = _safe_number(
        latest.get(
            "average_reranked",
            0
        )
    )

    # Verification correctness should use
    # evaluation accuracy, NOT the percentage of
    # answers classified as SUPPORTED.
    latest_verification_accuracy = (
        latest_accuracy
    )

    latest_run_number = _safe_integer(
        latest.get("run", 0)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Accuracy",
            f"{latest_accuracy:.1f}%"
        )

    with c2:

        st.metric(
            "Response Time",
            f"{latest_response:.2f}s"
        )

    with c3:

        st.metric(
            "Attempts",
            f"{latest_attempts:.2f}"
        )

    with c4:

        st.metric(
            "Verification",
            f"{latest_verification_accuracy:.1f}%"
        )

    c5, c6, c7 = st.columns(3)

    with c5:

        st.metric(
            "Retrieved Chunks",
            f"{latest_retrieved:.2f}"
        )

    with c6:

        st.metric(
            "Reranked Chunks",
            f"{latest_reranked:.2f}"
        )

    with c7:

        st.metric(
            "Run Number",
            latest_run_number
        )

    # ========================================================
    # EVALUATION RUN DETAILS
    # ========================================================

    st.subheader(
        "📋 Evaluation Run Details"
    )

    display_columns = [
        "run",
        "accuracy",
        "average_response_time",
        "average_attempts",
        "average_retrieved",
        "average_reranked",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    display_df = df[
        available_columns
    ].copy()

    rename_map = {

        "run":
            "Run",

        "accuracy":
            "Accuracy (%)",

        "average_response_time":
            "Response (s)",

        "average_attempts":
            "Attempts",

        "average_retrieved":
            "Retrieved Chunks",

        "average_reranked":
            "Reranked Chunks",
    }

    display_df.columns = [
        rename_map.get(
            column,
            column
        )
        for column in display_df.columns
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ACCURACY CHART
    # ========================================================

    st.subheader(
        "📈 Accuracy Across Evaluation Runs"
    )

    accuracy_chart = df[
        ["run", "accuracy"]
    ].copy()

    accuracy_chart = (
        accuracy_chart
        .set_index("run")
    )

    st.line_chart(
        accuracy_chart,
        y="accuracy"
    )

    # ========================================================
    # RESPONSE TIME CHART
    # ========================================================

    st.subheader(
        "⏱️ Response Time Across Runs"
    )

    response_chart = df[
        [
            "run",
            "average_response_time"
        ]
    ].copy()

    response_chart = (
        response_chart
        .set_index("run")
    )

    st.line_chart(
        response_chart,
        y="average_response_time"
    )

    # ========================================================
    # ATTEMPTS CHART
    # ========================================================

    st.subheader(
        "🔄 Attempts Across Runs"
    )

    attempts_chart = df[
        [
            "run",
            "average_attempts"
        ]
    ].copy()

    attempts_chart = (
        attempts_chart
        .set_index("run")
    )

    st.line_chart(
        attempts_chart,
        y="average_attempts"
    )

    # ========================================================
    # RETRIEVAL & RERANKING
    # ========================================================

    st.subheader(
        "📚 Retrieval & Reranking"
    )

    retrieval_chart = df[
        [
            "run",
            "average_retrieved",
            "average_reranked"
        ]
    ].copy()

    retrieval_chart = (
        retrieval_chart
        .set_index("run")
    )

    retrieval_chart.columns = [
        "Retrieved Chunks",
        "Reranked Chunks"
    ]

    st.line_chart(
        retrieval_chart
    )

    # ========================================================
    # VERIFICATION ACCURACY
    # ========================================================

    st.subheader(
        "🛡️ Verification Accuracy"
    )

    verification_chart = df[
        [
            "run",
            "accuracy"
        ]
    ].copy()

    verification_chart = (
        verification_chart
        .set_index("run")
    )

    verification_chart.columns = [
        "Verification Accuracy (%)"
    ]

    st.line_chart(
        verification_chart
    )

    # ========================================================
    # SYSTEM HEALTH
    # ========================================================

    st.subheader(
        "💚 RAG System Health"
    )

    health_problems = []

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if latest_accuracy < 80:

        health_problems.append(
            "accuracy is below 80%"
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    if latest_retrieved <= 0:

        health_problems.append(
            "retrieval metrics are zero"
        )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    if latest_reranked <= 0:

        health_problems.append(
            "reranking metrics are zero"
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Do NOT use the old verification_success field here.
    #
    # The evaluation dataset contains:
    #
    # 3 supported questions
    # 2 unsupported questions
    #
    # Therefore:
    #
    # 3 / 5 = 60%
    #
    # does NOT mean the verifier failed.
    #
    # The correct metric is evaluation accuracy.
    #

    if latest_verification_accuracy < 80:

        health_problems.append(
            "verification accuracy is below 80%"
        )

    # --------------------------------------------------------
    # Response time
    # --------------------------------------------------------

    if latest_response > 15:

        health_problems.append(
            "response time is above 15 seconds"
        )

    # --------------------------------------------------------
    # Display health
    # --------------------------------------------------------

    if not health_problems:

        st.success(
            "🟢 HEALTHY — RAG pipeline is "
            "performing within acceptable limits."
        )

    else:

        st.warning(
            "🟡 NEEDS IMPROVEMENT — "
            + "; ".join(health_problems)
            + "."
        )

    # ========================================================
    # PERFORMANCE INTERPRETATION
    # ========================================================

    st.subheader(
        "🔎 Performance Interpretation"
    )

    if latest_accuracy >= 95:

        st.success(
            "🎯 Accuracy is excellent. "
            "The evaluation pipeline is correctly "
            "handling the current test set."
        )

    elif latest_accuracy >= 80:

        st.info(
            "Accuracy is acceptable, but there is "
            "room for improvement."
        )

    else:

        st.error(
            "Accuracy requires improvement."
        )

    if latest_response <= 5:

        st.success(
            "⚡ Response time is excellent."
        )

    elif latest_response <= 15:

        st.info(
            "⏱️ Response time is acceptable."
        )

    else:

        st.warning(
            f"🐢 Response time is high "
            f"({latest_response:.2f}s). "
            "This is the main optimization target."
        )

    # ========================================================
    # EXPORT
    # ========================================================

    st.subheader(
        "📥 Export Evaluation Data"
    )

    csv_data = (
        df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Performance CSV",
        data=csv_data,
        file_name="rag_performance_history.csv",
        mime="text/csv"
    )