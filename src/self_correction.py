"""
Self-Correction Module

Generates an improved search query when the
RAG answer is not supported by the retrieved evidence.
"""

from groq import Groq


def self_correct_query(
    question: str,
    previous_query: str,
    answer: str,
    context: str,
    llm: Groq,
) -> str:
    """
    Generate an improved search query after
    answer verification fails.
    """

    prompt = f"""
You are a Self-Correcting RAG Query Optimizer.

The RAG system generated an answer that was NOT
supported by the retrieved document evidence.

Your task is to generate a better search query
for the next retrieval attempt.

IMPORTANT RULES:

1. Do NOT answer the user's question.
2. Return ONLY the improved search query.
3. Preserve the original meaning of the question.
4. Identify what information was missing.
5. Make the query more specific.
6. Use important keywords from the original question.
7. Avoid unnecessary words.
8. Do not invent information.
9. The query must be suitable for vector database search.
10. Improve the previous query instead of repeating it.

ORIGINAL QUESTION:
{question}

PREVIOUS SEARCH QUERY:
{previous_query}

PREVIOUS ANSWER:
{answer}

RETRIEVED DOCUMENT CONTEXT:
{context}

IMPROVED SEARCH QUERY:
"""

    try:

        response = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document retrieval "
                        "query optimization system."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        corrected_query = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # Remove accidental quotation marks
        corrected_query = corrected_query.strip(
            "\"'`"
        )

        # Safety check
        if not corrected_query:
            return previous_query

        return corrected_query

    except Exception as error:

        print(
            "Self-correction error:",
            error
        )

        # If correction fails,
        # continue using the previous query.
        return previous_query