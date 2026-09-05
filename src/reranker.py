from sentence_transformers import CrossEncoder


# ==========================================
# STEP 1: Load reranking model
# ==========================================

print("Loading reranking model...")

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranking model loaded!")


# ==========================================
# STEP 2: Example question
# ==========================================

question = "What is Retrieval-Augmented Generation?"


# ==========================================
# STEP 3: Example retrieved chunks
# ==========================================

documents = [
    "RAG combines information retrieval with text generation.",
    "Machine learning models can be trained using different optimization techniques.",
    "Retrieval-Augmented Generation retrieves relevant information from an external knowledge source before generating an answer.",
    "Python is a popular programming language.",
    "RAG can help ground language model responses using retrieved documents."
]


# ==========================================
# STEP 4: Create question-document pairs
# ==========================================

pairs = []

for document in documents:

    pairs.append(
        [question, document]
    )


# ==========================================
# STEP 5: Calculate relevance scores
# ==========================================

scores = model.predict(pairs)


# ==========================================
# STEP 6: Combine documents and scores
# ==========================================

ranked_documents = list(
    zip(documents, scores)
)


# ==========================================
# STEP 7: Sort by relevance
# ==========================================

ranked_documents.sort(
    key=lambda x: x[1],
    reverse=True
)


# ==========================================
# STEP 8: Display results
# ==========================================

print("\n===================================")
print("RERANKING RESULTS")
print("===================================\n")


for rank, (document, score) in enumerate(
    ranked_documents,
    start=1
):

    print(f"Rank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {document}")
    print("-----------------------------------")