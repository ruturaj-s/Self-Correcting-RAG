from sentence_transformers import SentenceTransformer


# --------------------------------
# Load the embedding model
# --------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# --------------------------------
# Example text
# --------------------------------

texts = [
    "Retrieval-Augmented Generation uses external knowledge.",
    "RAG retrieves relevant documents before generating an answer.",
    "Python is a programming language."
]


# --------------------------------
# Convert text into embeddings
# --------------------------------

embeddings = model.encode(texts)


# --------------------------------
# Display results
# --------------------------------

print("\n===================================")
print("EMBEDDING TEST")
print("===================================")

print("Number of texts:", len(texts))

print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])
