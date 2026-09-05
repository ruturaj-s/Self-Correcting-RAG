import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: Load embedding model
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# STEP 2: Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="rag_documents"
)


# ==========================================
# STEP 3: Ask a question
# ==========================================

question = input("\nAsk a question about your PDF: ")


# ==========================================
# STEP 4: Convert question into embedding
# ==========================================

question_embedding = model.encode(
    [question]
).tolist()


# ==========================================
# STEP 5: Search the vector database
# ==========================================

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)


# ==========================================
# STEP 6: Display retrieved chunks
# ==========================================

print("\n===================================")
print("RETRIEVED INFORMATION")
print("===================================")

documents = results["documents"][0]

for i, document in enumerate(documents, start=1):

    print(f"\n========== RESULT {i} ==========\n")

    print(document)

    print("\n===================================")