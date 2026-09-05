import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ==========================================
# CONFIGURATION
# ==========================================

PDF_PATH = "data/documents/sample.pdf"

CHUNK_SIZE = 1000
OVERLAP = 200


# ==========================================
# STEP 1: Read PDF page by page
# ==========================================

print("Reading PDF...")

reader = PdfReader(PDF_PATH)

documents = []

for page_number, page in enumerate(
    reader.pages,
    start=1
):
    text = page.extract_text()

    if text:
        documents.append({
            "page": page_number,
            "text": text
        })


print(
    f"PDF pages loaded: {len(documents)}"
)


# ==========================================
# STEP 2: Create chunks with metadata
# ==========================================

chunks = []
metadatas = []
ids = []

chunk_id = 0

for document in documents:

    page_number = document["page"]
    text = document["text"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

            metadatas.append({
                "source": "sample.pdf",
                "page": page_number
            })

            ids.append(
                f"chunk_{chunk_id}"
            )

            chunk_id += 1

        start = end - OVERLAP


print(
    f"Total chunks created: {len(chunks)}"
)


# ==========================================
# STEP 3: Load embedding model
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# STEP 4: Create embeddings
# ==========================================

print("\nCreating embeddings...")

embeddings = model.encode(chunks)

print("Embeddings created!")


# ==========================================
# STEP 5: Connect to ChromaDB
# ==========================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# ==========================================
# STEP 6: Create/get collection
# ==========================================

collection = client.get_or_create_collection(
    name="rag_documents"
)


# ==========================================
# STEP 7: Store data
# ==========================================

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# ==========================================
# STEP 8: Display result
# ==========================================

print("\n===================================")
print("VECTOR DATABASE UPDATED")
print("===================================\n")

print("Total chunks:", len(chunks))

print("Metadata added: source + page")

print("\nKnowledge base is ready! 🚀")