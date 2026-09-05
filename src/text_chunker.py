from pypdf import PdfReader


# -----------------------------
# STEP 1: Read the PDF
# -----------------------------

pdf_path = "data/documents/sample.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"


# -----------------------------
# STEP 2: Create chunks
# -----------------------------

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(full_text):

    end = start + chunk_size

    chunk = full_text[start:end]

    chunks.append(chunk)

    start = end - overlap


# -----------------------------
# STEP 3: Display results
# -----------------------------

print("===================================")
print("TEXT CHUNKER")
print("===================================")

print(f"Total characters: {len(full_text)}")

print(f"Total chunks: {len(chunks)}")


for i, chunk in enumerate(chunks[:5], start=1):

    print(f"\n========== CHUNK {i} ==========\n")

    print(chunk)

    print("\n===================================")