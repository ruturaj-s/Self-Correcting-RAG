from pypdf import PdfReader

# Location of our PDF
pdf_path = "data/documents/sample.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

print("===================================")
print("PDF READER")
print("===================================")

print(f"Number of pages: {len(reader.pages)}")

# Read every page
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n========== PAGE {page_number} ==========\n")
    print(text)