from ingest import load_pdfs
from rag import chunk_text, build_vectorstore, answer_question

text = load_pdfs()
chunks = chunk_text(text)
vs = build_vectorstore(chunks)

while True:
    q = input("Ask a question: ")
    print(answer_question(q, vs))

'''
debug_chunks.py(from ingest import load_pdfs
from rag import chunk_text

text = load_pdfs()
print("Total characters loaded from PDFs:", len(text))

chunks = chunk_text(text)
print("Number of chunks:", len(chunks))

if chunks:
    print("First chunk preview:\n", chunks[0][:300])
else:
    print("No chunks created. Check your PDFs / ingest pipeline.")"""'''