import streamlit as st
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

st.title("Chat with PDF (RAG Application)")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# vector database
client = chromadb.Client()
collection = client.get_or_create_collection(name="pdf_data")

# upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    embeddings = model.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )

    st.success("PDF uploaded and processed!")

# ask question
question = st.text_input("Ask a question about the PDF")

if question:

    q_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=2
    )

    context = " ".join(results["documents"][0])

    # SHORT ANSWER (first 2 sentences)
    answer = ". ".join(context.split(". ")[:2])

    st.subheader("Answer")
    st.write(answer)

# clear database
if st.button("Clear Vector Database"):
    client.delete_collection("pdf_data")
    st.success("Database Cleared!")