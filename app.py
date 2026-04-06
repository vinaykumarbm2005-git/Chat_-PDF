import streamlit as st
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

st.title("Chat with PDF (RAG Application)")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create vector database
client = chromadb.Client()
collection = client.get_or_create_collection(name="pdf_data")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf", key="pdf_upload")

# Process PDF
if pdf_file:

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Split text into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Create embeddings
    embeddings = model.encode(chunks).tolist()

    # Store in vector database
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )

    st.success("PDF uploaded and processed!")

# Ask question
question = st.text_input("Ask a question about the PDF")

if question:

    if collection.count() == 0:
        st.warning("Please upload a PDF first.")

    else:
        q_embedding = model.encode([question]).tolist()

        results = collection.query(
            query_embeddings=q_embedding,
            n_results=2
        )

        context = " ".join(results["documents"][0])

        # Similarity distance check
        distance = results["distances"][0][0]

        if distance > 1.2:
            st.subheader("Answer")
            st.write("The document does not contain this information.")

        else:
            # Short answer from retrieved context
            answer = ". ".join(context.split(". ")[:2])

            st.subheader("Answer")
            st.write(answer)

# Clear vector database
if st.button("Clear Vector Database"):
    client.delete_collection("pdf_data")
    st.success("Database Cleared!")