import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import subprocess
import streamlit as st
import pymupdf 
import docx

## Step 1 - Load and convert to chunk
csv_path="Resaleflatpricesfrom2024.csv"

def load_and_chunk_csvs(csv_files):
    all_chunks = []
    for uploaded_file in csv_files:
        df = pd.read_csv(uploaded_file)
        chunks = df.apply(lambda row: (
            f"Month: {row['month']}, Town: {row['town']}, Flat type: {row['flat_type']}, "
            f"Block: {row['block']}, Street: {row['street_name']}, Storey range: {row['storey_range']}, "
            f"Floor area: {row['floor_area_sqm']} sqm, Flat model: {row['flat_model']}, "
            f"Lease start: {row['lease_commence_date']}, Remaining lease: {row['remaining_lease']}, "
            f"Resale price: ${row['resale_price']}"), axis=1).tolist()
        all_chunks.extend(chunks)
    return all_chunks
    #  -- Month: 2024-02, Town: ANG MO KIO, Flat type: 2 ROOM, Block: 406, Street: ANG MO KIO AVE 10, Storey range: 01 TO 03, Floor area: 44.0 sqm, Flat model: Improved, Lease start: 1979, Remaining lease: 54 years 4 months, Resale price: $285000

def extract_text_from_pdf(file):
    text = ""
    with pymupdf.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_and_chunk_other_docs(files):
    all_chunks = []
    for f in files:
        if f.name.endswith(".pdf"):
            text = extract_text_from_pdf(f)
        elif f.name.endswith(".docx"):
            text = extract_text_from_docx(f)
        else:
            continue
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
        all_chunks.extend(paragraphs)
    return all_chunks

## Step 2 - Embedding Text Chunks
def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return embeddings, embedder

## Step 3 - Index Embeddings with FAISS
def create_faiss_index(embeddings, chunks, index_path='resale_index.faiss', chunks_path='resale_chunks.pkl'):
    matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

## Step 4 - Query the Index with a User Question
def load_faiss_and_chunks(index_path='resale_index.faiss', chunks_path='resale_chunks.pkl'):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_chunks(query, embedder, index, chunks, top_k=5):
    q_embedding = embedder.encode([query]).astype("float32")
    D, I = index.search(q_embedding, top_k)
    return [chunks[i] for i in I[0]]

## Step 5 - Build Prompt & Call Ollama
def build_prompt(chunks, question):
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    return f"""You are a helpful assistant. Use the provided context to answer the user's question.

Context:
{context}

Question: {question}
Answer:"""

def call_ollama(prompt, model_name="llama2"):
    process = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=prompt.encode("utf-8"))
    return out.decode("utf-8")

def run_rag_chatbot(csv_path, question, model_name="llama2"):
    print("📦 Loading and chunking data...")
    chunks = load_and_chunk_csvs(csv_path)

    print("🧠 Generating embeddings...")
    embeddings, embedder = generate_embeddings(chunks)

    print("📂 Creating FAISS index...")
    create_faiss_index(embeddings, chunks)

    print("🔍 Loading index and retrieving relevant chunks...")
    index, chunks_loaded = load_faiss_and_chunks()
    top_chunks = retrieve_chunks(question, embedder, index, chunks_loaded)

    print("📝 Building prompt and generating answer...")
    prompt = build_prompt(top_chunks, question)
    response = call_ollama(prompt, model_name=model_name)

    print("\n🗨️ Answer:\n")
    print(response)

# Example usage:
# run_rag_chatbot(csv_path, "What are the recent resale prices of 5-room flats in Yishun in block 155?")

# ------------------ Streamlit UI ------------------ #
st.title("🏠 HDB Resale Flat RAG Chatbot")

uploaded_csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
uploaded_docs = st.file_uploader("Upload PDF or DOCX documents", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_csvs or uploaded_docs:
    question = st.text_input("Ask a question about your uploaded documents:",
                             placeholder="e.g. What are the resale prices for 2-room flats in Ang Mo Kio?")

    if question:
        with st.spinner("Processing your question..."):
            csv_chunks = load_and_chunk_csvs(uploaded_csvs) if uploaded_csvs else []
            doc_chunks = load_and_chunk_other_docs(uploaded_docs) if uploaded_docs else []
            chunks = csv_chunks + doc_chunks

            embeddings, embedder = generate_embeddings(chunks)
            create_faiss_index(embeddings, chunks)
            index, chunks_loaded = load_faiss_and_chunks()
            top_chunks = retrieve_chunks(question, embedder, index, chunks_loaded)
            prompt = build_prompt(top_chunks, question)
            response = call_ollama(prompt)

        st.markdown("### 🧠 Answer:")
        st.write(response)

        with st.expander("🔍 Retrieved Context Chunks"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**{i+1}.** {chunk}")