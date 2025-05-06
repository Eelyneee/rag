import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import subprocess
import streamlit as st

## Step 1 - Load and convert to chunk
csv_path="Resaleflatpricesfrom2024.csv"

def load_and_chunk_csv(csv_path):
    df = pd.read_csv(csv_path)
    def row_to_text(row):
        return (
            f"Month: {row['month']}, Town: {row['town']}, Flat type: {row['flat_type']}, "
            f"Block: {row['block']}, Street: {row['street_name']}, Storey range: {row['storey_range']}, "
            f"Floor area: {row['floor_area_sqm']} sqm, Flat model: {row['flat_model']}, "
            f"Lease start: {row['lease_commence_date']}, Remaining lease: {row['remaining_lease']}, "
            f"Resale price: ${row['resale_price']}"
        )
    return df.apply(row_to_text, axis=1).tolist()
    #  -- Month: 2024-02, Town: ANG MO KIO, Flat type: 2 ROOM, Block: 406, Street: ANG MO KIO AVE 10, Storey range: 01 TO 03, Floor area: 44.0 sqm, Flat model: Improved, Lease start: 1979, Remaining lease: 54 years 4 months, Resale price: $285000

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
    chunks = load_and_chunk_csv(csv_path)

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

uploaded_file = st.file_uploader("Upload your resale_flat_prices.csv file", type=["csv"])

if uploaded_file:
    question = st.text_input("Ask a question about the HDB resale data:",
                             placeholder="e.g. What are the resale prices for 2-room flats in Ang Mo Kio?")

    if question:
        with st.spinner("Processing your question..."):
            chunks = load_and_chunk_csv(uploaded_file)
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