# 🏠 HDB Resale Flat RAG Chatbot

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about **HDB resale flat prices** in Singapore from **January 2024 to May 2025**. It uses local AI models via **Ollama**, text embeddings with **SentenceTransformers**, and fast semantic search using **FAISS**.

## 🚀 Features

- 💬 Ask natural questions like:
  - "What are the resale prices of 2-room flats in Ang Mo Kio?"
  - "Which towns have the highest prices for 5-room flats in 2024?"
- 📚 Retrieves real data from your uploaded CSV
- 🧠 Uses an LLM (e.g., `llama2` or `mistral`) running locally via Ollama to generate context-aware answers
- ⚡ Fast search with FAISS & semantic embeddings
- 🎛️ Streamlit web interface

## 🧱 Tech Stack

| Component        | Tool / Library              |
|------------------|-----------------------------|
| Language Model   | [Ollama](https://ollama.com) (`llama2`) |
| Embeddings       | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB        | `faiss-cpu`                 |
| UI               | `Streamlit`                 |
| Language         | Python                      |

## 📂 File Structure
├── app.py # Main Streamlit app
├── resale_index.faiss # Saved FAISS index (auto-generated)
├── resale_chunks.pkl # Saved text chunks (auto-generated)
├── Resaleflatpricesfrom2024.csv # Your resale data (uploaded through UI)

## 📦 Setup Instructions

1. **Install dependencies** (use a virtualenv or conda env):
   ```bash
   pip install -r requirements.txt
2. Install and run Ollama:
    - Download: https://ollama.com/download
3. Run the app:
    ```bash
    streamlit run app.py
4. Upload your CSV (should have columns like month, town, flat_type, resale_price, etc.) and start chatting!

## 📈 Sample Questions to Try
- "What is the average price of 3-room flats in Toa Payoh in 2023?"
- "How much are 4-room flats selling for in Queenstown recently?"
- "Show resale price trends for Bukit Batok."

## 🔐 Local & Private
No cloud APIs needed. All data and model inference are done locally, making it ideal for privacy-focused use cases.

## 🛠 Future Improvements
- Add charts/visuals (e.g., matplotlib or Altair)
- Support average price or trendline calculations
- Enhance response formatting (tables, summaries)
- Add multi-question memory/chat history



## Data Source
Housing & Development Board. (2021). Resale flat prices based on registration date from Jan-2017 onwards (2025) [Dataset]. data.gov.sg. Retrieved May 6, 2025 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view