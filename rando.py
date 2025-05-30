# src/ingest.py

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document


# 🔧 Step 1: Load PDFs from /data folder
def load_all_pdfs(data_dir: str) -> list[Document]:
    all_docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            pages = loader.load()
            all_docs.extend(pages)
    return all_docs

# ✂️ Step 2: Split by numbered section headers like “1.”, “2.1.”, etc.
def split_by_headers(docs: list[Document]) -> list[Document]:
    section_docs = []
    section_pattern = r"(?m)(?=^\d+(\.\d+)*\.\s)"  # e.g., matches “1. ” or “2.1. ” at line start

    for doc in docs:
        import re
        matches = list(re.finditer(section_pattern, doc.page_content))
        starts = [m.start() for m in matches] + [len(doc.page_content)]
        for i in range(len(starts) - 1):
            chunk = doc.page_content[starts[i]:starts[i+1]].strip()
            if chunk:
                section_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return section_docs

# 🧩 Step 3: Further split big sections (>2000 chars) into ~400-char chunks
def recursive_chunking(section_chunks: list[Document]) -> list[Document]:
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=4000,
        chunk_overlap=500,
        length_function=len,
    )

    final_chunks = []
    for chunk in section_chunks:
        if len(chunk.page_content) > 2000:
            texts = char_splitter.split_text(chunk.page_content)
            for t in texts:
                final_chunks.append(Document(page_content=t, metadata=chunk.metadata))
        else:
            final_chunks.append(chunk)
    return final_chunks

# 🚀 Orchestration
if __name__ == "__main__":
    data_path = str(Path(__file__).resolve().parent.parent / "data")
    print(f"📄 Loading PDFs from {data_path}")
    raw_docs = load_all_pdfs(data_path)
    
    print(f"🔍 Splitting by section headers")
    section_chunks = split_by_headers(raw_docs)
    
    print(f"🧩 Further splitting large chunks")
    final_chunks = recursive_chunking(section_chunks)

    print(f"✅ Done. Total final chunks: {len(final_chunks)}")





# src/embed.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pickle
import os

# ⚠️ Use same chunking logic and data path from ingest step
from ingest import split_by_headers, load_all_pdfs

DATA_DIR = "./data"
DB_DIR = "./vectorstore"

# ✅ Choose a good multilingual embedding model (handles checkmarks, ₹ symbols, etc.)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ✅ Step 1: Load and split PDFs into final chunks
docs = load_all_pdfs(DATA_DIR)
final_chunks = split_by_headers(docs)

# ✅ Step 2: Convert chunks to vector embeddings and save to FAISS
print(f"Embedding {len(final_chunks)} chunks...")

vectorstore = FAISS.from_documents(final_chunks, embedding_model)

# Create output folder if it doesn't exist
os.makedirs(DB_DIR, exist_ok=True)

# Save vector index and document store
vectorstore.save_local(DB_DIR)
print(f"\n✅ FAISS vector store saved to: {DB_DIR}")



import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ─── NEW IMPORTS ───────────────────────────────────────────────────────────────
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# ────────────────────────────────────────────────────────────────────────────────

# ---- CONFIG ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH            = "vectorstore"  
MODEL_NAME            = "gemma3:1b"

# ---- INIT COMPONENTS ----
@st.cache_resource
def load_chain():
    # 1️⃣ Load embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 2️⃣ Create a retriever with MMR enabled
    retriever = db.as_retriever(
        search_type="mmr",      # ← Maximum Marginal Relevance
        search_kwargs={
            "k": 3,              # top 3 final docs
            "fetch_k": 10        # fetch 10 then re-rank
        }
    )

    # 3️⃣ Setup conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",     # matches the chain’s input dict key
        output_key="answer",      # matches the chain’s output dict key
        return_messages=True
    )

    # 4️⃣ Build a conversational RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model=MODEL_NAME),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain

# Instantiate the chain once
qa_chain = load_chain()

# ---- UI ----
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask me anything about HR policies! (Try mentioning a specific policy name for filtering)")

# Hold chat history in session
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Your question:")

if query:
    # ─── OPTIONAL: metadata filter based on policy mentions ────────────────────
    # e.g. if user asks “in the Leave Policy…” we only retrieve from leave_policy.pdf
    policy_filter = {}
    lowered = query.lower()
    if "leave policy"   in lowered: policy_filter["source"] = "leave_policy.pdf"
    elif "referral policy" in lowered: policy_filter["source"] = "employee_referral_policy.pdf"
    # add more mappings as needed…
    # ────────────────────────────────────────────────────────────────────────────

    with st.spinner("Thinking..."):
        result = qa_chain(
            {"question": query, 
             **({"metadata_filter": policy_filter} if policy_filter else {})}
        )

    # Save history
    st.session_state.history.append((query, result["answer"]))

    # Display answer
    st.subheader("📌 Answer:")
    st.write(result["answer"])

    # Display sources
    with st.expander("📄 Sources"):
        for doc in result["source_documents"]:
            st.markdown(f"- **Source**: {doc.metadata.get('source', 'unknown')} | **Page**: {doc.metadata.get('page', 'n/a')}")
            st.markdown(doc.page_content[:200] + "…")

# Display chat history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    for user_q, bot_a in st.session_state.history:
        st.markdown(f"**You:** {user_q}")  
        st.markdown(f"**Bot:** {bot_a}")
