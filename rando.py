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





appp



import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH           = "vectorstore"
MODEL_NAME           = "gemma3:1b"

# ─── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACME HR Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stChatMessage { border-radius:12px; padding:8px; margin-bottom:4px; }
  .stChatMessage.user { background-color:#e6f7ff; }
  .stChatMessage.assistant { background-color:#f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='display:flex; align-items:center;'>
      <img src="https://path.to/your/logo.png" width="50" style='margin-right:12px'/>
      <h1 style='margin:0; font-family:sans-serif;'>ACME Corp HR Assistant</h1>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# ─── LOAD & CACHE RAG CHAIN ───────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    llm = Ollama(
        model=MODEL_NAME,
        streaming=True,
        callbacks=[StreamlitCallbackHandler()]
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

qa_chain = load_chain()

# ─── SESSION STATE FOR CHAT & TOPICS ───────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "topic_filter" not in st.session_state:
    st.session_state.topic_filter = {}

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar.expander("📂 Manage Policies"):
    uploaded = st.file_uploader("Upload new policy PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Rebuild Index"):
        # TODO: save uploads to disk, re-run your ingestion + indexing pipeline,
        # then clear cache so load_chain() reloads new vectorstore.
        st.cache_resource.clear()
        st.success("Vectorstore rebuilt!")

st.sidebar.markdown("### 🔍 Quick Topics")
for topic, src in [("Leave Policy", "leave_policy.pdf"),
                   ("Referral Policy", "employee_referral_policy.pdf"),
                   ("Benefits", None),
                   ("Payroll", None)]:
    if st.sidebar.button(topic):
        st.session_state.history = []
        if src:
            st.session_state.topic_filter = {"source": src}
        else:
            st.session_state.topic_filter = {}
        st.experimental_rerun()

# ─── CHAT INTERFACE ────────────────────────────────────────────────────────────
# 1️⃣ Display existing messages
for msg in st.session_state.history:
    st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"]=="user" else "🤖").write(msg["text"])

# 2️⃣ Get user input
if prompt := st.chat_input("Ask me about any HR policy…"):
    # Save & show user
    st.session_state.history.append({"role": "user", "text": prompt})

    # Build metadata filter from sidebar or detect from text
    meta = st.session_state.topic_filter.copy()
    if "leave policy" in prompt.lower():
        meta["source"] = "leave_policy.pdf"
    elif "referral policy" in prompt.lower():
        meta["source"] = "employee_referral_policy.pdf"

    # 3️⃣ Generate answer with spinner + streaming
    with st.spinner("Thinking…"):
        result = qa_chain({
            "question": prompt,
            **({"metadata_filter": meta} if meta else {})
        })
    answer = result["answer"]

    # 4️⃣ Save & show assistant
    st.session_state.history.append({"role": "assistant", "text": answer})
    st.chat_message("assistant", avatar="🤖").write(answer)

    # 5️⃣ Show sources in two columns
    with st.expander("📄 Source Snippets"):
        cols = st.columns(2)
        docs = result["source_documents"]
        for i, doc in enumerate(docs):
            col = cols[i % 2]
            src = doc.metadata.get("source", "unknown")
            pg  = doc.metadata.get("page", "n/a")
            snippet = doc.page_content[:150].replace("\n", " ") + "…"
            col.markdown(f"**{src} (pg {pg})**")
            col.write(snippet)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 ACME Corp • For internal use only • hr@acme.com")






appppppppp



import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH           = "vectorstore"
MODEL_NAME           = "gemma3:1b"

# ─── PAGE & THEME SETUP ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PwC HR Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Inline CSS for PwC theme & chat bubbles
st.markdown("""
<style>
  /* Backgrounds and fonts */
  .appview-container { background-color: #FFFFFF; color: #333333; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
  .stButton>button { background-color: #FF8000; border-radius: 6px; color: white; }
  .stSidebar .sidebar-content { background-color: #F7F7F7; }
  /* Chat bubbles */
  .stChatMessage.user { background-color: #FFEFE0 !important; border-radius:12px; padding:8px; margin:4px 0; }
  .stChatMessage.assistant { background-color: #F0F0F0 !important; border-radius:12px; padding:8px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; padding-bottom:8px;'>
  <img src="https://your.pwc.logo/url.png" width="48" style='margin-right:12px;'/>
  <h1 style='margin:0; color:#333;'>PwC HR Assistant</h1>
</div>
<hr>
""", unsafe_allow_html=True)

# ─── CACHED RESOURCE LOADER ────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Embeddings + Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
    )
    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question", output_key="answer", return_messages=True
    )
    return retriever, memory

retriever, memory = load_resources()

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "topic_filter" not in st.session_state:
    st.session_state.topic_filter = {}

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar.expander("📂 Manage Policies"):
    uploads = st.file_uploader("Upload policy PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Rebuild Index"):
        # TODO: handle saving uploads & re-indexing
        st.cache_resource.clear()
        st.success("✅ Vectorstore rebuilt with new docs!")

st.sidebar.markdown("### 🔍 Quick Topics")
for label, src in [
    ("Leave Policy", "leave_policy.pdf"),
    ("Referral Policy", "employee_referral_policy.pdf"),
    ("Benefits", None),
    ("Payroll", None),
]:
    if st.sidebar.button(label):
        st.session_state.history = []
        st.session_state.topic_filter = {"source": src} if src else {}
        st.experimental_rerun()

# ─── CHAT UI ───────────────────────────────────────────────────────────────────
# 1️⃣ Render existing chat
for msg in st.session_state.history:
    st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"]=="user" else "🤖").write(msg["text"])

# 2️⃣ Capture user input
if prompt := st.chat_input("Ask me about any HR policy…"):
    st.session_state.history.append({"role": "user", "text": prompt})

    # build metadata filter
    meta = st.session_state.topic_filter.copy()
    if "leave policy" in prompt.lower():      meta["source"] = "leave_policy.pdf"
    if "referral policy" in prompt.lower():  meta["source"] = "employee_referral_policy.pdf"

    # placeholder for streaming output
    response_container = st.empty()
    stream_handler = StreamlitCallbackHandler(parent_container=response_container)

    # streaming LLM + chain
    llm = Ollama(model=MODEL_NAME, streaming=True, callbacks=[stream_handler])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    # 3️⃣ Run the chain
    with st.spinner("Thinking…"):
        result = chain({
            "question": prompt,
            **({"metadata_filter": meta} if meta else {})
        })

    # 4️⃣ Show & store assistant reply
    answer = result["answer"]
    st.session_state.history.append({"role": "assistant", "text": answer})
    st.chat_message("assistant", avatar="🤖").write(answer)

    # 5️⃣ Two-column sources
    with st.expander("📄 Source Snippets"):
        cols = st.columns(2)
        for i, doc in enumerate(result["source_documents"]):
            col = cols[i % 2]
            src = doc.metadata.get("source", "unknown")
            pg  = doc.metadata.get("page", "n/a")
            snippet = doc.page_content[:150].replace("\n", " ") + "…"
            col.markdown(f"**{src}** (pg {pg})")
            col.write(snippet)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 PwC • For internal use only • hr_chatbot@pwc.com")







apppppppppppppppppppppppppppppppppppppppp








import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH           = "vectorstore"
MODEL_NAME           = "gemma3:1b"

# ─── PAGE & THEME SETUP ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PwC HR Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Inline CSS for PwC theme & chat bubbles
st.markdown("""
<style>
  .appview-container { background-color: #FFFFFF; color: #333333; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
  .stButton>button { background-color: #FF8000; border-radius: 6px; color: white; }
  .stSidebar .sidebar-content { background-color: #F7F7F7; }
  .stChatMessage.user { background-color: #FFEFE0 !important; border-radius:12px; padding:8px; margin:4px 0; }
  .stChatMessage.assistant { background-color: #F0F0F0 !important; border-radius:12px; padding:8px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)
# ─── HEADER ───────────────────────────────────────────────────────────────────
logo_path = "images.png"
col1, col2 = st.columns([1, 8], gap="small")
with col1:
    st.image(logo_path, width=88)
with col2:
    st.markdown(
        "<h1 style='margin:0; color:#333;'>HR Assistant</h1>",
        unsafe_allow_html=True
    )

# ─── LOAD & CACHE RAG RESOURCES ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question", output_key="answer", return_messages=True
    )
    return retriever, memory

retriever, memory = load_resources()

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "topic_filter" not in st.session_state:
    st.session_state.topic_filter = {}

# ─── SIDEBAR: Conversation History ────────────────────────────────────────────
st.sidebar.markdown("### 🕑 Conversation History")
for entry in st.session_state.history:
    role = "You" if entry["role"] == "user" else "Bot"
    st.sidebar.markdown(f"**{role}:** {entry['text']}")

# ─── CHAT INTERFACE ────────────────────────────────────────────────────────────
# Display past chat messages
for msg in st.session_state.history:
    st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"]=="user" else "🤖").write(msg["text"])

# Capture user input
if prompt := st.chat_input("Ask me about any HR policy…"):
    st.session_state.history.append({"role": "user", "text": prompt})

    # Prepare metadata filter if needed
    meta = {}
    if "leave policy" in prompt.lower():
        meta["source"] = "leave_policy.pdf"
    if "referral policy" in prompt.lower():
        meta["source"] = "employee_referral_policy.pdf"

    # Set up streaming response area
    response_container = st.empty()
    stream_handler = StreamlitCallbackHandler(parent_container=response_container)

    # Build streaming LLM & chain
    llm = Ollama(model=MODEL_NAME, streaming=True, callbacks=[stream_handler])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    # Generate and display answer
    with st.spinner("Thinking…"):
        result = chain({
            "question": prompt,
            **({"metadata_filter": meta} if meta else {})
        })
    answer = result["answer"]

    st.session_state.history.append({"role": "assistant", "text": answer})
    st.chat_message("assistant", avatar="🤖").write(answer)

    # Show source snippets
    with st.expander("📄 Source Snippets"):
        cols = st.columns(2)
        for i, doc in enumerate(result["source_documents"]):
            col = cols[i % 2]
            src = doc.metadata.get("source", "unknown")
            pg  = doc.metadata.get("page", "n/a")
            snippet = doc.page_content[:150].replace("\n", " ") + "…"
            col.markdown(f"**{src}** (pg {pg})")
            col.write(snippet)

