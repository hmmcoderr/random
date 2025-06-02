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




appppppppsidebarrrrrrrrrrrrrrr
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

# ---- PAGE SETUP & LOGO ----
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Display company logo (place logo.png in the same directory)
logo_path = "logo.png"
if st.sidebar:  # show logo in sidebar
    try:
        st.sidebar.image(logo_path, width=120)
    except Exception:
        pass

# ─── INIT COMPONENTS ----
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
    policy_filter = {}
    lowered = query.lower()
    if "leave policy"   in lowered: policy_filter["source"] = "leave_policy.pdf"
    elif "referral policy" in lowered: policy_filter["source"] = "employee_referral_policy.pdf"
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

# ---- SIDEBAR HISTORY ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 🕑 Conversation History")
for user_q, bot_a in st.session_state.history:
    st.sidebar.markdown(f"**You:** {user_q}")  
    st.sidebar.markdown(f"**Bot:** {bot_a}")

uyjfyujfvyhvjhgj




policy_keywords = {
    "leave policy":                "leave_policy.pdf",
    "employee referral policy":    "employee_referral_policy.pdf",
    "total rewards policy":        "total_rewards_policy.pdf",
    "policy manual":               "policy_manual.pdf",
}

policy_filter = {}
lowered = query.lower()

for keyword, filename in policy_keywords.items():
    if keyword in lowered:
        policy_filter["source"] = filename
        break


ingest.py#2
# ingest.py

import os
import re
import pickle
from pathlib import Path

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PICKLE = DATA_DIR / "chunked_docs.pkl"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_page_tables(pdf_path: str, page_number: int) -> str:
    """
    If the given page contains a table, convert it to Markdown and return as a string.
    Otherwise, return an empty string.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        table = page.extract_table()
        if not table:
            return ""
        # Build a Markdown table
        header = table[0]
        separators = ["---"] * len(header)
        rows = table[1:]
        md_lines = []
        # Header row
        md_lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        # Separator row
        md_lines.append("| " + " | ".join(separators) + " |")
        # Data rows
        for row in rows:
            # Replace None with empty string
            row_cells = [str(cell) if cell is not None else "" for cell in row]
            md_lines.append("| " + " | ".join(row_cells) + " |")
        return "\n".join(md_lines)


def load_and_process_all_pdfs(data_dir: Path) -> list[Document]:
    """
    Iterate through every PDF in `data_dir`, extract text + tables from each page,
    and return a list of Documents (one per page) with metadata {"source", "page"}.
    """
    docs = []
    for filepath in sorted(data_dir.iterdir()):
        if filepath.suffix.lower() != ".pdf":
            continue

        filename = filepath.name
        with pdfplumber.open(str(filepath)) as pdf:
            num_pages = len(pdf.pages)
            for page_idx in range(num_pages):
                page = pdf.pages[page_idx]
                text = page.extract_text() or ""
                table_md = extract_page_tables(str(filepath), page_idx)

                # Append table Markdown if present
                combined = text
                if table_md:
                    combined = combined + "\n\n" + table_md

                # Only create a Document if there's any content
                if combined.strip():
                    metadata = {
                        "source": filename,
                        "page": page_idx + 1,  # 1-based indexing
                    }
                    docs.append(Document(page_content=combined, metadata=metadata))
    return docs


def split_by_section_headers(pages: list[Document]) -> list[Document]:
    """
    Split each page’s content by numbered section headers (e.g., “1. ”, “2.1. ”).
    Return a list of Documents, each corresponding to one section chunk.
    """
    section_pattern = r"(?m)(?=^\d+(\.\d+)*\.\s)"
    section_chunks = []

    for doc in pages:
        text = doc.page_content
        matches = list(re.finditer(section_pattern, text))
        if not matches:
            # No headers on this page; keep entire page as one chunk
            section_chunks.append(Document(page_content=text, metadata=dict(doc.metadata)))
            continue

        starts = [m.start() for m in matches] + [len(text)]
        for i in range(len(starts) - 1):
            chunk_text = text[starts[i]:starts[i + 1]].strip()
            if not chunk_text:
                continue
            new_meta = dict(doc.metadata)
            new_meta["section_index"] = i + 1
            section_chunks.append(Document(page_content=chunk_text, metadata=new_meta))

    return section_chunks


def recursive_chunking(section_chunks: list[Document]) -> list[Document]:
    """
    For any section chunk > 1200 characters, further split into ~1200-char chunks
    with ~200-char overlap, respecting paragraph breaks where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1200,
        chunk_overlap=200,
        length_function=lambda x: len(x),
    )

    final_chunks = []
    for sec in section_chunks:
        content = sec.page_content
        if len(content) <= 1200:
            # No further splitting needed
            final_chunks.append(sec)
        else:
            sub_texts = splitter.split_text(content)
            for idx, txt in enumerate(sub_texts):
                meta = dict(sec.metadata)
                meta["subchunk_index"] = idx + 1
                final_chunks.append(Document(page_content=txt, metadata=meta))
    return final_chunks


# ─── MAIN ORCHESTRATION ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"📄 Loading and processing PDFs from {DATA_DIR}…")
    raw_pages = load_and_process_all_pdfs(DATA_DIR)
    print(f"   • Total pages processed: {len(raw_pages)}")

    print("🔍 Splitting pages by numbered section headers…")
    section_chunks = split_by_section_headers(raw_pages)
    print(f"   • Total section-level chunks: {len(section_chunks)}")

    print("✂️ Recursively chunking large sections (>1200 chars)…")
    final_chunks = recursive_chunking(section_chunks)
    print(f"   • Total final chunks: {len(final_chunks)}")

    # Persist chunk list to disk for reuse during embedding
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(final_chunks, f)
    print(f"✅ All chunks saved to: {OUTPUT_PICKLE}")



embed.py#2
# embed.py

import os
import pickle
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Assume this script lives in the same “src/” directory as ingest.py,
# so BASE_DIR points to project root.
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_PICKLE = BASE_DIR / "data" / "chunked_docs.pkl"
FAISS_DIR = BASE_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # 1️⃣ Load pre-chunked Documents from the pickle generated by ingest.py
    if not CHUNK_PICKLE.exists():
        raise FileNotFoundError(
            f"❌ Chunk pickle not found at {CHUNK_PICKLE}. "
            "Run ingest.py first to generate chunked_docs.pkl."
        )

    with open(CHUNK_PICKLE, "rb") as f:
        all_chunks = pickle.load(f)

    print(f"🔍 Loaded {len(all_chunks)} pre-chunked Document objects.")

    # 2️⃣ Initialize the embedding model (must match retrieval side)
    print(f"⚙️ Initializing embeddings with model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 3️⃣ Build a new FAISS index from documents
    print("⏳ Creating FAISS vector store from chunks...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    # 4️⃣ Ensure the output directory exists
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    # 5️⃣ Save the FAISS index locally
    vectorstore.save_local(str(FAISS_DIR))
    print(f"✅ FAISS vector store saved to: {FAISS_DIR}")


if __name__ == "__main__":
    main()


chat.py#2
# app.py

import os
import streamlit as st
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="HR Policy Chatbot", layout="wide")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"

# Keywords → PDF filename mapping for metadata filtering
POLICY_KEYWORDS = {
    "employee referral":    "employee_referral_policy.pdf",
    "leave policy":         "leave_policy.pdf",
    "total rewards":        "total_rewards_policy.pdf",
    "policy manual":        "policy_manual.pdf",
}


# ─── LOAD & SET UP RAG CHAIN ─────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    # 1️⃣ Initialize embeddings (must match index)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2️⃣ Load FAISS index
    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_DIR}. Run embed.py first."
        )
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 3️⃣ Create retriever (using simple similarity; adjust k/fetch_k as needed)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,        # return top 5
            "fetch_k": 10  # fetch 10, then pick 5
        }
    )

    # 4️⃣ Conversation memory for back-and-forth
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    # 5️⃣ Custom system prompt to enforce grounding and table interpretation
    system_prompt = """
You are an HR‐policy assistant. Use ONLY the provided context to answer.
- If the context contains a Markdown table, interpret “✓” as “Yes/Eligible” and “✗” as “No/Not Eligible.”
- If the answer cannot be found in the context, respond: “I’m sorry, I don’t have that information.”
"""

    # 6️⃣ Combine prompt template placing context + question under system instructions
    combine_prompt =  PromptTemplate(
        input_variables=["context", "question"],
        template=(
            system_prompt
            + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
    )

    # 7️⃣ Build the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model=LLM_MODEL),
        retriever=retriever,
        memory=memory,                        # retains chat history for follow-ups
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
    )
    return chain


qa_chain = load_chain()


# ─── STREAMLIT UI ────────────────────────────────────────────────────────────

st.title("🤖 HR Policy Chatbot")
st.markdown("Ask me anything about HR policies. You can also ask follow-up questions in context.")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("Your question:")

if query:
    # 1️⃣ Build metadata_filter if the query mentions a known policy keyword
    policy_filter = {}
    lowered = query.lower()
    for kw, filename in POLICY_KEYWORDS.items():
        if kw in lowered:
            policy_filter["source"] = filename
            break

    # 2️⃣ Call the RAG chain
    with st.spinner("Thinking..."):
        if policy_filter:
            result = qa_chain({"question": query, "metadata_filter": policy_filter})
        else:
            result = qa_chain({"question": query})

    answer = result["answer"]
    source_docs = result["source_documents"]

    # 3️⃣ Append to conversation history
    st.session_state.history.append((query, answer))

    # 4️⃣ Display the answer
    st.subheader("📌 Answer:")
    st.write(answer)

    # 5️⃣ Show retrieved source chunks
    with st.expander("📄 Source Chunks"):
        for doc in source_docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            section_idx = doc.metadata.get("section_index", "")
            subchunk_idx = doc.metadata.get("subchunk_index", "")
            st.markdown(
                f"- **File:** {src} | **Page:** {page} "
                f"{f'| Section: {section_idx}' if section_idx else ''} "
                f"{f'| Subchunk: {subchunk_idx}' if subchunk_idx else ''}"
            )
            snippet = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"    > {snippet[:200]}…")

# Sidebar: conversation history
st.sidebar.markdown("### 🕑 Conversation History")
for user_q, bot_a in st.session_state.history:
    st.sidebar.markdown(f"**You:** {user_q}")
    st.sidebar.markdown(f"**Bot:** {bot_a}")




import os
import json
import streamlit as st
from pathlib import Path

# ─── OCR LIBRARY: We now use EasyOCR instead of Tesseract ─────────────────────
import easyocr
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="HR Policy Chatbot", layout="wide")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"
HISTORY_FILE = BASE_DIR / "data" / "chat_history.json"

POLICY_KEYWORDS = {
    "employee referral":    "employee_referral_policy.pdf",
    "leave policy":         "leave_policy.pdf",
    "total rewards":        "total_rewards_policy.pdf",
    "policy manual":        "policy_manual.pdf",
}

# ─── LOAD PERSISTED HISTORY ──────────────────────────────────────────────────
if "history" not in st.session_state:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                st.session_state.history = json.load(f)
            except json.JSONDecodeError:
                st.session_state.history = []
    else:
        st.session_state.history = []


# ─── LOAD & SET UP RAG CHAIN ─────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_DIR}. Run embed.py first."
        )
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "fetch_k": 10})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    system_prompt = """
You are an HR‐policy assistant. Use ONLY the provided context to answer.
- If the context contains a Markdown table, interpret “✓” as “Yes/Eligible” and “✗” as “No/Not Eligible.”
- If the answer cannot be found in the context, respond: “I’m sorry, I don’t have that information.”
"""

    combine_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            system_prompt
            + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model=LLM_MODEL),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
    )
    return chain

qa_chain = load_chain()

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask me anything about HR policies. You can also ask follow-up questions in context.")

# ─── NEW: Image uploader + EasyOCR ───────────────────────────────────────────
uploaded_file = st.file_uploader("Upload an image (PNG/JPG):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Convert to RGB array
    img_array = __import__('numpy').array(image.convert("RGB"))

    # Create & cache the EasyOCR reader (only loads once per session)
    @st.cache_resource(show_spinner=False)
    def get_easyocr_reader():
        return easyocr.Reader(['en'], gpu=False)

    reader = get_easyocr_reader()
    ocr_result = reader.readtext(img_array)

    # Combine all detected text segments
    ocr_text = " ".join([segment[1] for segment in ocr_result])
    st.text_area("📝 Extracted text from image:", ocr_text, height=200)

    if ocr_text.strip():
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(x),
        )
        ocr_subtexts = splitter.split_text(ocr_text)

        ocr_docs = []
        for idx, chunk in enumerate(ocr_subtexts):
            meta = {"source": uploaded_file.name, "ocr_chunk": idx + 1}
            ocr_docs.append(Document(page_content=chunk, metadata=meta))

        qa_chain.retriever.vectorstore.add_documents(ocr_docs)

# ─── USER INPUT & QA LOGIC ───────────────────────────────────────────────────
query = st.text_input("Your question:")

if query:
    policy_filter = {}
    lowered = query.lower()
    for kw, filename in POLICY_KEYWORDS.items():
        if kw in lowered:
            policy_filter["source"] = filename
            break

    with st.spinner("Thinking..."):
        if policy_filter:
            result = qa_chain({"question": query, "metadata_filter": policy_filter})
        else:
            result = qa_chain({"question": query})

    answer = result["answer"]
    source_docs = result["source_documents"]

    entry = {"question": query, "answer": answer}
    st.session_state.history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, indent=2)

    st.subheader("📌 Answer:")
    st.write(answer)

    with st.expander("📄 Source Chunks"):
        for doc in source_docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            section_idx = doc.metadata.get("section_index", "")
            subchunk_idx = doc.metadata.get("subchunk_index", "")
            st.markdown(
                f"- **File:** {src} | **Page:** {page} "
                f"{f'| Section: {section_idx}' if section_idx else ''} "
                f"{f'| Subchunk: {subchunk_idx}' if subchunk_idx else ''}"
            )
            snippet = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"    > {snippet[:200]}…")

# ─── SIDEBAR: Conversation History (latest on top, question as expander) ────
st.sidebar.markdown("### 🕑 Conversation History")
for entry in reversed(st.session_state.history):
    user_q = entry["question"]
    bot_a   = entry["answer"]
    with st.sidebar.expander(user_q):
        st.write(bot_a)







history +2 step rag
import os                                   # unchanged
import json                                 # unchanged
import streamlit as st                      # unchanged
from pathlib import Path                    # unchanged

# ─── NEW & CHANGED IMPORTS ───────────────────────────────────────────────────
import numpy as np                          # new addition: for cosine similarity

from langchain.schema import Document        # new addition: to build source_docs manually
from langchain.text_splitter import RecursiveCharacterTextSplitter  # unchanged

from langchain_community.embeddings import HuggingFaceEmbeddings  # unchanged
from langchain_community.vectorstores import FAISS                 # unchanged
from langchain_community.llms import Ollama                        # unchanged
from langchain.memory import ConversationBufferMemory              # unchanged (optional)
from langchain.chains import ConversationalRetrievalChain          # unchanged (we won’t use its retrieval in two-stage)
from langchain.prompts import PromptTemplate                       # unchanged

st.set_page_config(page_title="HR Policy Chatbot", layout="wide")  # unchanged

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent                  # unchanged
FAISS_DIR = BASE_DIR / "vectorstore"                                # unchanged
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # unchanged
LLM_MODEL = "gemma3:4b"                                             # unchanged

# Where to keep the chat history
HISTORY_FILE = BASE_DIR / "data" / "chat_history.json"             # unchanged

# Keywords → PDF filename mapping for metadata filtering
POLICY_KEYWORDS = {                                                 # unchanged
    "employee referral":    "employee_referral_policy.pdf",
    "leave policy":         "leave_policy.pdf",
    "total rewards":        "total_rewards_policy.pdf",
    "policy manual":        "policy_manual.pdf",
}


# ─── LOAD PERSISTED HISTORY ──────────────────────────────────────────────────
if "history" not in st.session_state:                               # unchanged
    if HISTORY_FILE.exists():                                        # unchanged
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:         # unchanged
            try:                                                     # unchanged
                st.session_state.history = json.load(f)              # unchanged
            except json.JSONDecodeError:                             # unchanged
                st.session_state.history = []                        # unchanged
    else:                                                            # unchanged
        st.session_state.history = []                                 # unchanged


# ─── ORIGINAL LOAD & SET UP RAG CHAIN (commented out; two-stage will replace retrieval) ────
# @st.cache_resource
# def load_chain():
#     # 1️⃣ Initialize embeddings (must match index)
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#
#     # 2️⃣ Load FAISS index
#     if not FAISS_DIR.exists():
#         raise FileNotFoundError(
#             f"FAISS index not found at {FAISS_DIR}. Run embed.py first."
#         )
#     db = FAISS.load_local(
#         str(FAISS_DIR),
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#
#     # 3️⃣ Create retriever
#     retriever = db.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5, "fetch_k": 10}
#     )
#
#     # 4️⃣ Conversation memory (only in-memory; chat_history persists separately)
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         input_key="question",
#         output_key="answer",
#         return_messages=True
#     )
#
#     # 5️⃣ Custom system prompt
#     system_prompt = """
# You are an HR‐policy assistant. Use ONLY the provided context to answer.
# - If the context contains a Markdown table, interpret “✓” as “Yes/Eligible” and “✗” as “No/Not Eligible.”
# - If the answer cannot be found in the context, respond: “I’m sorry, I don’t have that information.”
# """
#
#     # 6️⃣ Combine prompt template
#     combine_prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=(
#             system_prompt
#             + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
#         ),
#     )
#
#     # 7️⃣ Build the ConversationalRetrievalChain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=Ollama(model=LLM_MODEL),
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=True,
#         combine_docs_chain_kwargs={"prompt": combine_prompt},
#     )
#     return chain
#
# qa_chain = load_chain()


# ─── NEW: LOAD FAISS + MINI‐LM EMBEDDINGS + RECONSTRUCT ALL VECTORS ─────────────
@st.cache_resource
def load_resources():
    # 1️⃣ Initialize MiniLM embeddings (for stage 1 filtering)
    miniLM = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2️⃣ Load FAISS index
    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_DIR}. Run embed.py first."
        )
    faiss_db = FAISS.load_local(
        str(FAISS_DIR),
        miniLM,  # use MiniLM for embedding interface
        allow_dangerous_deserialization=True
    )

    # 3️⃣ Reconstruct all stored embeddings into a NumPy array
    total_vectors = faiss_db._faiss_index.ntotal
    all_emb_matrix = faiss_db._faiss_index.reconstruct_n(0, total_vectors)  # shape: (N, dim)

    # 4️⃣ Extract parallel lists: chunk texts and metadata
    all_chunks_texts = faiss_db._texts.copy()       # list of chunk strings
    all_chunks_metas  = faiss_db._metadatas.copy()  # list of metadata dicts

    return faiss_db, miniLM, all_chunks_texts, all_chunks_metas, all_emb_matrix

# Invoke loader at top level
faiss_db, miniLM, all_chunks_texts, all_chunks_metas, all_embeddings = load_resources()


# ─── NEW: TWO‐STAGE RAG HELPER FUNCTION ───────────────────────────────────────
def answer_query_two_stage(query: str, miniLM, texts, metas, embeddings):
    # 1️⃣ Compute query embedding (MiniLM)
    q_emb = miniLM.embed_query(query)            # returns list[float]
    q_vec = np.array(q_emb, dtype="float32")      # shape: (dim,)

    # 2️⃣ Cosine similarity vs. all stored embeddings
    q_norm = q_vec / np.linalg.norm(q_vec)
    emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # (N, dim) normalized
    sims = emb_norms @ q_norm                   # shape: (N,)

    # 3️⃣ Select top_k indices
    top_k = 2                                   # pick top 2 chunks
    idx_sorted = np.argsort(sims)[-top_k:][::-1]  # descending order

    # 4️⃣ Build condensed context string
    selected_texts = [texts[i] for i in idx_sorted]
    selected_metas  = [metas[i]  for i in idx_sorted]
    condensed_context = "\n---\n".join(selected_texts)

    # 5️⃣ Construct system prompt + context + question
    system_prompt = """
You are an HR‐policy assistant. Use ONLY the provided context to answer.
- If the context contains a Markdown table, interpret “✓” as “Yes/Eligible” and “✗” as “No/Not Eligible.”
- If the answer cannot be found in the context, respond: “I’m sorry, I don’t have that information.”
"""
    prompt = f"""{system_prompt}

Context:
{condensed_context}

Question:
{query}

Answer:
"""
    # Call Gemma 4B once
    llm = Ollama(model=LLM_MODEL)
    answer = llm(prompt)

    # 6️⃣ Prepare source_docs list in same structure
    source_docs = []
    for idx in idx_sorted:
        doc = Document(page_content=texts[idx], metadata=metas[idx])
        source_docs.append(doc)

    return answer, source_docs


# ─── STREAMLIT UI ────────────────────────────────────────────────────────────
st.title("🤖 HR Policy Chatbot")                                           # unchanged
st.markdown("Ask me anything about HR policies. You can also ask follow-up questions in context.")  # unchanged

# User input
query = st.text_input("Your question:")                                   # unchanged

if query:
    # 1️⃣ Build metadata_filter if the query mentions a known policy keyword
    policy_filter = {}                                                    # unchanged
    lowered = query.lower()                                               # unchanged
    for kw, filename in POLICY_KEYWORDS.items():                          # unchanged
        if kw in lowered:                                                 # unchanged
            policy_filter["source"] = filename                            # unchanged
            break                                                         # unchanged

    # ─── TWO‐STAGE RAG INSTEAD OF qa_chain CALL ────────────────────────────
    with st.spinner("Thinking (two‐stage RAG)…"):                          # changed
        if policy_filter:
            # 1.a) Apply metadata filter to FAISS (LangChain’s pseudo‐API)
            # You may need to adapt to your installed LangChain version.
            filtered_db = faiss_db.filter(lambda m: m.get("source") == filename)  # new addition

            # Reconstruct filtered embeddings & texts & metas
            filtered_total = filtered_db._faiss_index.ntotal                           # new addition
            emb_matrix_filt = filtered_db._faiss_index.reconstruct_n(0, filtered_total)  # new addition
            texts_filt = filtered_db._texts.copy()                                     # new addition
            metas_filt = filtered_db._metadatas.copy()                                  # new addition

            answer, source_docs = answer_query_two_stage(
                query,
                miniLM,
                texts_filt,
                metas_filt,
                emb_matrix_filt
            )                                                                         # new addition
        else:
            # No metadata filter → use full index
            answer, source_docs = answer_query_two_stage(
                query,
                miniLM,
                all_chunks_texts,
                all_chunks_metas,
                all_embeddings
            )                                                                         # new addition

    # 2️⃣ Persist the new Q/A pair to session state and to disk
    entry = {"question": query, "answer": answer}                               # unchanged
    st.session_state.history.append(entry)                                      # unchanged
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:                         # unchanged
        json.dump(st.session_state.history, f, indent=2)                         # unchanged

    # 3️⃣ Display the answer
    st.subheader("📌 Answer:")                                                   # unchanged
    st.write(answer)                                                            # unchanged

    # 4️⃣ Show retrieved source chunks
    with st.expander("📄 Source Chunks"):                                        # unchanged
        for doc in source_docs:                                                 # changed: iterating source_docs from two-stage
            src = doc.metadata.get("source", "unknown")                          # unchanged
            page = doc.metadata.get("page", "n/a")                               # unchanged
            section_idx = doc.metadata.get("section_index", "")                  # unchanged
            subchunk_idx = doc.metadata.get("subchunk_index", "")                # unchanged
            st.markdown(                                                         # unchanged
                f"- **File:** {src} | **Page:** {page} "
                f"{f'| Section: {section_idx}' if section_idx else ''} "
                f"{f'| Subchunk: {subchunk_idx}' if subchunk_idx else ''}"
            )
            snippet = doc.page_content.strip().replace("\n", " ")                # unchanged
            st.markdown(f"    > {snippet[:200]}…")                                 # unchanged


# ─── SIDEBAR: Conversation History ───────────────────────────────────────────
st.sidebar.markdown("### 🕑 Conversation History")                            # unchanged
for entry in reversed(st.session_state.history):                               # unchanged
    user_q = entry["question"]                                                # unchanged
    bot_a   = entry["answer"]                                                  # unchanged
    with st.sidebar.expander(user_q):                                          # unchanged
        st.write(bot_a)                                                         # unchanged
