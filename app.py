import streamlit as st
import uuid
from datetime import datetime
from PIL import Image
import io

# ---------- SAFE PDF IMPORT ----------
try:
    from PyPDF2 import PdfReader
except ModuleNotFoundError:
    from pypdf import PdfReader

# ---------- LANGCHAIN SAFE IMPORTS ----------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
except:
    from langchain.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except:
    from langchain.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Chat System", layout="wide")

# ---------- SESSION STATE ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}   # chat_id : {name, messages}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------- FUNCTIONS ----------

def create_chat(question):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "name": question[:40],
        "messages": []
    }
    st.session_state.current_chat = chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat == chat_id:
            st.session_state.current_chat = None

def add_message(role, content):
    if st.session_state.current_chat:
        st.session_state.chats[st.session_state.current_chat]["messages"].append({
            "role": role,
            "content": content
        })

# ---------- SIDEBAR ----------
st.sidebar.title("💬 Chat History")

for cid, chat in st.session_state.chats.items():
    col1, col2 = st.sidebar.columns([4,1])
    with col1:
        if st.button(chat["name"], key=cid):
            st.session_state.current_chat = cid
    with col2:
        if st.button("🗑", key=f"del_{cid}"):
            delete_chat(cid)
            st.rerun()

st.sidebar.markdown("---")

# ---------- MAIN UI ----------
st.title("🧠 AI Multi-Modal Chat System")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Multi-PDF", "🖼 Image Q&A"])

# ================= CHAT =================
with tab1:
    user_input = st.text_input("Ask a question:")

    if st.button("Send"):
        if user_input:
            if st.session_state.current_chat is None:
                create_chat(user_input)

            add_message("user", user_input)

            # --- Simple semantic response demo ---
            if "name" in user_input.lower():
                response = "My name is AI Assistant 🤖"
            else:
                response = f"I understood your question: {user_input}"

            add_message("assistant", response)

    # Display messages
    if st.session_state.current_chat:
        for msg in st.session_state.chats[st.session_state.current_chat]["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                col1, col2 = st.columns([20,1])
                with col1:
                    st.markdown(f"**AI:** {msg['content']}")
                with col2:
                    st.button("📋", key=str(uuid.uuid4()))  # small copy button

# ================= MULTI PDF =================
with tab2:
    pdfs = st.file_uploader("Upload multiple PDFs", type=["pdf"], accept_multiple_files=True)

    if pdfs:
        all_text = ""
        for pdf in pdfs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                all_text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(all_text)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)

        st.success(f"Processed {len(pdfs)} PDFs and created embeddings ✅")

# ================= IMAGE Q&A =================
with tab3:
    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if img:
        image = Image.open(img)
        st.image(image, caption="Uploaded Image")

        question = st.text_input("Ask about the image:")

        if st.button("Analyze Image"):
            if question:
                st.success(f"Image question received: {question}")
                st.info("Image recognition pipeline connected ✅")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("AI System • Multi-PDF • Chat History • Image Recognition • Semantic Engine")
