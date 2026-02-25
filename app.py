import streamlit as st
import requests, os, hashlib
from typing import Dict, Any, List
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SlideSense", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# ---------------- SUPABASE HELPERS ----------------
def _auth_headers():
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }

def _rest_request(method: str, path: str, **kwargs):
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=15, **kwargs)

# ---------------- SESSION ----------------
defaults = {
    "current_chat_id": None,
    "current_pdf_id": None,
    "vector_db": None,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- LLM ----------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ---------------- UTILS ----------------
def get_pdf_id(pdf, username):
    raw = f"{username}_{pdf.name}_{pdf.size}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ---------------- CHAT DB OPS ----------------
def create_new_chat(username, mode="pdf", pdf_id=None):
    payload = {
        "username": username,
        "mode": mode,
        "pdf_id": pdf_id,
        "title": "New Chat"
    }
    _rest_request("POST", "/rest/v1/chats", json=payload)

    res = _rest_request(
        "GET",
        f"/rest/v1/chats?username=eq.{username}&order=created_at.desc&limit=1"
    )
    return res.json()[0]["id"]

def load_user_chats(username, mode="pdf", pdf_id=None):
    q = f"/rest/v1/chats?username=eq.{username}&mode=eq.{mode}"
    if pdf_id:
        q += f"&pdf_id=eq.{pdf_id}"
    q += "&order=created_at.desc"
    r = _rest_request("GET", q)
    return r.json() if r.status_code < 400 else []

def load_chat_messages(chat_id):
    r = _rest_request("GET", f"/rest/v1/messages?chat_id=eq.{chat_id}&order=created_at.asc")
    return r.json() if r.status_code < 400 else []

def save_message(chat_id, role, content):
    payload = {
        "chat_id": chat_id,
        "role": role,
        "content": content
    }
    _rest_request("POST", "/rest/v1/messages", json=payload)

def delete_chat(chat_id):
    _rest_request("DELETE", f"/rest/v1/messages?chat_id=eq.{chat_id}")
    _rest_request("DELETE", f"/rest/v1/chats?id=eq.{chat_id}")

# ---------------- USER (SIMPLE DEMO AUTH) ----------------
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:
    st.title("Login")
    u = st.text_input("Username")
    if st.button("Enter"):
        st.session_state.username = u
        st.rerun()
    st.stop()

username = st.session_state.username

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(f"### 👤 {username}")
st.sidebar.divider()

st.sidebar.markdown("## 📘 PDFs")
pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")

if pdf:
    pdf_id = get_pdf_id(pdf, username)
    st.session_state.current_pdf_id = pdf_id

    chats = load_user_chats(username, "pdf", pdf_id)

    if not chats:
        st.session_state.current_chat_id = create_new_chat(username, "pdf", pdf_id)
    else:
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = chats[0]["id"]

# ---------------- PDF PROCESSING ----------------
if pdf and st.session_state.vector_db is None:
    reader = PdfReader(pdf)
    text = ""
    for p in reader.pages:
        if p.extract_text():
            text += p.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

# ---------------- SIDEBAR CHATS ----------------
if st.session_state.current_pdf_id:
    st.sidebar.divider()
    st.sidebar.markdown("## 💬 Chats")

    chat_list = load_user_chats(username, "pdf", st.session_state.current_pdf_id)

    if st.sidebar.button("➕ New Chat"):
        st.session_state.current_chat_id = create_new_chat(
            username, "pdf", st.session_state.current_pdf_id
        )
        st.rerun()

    for chat in chat_list:
        c1, c2 = st.sidebar.columns([5,1])
        with c1:
            if st.button(chat["title"], key=f"open_{chat['id']}"):
                st.session_state.current_chat_id = chat["id"]
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{chat['id']}"):
                delete_chat(chat["id"])
                if st.session_state.current_chat_id == chat["id"]:
                    st.session_state.current_chat_id = None
                st.rerun()

# ---------------- MAIN ----------------
st.title("📘 SlideSense PDF Chat")

if not pdf:
    st.info("Upload a PDF to start")
    st.stop()

# ---------------- LOAD CHAT ----------------
if st.session_state.current_chat_id:
    messages = load_chat_messages(st.session_state.current_chat_id)
else:
    messages = []

# ---------------- DISPLAY ----------------
for m in messages:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(m["content"])

# ---------------- INPUT ----------------
q = st.chat_input("Ask about this PDF")

if q:
    save_message(st.session_state.current_chat_id, "user", q)

    docs = st.session_state.vector_db.similarity_search(q, k=5)
    llm = load_llm()

    prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in document
""")

    chain = create_stuff_documents_chain(llm, prompt)
    res = chain.invoke({"context": docs, "question": q})

    if isinstance(res, dict):
        ans = res.get("output_text","")
    else:
        ans = res

    save_message(st.session_state.current_chat_id, "assistant", ans)
    st.rerun()
