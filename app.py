import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components
from datetime import datetime, timezone

from PyPDF2 import PdfReader
from PIL import Image
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# ---- Streamlit Secrets (REQUIRED) ----
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

# ---- Tables ----
USER_TABLE = "user_profiles"
CHATS_TABLE = "chats"
MESSAGES_TABLE = "messages"
CHAT_META_TABLE = "chat_metadata"

# -------------------- SUPABASE HELPERS --------------------
def sb_headers():
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }

def sb_get(path, params=None):
    url = f"{SUPABASE_URL}{path}"
    r = requests.get(url, headers=sb_headers(), params=params, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json()

def sb_post(path, data):
    url = f"{SUPABASE_URL}{path}"
    r = requests.post(url, headers=sb_headers(), json=data, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json()

def sb_patch(path, data):
    url = f"{SUPABASE_URL}{path}"
    r = requests.patch(url, headers=sb_headers(), json=data, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json()

# -------------------- AUTH --------------------
def hash_pw(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def sign_up(username, password):
    rows = sb_get(f"/rest/v1/{USER_TABLE}", {
        "select": "id",
        "username": f"eq.{username}"
    })
    if rows:
        return "Username already exists"
    sb_post(f"/rest/v1/{USER_TABLE}", {
        "username": username,
        "password_hash": hash_pw(password),
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    return None

def sign_in(username, password):
    rows = sb_get(f"/rest/v1/{USER_TABLE}", {
        "select": "*",
        "username": f"eq.{username}"
    })
    if not rows:
        return None
    u = rows[0]
    if u["password_hash"] != hash_pw(password):
        return None
    return u

# -------------------- MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device

# -------------------- UTILS --------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def render_answer_with_copy(answer):
    st.markdown(answer)
    safe = json.dumps(answer)
    components.html(f"""
    <button onclick="navigator.clipboard.writeText({safe});"
    style="margin-top:6px;padding:6px 12px;border-radius:6px;border:1px solid #ccc;cursor:pointer;">
    Copy
    </button>
    """, height=40)

# -------------------- CHAT SYSTEM --------------------
def auto_title(q): 
    return q[:40]

def create_chat(user_id, mode, first_q):
    now = datetime.now(timezone.utc).isoformat()
    chat = sb_post(f"/rest/v1/{CHATS_TABLE}", {
        "user_id": user_id,
        "title": auto_title(first_q),
        "mode": mode,
        "pinned": False,   # ✅ correct column
        "created_at": now,
        "updated_at": now
    })[0]

    sb_post(f"/rest/v1/{CHAT_META_TABLE}", {
        "chat_id": chat["id"],
        "summary": "",
        "token_count": 0,
        "last_message_at": now,
        "last_message_preview": first_q[:120]
    })
    return chat

def fetch_user_chats(user_id):
    chats = sb_get(f"/rest/v1/{CHATS_TABLE}", {
        "select": "*",
        "user_id": f"eq.{user_id}",
        "order": "pinned.desc,updated_at.desc",
    })

    chats = [c for c in chats if c.get("deleted_at") is None]

    if not chats:
        return []

    ids = [c["id"] for c in chats]
    meta = sb_get(f"/rest/v1/{CHAT_META_TABLE}", {
        "select": "*",
        "chat_id": f"in.({','.join(ids)})"
    })

    meta_map = {m["chat_id"]: m for m in meta}

    for c in chats:
        c["meta"] = meta_map.get(c["id"], {})
    return chats

# -------------------- LOGIN UI --------------------
def login_ui():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"), height=320)

    with col2:
        st.markdown("## 🔐 SlideSense Login")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                user = sign_in(u,p)
                if not user:
                    st.error("Invalid credentials")
                else:
                    st.session_state.user = user
                    st.rerun()

        with tab2:
            u = st.text_input("New Username")
            p = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                err = sign_up(u,p)
                if err:
                    st.error(err)
                else:
                    st.success("Account created!")

# -------------------- SESSION --------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- AUTH CHECK --------------------
if not st.session_state.user:
    login_ui()
    st.stop()

user = st.session_state.user

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {user['username']}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- SIDEBAR CHATS --------------------
st.sidebar.markdown("### 💬 Chats")

chats = fetch_user_chats(user["id"])

for c in chats:
    label = "📌 " + c["title"] if c.get("pinned") else c["title"]
    if st.sidebar.button(label, key=c["id"]):
        st.session_state.active_chat = c["id"]

# ==================== PDF MODE ====================
if mode == "📘 PDF Analyzer":

    st.markdown("## 📘 PDF Analyzer")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = []
                for p in reader.pages:
                    if p.extract_text():
                        text.append(p.extract_text())

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_text("\n".join(text))

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.chat_input("Ask question from PDF")

        if q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
Context:
{context}

Question:
{q}

Rules:
- Answer only from document
- If not found say: Information not found in the document
"""

            ans = llm.invoke(prompt).content
            st.session_state.chat_history.append((q, ans))

        st.markdown("## 💬 Conversation")
        for q,a in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                render_answer_with_copy(a)

# ==================== IMAGE MODE ====================
if mode == "🖼 Image Q&A":

    st.markdown("## 🖼 Image Q&A")

    img_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        q = st.text_input("Ask question about image")

        if q:
            processor, model, device = load_blip()
            inputs = processor(img, q, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_length=10)
            short = processor.decode(out[0], skip_special_tokens=True)

            llm = load_llm()
            prompt = f"""
Question: {q}
Vision answer: {short}
Convert into one clear sentence.
"""
            ans = llm.invoke(prompt).content

            st.success("Answer:")
            render_answer_with_copy(ans)
