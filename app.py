import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components

# ---------- PDF READER FIX ----------
try:
    from PyPDF2 import PdfReader
except ModuleNotFoundError:
    from pypdf import PdfReader

from PIL import Image
import torch

# ---------- LANGCHAIN FIXED IMPORTS ----------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
USER_PROFILES_TABLE = "user_profiles"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY")
    st.stop()

# -------------------- SUPABASE HELPERS --------------------
def _auth_headers():
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }

def _rest_request(method: str, path: str, **kwargs):
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=10, **kwargs)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def set_session(user):
    st.session_state["session"] = {"user": user}

def current_user():
    sess = st.session_state.get("session")
    return sess["user"] if sess else None

def sign_up(username, password):
    if len(password) < 6:
        return "Password must be at least 6 characters."
    resp = _rest_request("GET", f"/rest/v1/{USER_PROFILES_TABLE}?select=username&username=eq.{username}")
    if resp.json():
        return "Username already exists."
    payload = {"username": username, "password_hash": _hash_password(password)}
    _rest_request("POST", f"/rest/v1/{USER_PROFILES_TABLE}", json=payload)
    return None

def sign_in(username, password):
    resp = _rest_request("GET", f"/rest/v1/{USER_PROFILES_TABLE}?select=id,username,password_hash&username=eq.{username}")
    rows = resp.json()
    if not rows:
        return "Invalid username or password."
    row = rows[0]
    if row["password_hash"] != _hash_password(password):
        return "Invalid username or password."
    set_session({"id": row["id"], "username": row["username"]})
    return None

def sign_out():
    st.session_state.pop("session", None)

# -------------------- HELPERS --------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def type_text(text, speed=0.03):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)

# ---------- SMALL COPY BUTTON ----------
def render_answer_with_copy(answer: str):
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(
        f"""
        <style>
        .copy-btn {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #ccc;
            cursor: pointer;
            background: #f7f7f7;
        }}
        </style>
        <button class="copy-btn" onclick="navigator.clipboard.writeText({safe_text});">📋</button>
        """,
        height=28,
    )

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

# -------------------- SESSION DEFAULTS --------------------
defaults = {
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- LOGIN UI --------------------
def login_ui():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"), height=300)
    with col2:
        type_text("🔐 Welcome to SlideSense")
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                err = sign_in(u, p)
                if err: st.error(err)
                else: st.rerun()

        with tab2:
            u = st.text_input("Username", key="su")
            p = st.text_input("Password", type="password", key="sp")
            if st.button("Create Account"):
                err = sign_up(u, p)
                if err: st.error(err)
                else: st.success("Account created!")

        with tab3:
            if st.button("Continue as Guest"):
                st.session_state["guest"] = True
                st.rerun()

# -------------------- AUTH CHECK --------------------
user = current_user()
if not user and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {user['username'] if user else 'Guest'}")

if st.sidebar.button("Logout"):
    sign_out()
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- CHAT HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

def delete_chat_from_db(question):
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    username = user["username"]
    _rest_request(
        "DELETE",
        "/rest/v1/chat_history"
        f"?username=eq.{username}"
        f"&question=eq.{question}"
        "&mode=eq.pdf"
    )

if st.session_state.chat_history:
    items = list(reversed(st.session_state.chat_history))
    labels = [q for q, _ in items]

    selected = st.sidebar.selectbox("Chats", labels)

    if selected:
        idx = next(i for i,(q,_) in enumerate(st.session_state.chat_history) if q==selected)
        q,a = st.session_state.chat_history[idx]

        with st.sidebar.expander("Chat Preview", True):
            st.markdown("**You**")
            st.write(q)
            st.markdown("**Assistant**")
            st.write(a)

        if st.sidebar.button("🗑 Delete Chat"):
            delete_chat_from_db(q)
            del st.session_state.chat_history[idx]
            st.rerun()

# ==================== PDF MODE ====================
if mode == "📘 PDF Analyzer":

    st.title("📘 PDF Analyzer")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for p in reader.pages:
                    if p.extract_text():
                        text += p.extract_text()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.chat_input("Ask something about the PDF")

        if q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": q})

            answer = res.get("output_text","") if isinstance(res,dict) else res
            st.session_state.chat_history.append((q, answer))

        st.markdown("## 💬 Conversation")
        for uq,ua in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(uq)
            with st.chat_message("assistant"):
                render_answer_with_copy(ua)

# ==================== IMAGE MODE ====================
if mode == "🖼 Image Q&A":
    st.title("🖼 Image Q&A")

    img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        q = st.text_input("Ask a question about the image")
        if q:
            processor, model, device = load_blip()
            inputs = processor(img, q, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=20)
            short = processor.decode(outputs[0], skip_special_tokens=True)

            llm = load_llm()
            final = llm.invoke(f"Convert into one clear sentence:\n{short}").content

            st.success("Answer:")
            render_answer_with_copy(final)
