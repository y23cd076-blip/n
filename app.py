import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib, uuid
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components

from pypdf import PdfReader
from PIL import Image
import torch

# ================= FIXED LANGCHAIN IMPORTS =================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# ==========================================================

from transformers import BlipProcessor, BlipForQuestionAnswering

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
USER_PROFILES_TABLE = "user_profiles"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase configuration.")
    st.stop()

# -------------------- AUTH HELPERS --------------------
def _auth_headers() -> Dict[str, str]:
    return {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}

def _rest_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=10, **kwargs)

def set_session(user: Dict[str, Any]) -> None:
    st.session_state["session"] = {"user": user}

def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    return sess["user"] if sess else None

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."

    resp = _rest_request("GET", f"/rest/v1/{USER_PROFILES_TABLE}?select=username&username=eq.{username}")
    if resp.json():
        return "Username already exists."

    payload = {"username": username, "password_hash": _hash_password(password)}
    _rest_request("POST", f"/rest/v1/{USER_PROFILES_TABLE}", json=payload)
    return None

def sign_in(username: str, password: str) -> Optional[str]:
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

def render_answer_with_copy(answer: str):
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(
        f"""<button onclick="navigator.clipboard.writeText({safe_text});"
        style="margin-top:4px;padding:4px 10px;border-radius:4px;border:1px solid #ccc;cursor:pointer;">
        Copy</button>""",
        height=40,
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
if "chats" not in st.session_state:
    st.session_state.chats = {}   # chat_id -> {history, vector_db}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "guest" not in st.session_state:
    st.session_state.guest = False

# -------------------- CHAT FUNCTIONS --------------------
def create_chat():
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = {
        "history": [],
        "vector_db": None
    }
    st.session_state.current_chat = cid

# -------------------- AUTH UI --------------------
def login_ui():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"), height=300)

    with col2:
        st.markdown("## 🔐 Welcome to SlideSense")
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
            if st.button("Continue as guest"):
                st.session_state.guest = True
                st.rerun()

# -------------------- AUTH CHECK --------------------
user = current_user()
if (not user) and not st.session_state.guest:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {user['username']}" if user else "Guest")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    sign_out()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- CHAT SIDEBAR --------------------
st.sidebar.markdown("## 💬 Chats")

if st.sidebar.button("➕ New Chat"):
    create_chat()
    st.rerun()

for cid in st.session_state.chats:
    if st.sidebar.button(f"Chat {cid}", key=cid):
        st.session_state.current_chat = cid
        st.rerun()

if not st.session_state.current_chat:
    create_chat()

chat = st.session_state.chats[st.session_state.current_chat]

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    st.markdown("## 📘 PDF Analyzer")

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if pdfs and chat["vector_db"] is None:
        with st.spinner("Processing PDFs..."):
            all_chunks, metadatas = [], []
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

            for pdf in pdfs:
                reader = PdfReader(pdf)
                for i, page in enumerate(reader.pages, 1):
                    txt = page.extract_text()
                    if txt:
                        chunks = splitter.split_text(txt)
                        for ch in chunks:
                            all_chunks.append(ch)
                            metadatas.append({"page": i, "pdf": pdf.name})

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            chat["vector_db"] = FAISS.from_texts(all_chunks, embeddings, metadatas=metadatas)

    q = st.chat_input("Ask about PDFs...")

    if q and chat["vector_db"]:
        llm = load_llm()
        docs = chat["vector_db"].similarity_search(q, k=5)

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
        ans = res.get("output_text", "") if isinstance(res, dict) else res

        chat["history"].append(("user", q))
        chat["history"].append(("assistant", ans))

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    st.markdown("## 🖼 Image Q&A")

    img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        q = st.chat_input("Ask about image...")
        if q:
            processor, model, device = load_blip()
            inputs = processor(img, q, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=10)
            short = processor.decode(outputs[0], skip_special_tokens=True)

            llm = load_llm()
            refined = llm.invoke(f"Question:{q}\nVision:{short}\nMake one clean sentence.").content

            chat["history"].append(("user", q))
            chat["history"].append(("assistant", refined))

# ==================== CHAT DISPLAY ====================
st.markdown("## 💬 Conversation")

for role, msg in chat["history"]:
    with st.chat_message(role):
        if role == "assistant":
            render_answer_with_copy(msg)
        else:
            st.markdown(msg)
