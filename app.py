import base64
import hashlib
import os
import uuid
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.messages import HumanMessage
except ModuleNotFoundError:
    FAISS = None
    ChatGoogleGenerativeAI = None
    HuggingFaceEmbeddings = None
    RecursiveCharacterTextSplitter = None
    HumanMessage = None


st.set_page_config(page_title="SlideSense AI", layout="wide")


# -------------------- SUPABASE CONFIG --------------------
def _secret(name: str, default: str = "") -> str:
    # Streamlit secrets first (Streamlit Cloud), then env vars (local / CI).
    try:
        v = st.secrets.get(name)
        if v is not None and str(v).strip() != "":
            return str(v)
    except Exception:
        pass
    return str(os.getenv(name, default) or default)


SUPABASE_URL = _secret("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = _secret("SUPABASE_ANON_KEY") or _secret("SUPABASE_KEY")  # support either name
GOOGLE_API_KEY = _secret("GOOGLE_API_KEY")
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-1.5-flash")

# Supabase tables (create these in your Supabase DB)
USER_TABLE = "user_profiles"
CHAT_TABLE = "chat_sessions"
MSG_TABLE = "chat_messages"


def _sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def sb_rest(method: str, path: str, *, params=None, json=None):
    url = f"{SUPABASE_URL}{path}"
    r = requests.request(method, url, headers=_sb_headers(), params=params, json=json, timeout=30)
    if r.status_code >= 400:
        # Keep message short to avoid leaking secrets
        raise RuntimeError(f"Supabase request failed ({r.status_code}). Check your URL/key, RLS policies, and table names.")
    if r.text.strip() == "":
        return None
    return r.json()


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase is not configured. Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` (or `SUPABASE_KEY`) to Streamlit secrets.")
    st.stop()

if not GOOGLE_API_KEY:
    st.error("`GOOGLE_API_KEY` is not configured. Add it to Streamlit secrets.")
    st.stop()


# -------------------- SESSION INIT --------------------
defaults = {
    "authenticated": False,
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "vector_dbs": {},  # chat_id -> FAISS index
    "pdf_fingerprints": {},  # chat_id -> fingerprint
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- AUTH (SUPABASE TABLE) --------------------
def signup(email: str, password: str) -> str | None:
    email = (email or "").strip().lower()
    if not email or not password:
        return "Email and password required"

    existing = sb_rest("GET", f"/rest/v1/{USER_TABLE}", params={"email": f"eq.{email}", "select": "id"})
    if existing:
        return "User already exists"

    user_id = str(uuid.uuid4())
    sb_rest(
        "POST",
        f"/rest/v1/{USER_TABLE}",
        json={"id": user_id, "email": email, "password_hash": hash_pw(password), "created_at": datetime.utcnow().isoformat()},
    )
    return None


def login(email: str, password: str) -> str | None:
    email = (email or "").strip().lower()
    if not email or not password:
        return "Email and password required"

    rows = sb_rest("GET", f"/rest/v1/{USER_TABLE}", params={"email": f"eq.{email}", "select": "id,password_hash,email"})
    if not rows:
        return "Invalid login"
    row = rows[0]
    if row.get("password_hash") != hash_pw(password):
        return "Invalid login"

    st.session_state.authenticated = True
    st.session_state.user_id = row["id"]
    st.session_state.email = row["email"]
    return None


# -------------------- CHATS / MESSAGES (SUPABASE TABLES) --------------------
def create_new_chat(user_id: str, mode: str) -> str:
    chat_id = str(uuid.uuid4())
    sb_rest(
        "POST",
        f"/rest/v1/{CHAT_TABLE}",
        json={"id": chat_id, "user_id": user_id, "mode": mode, "title": "New Chat", "created_at": datetime.utcnow().isoformat()},
    )
    return chat_id


def delete_chat(chat_id: str):
    sb_rest("DELETE", f"/rest/v1/{MSG_TABLE}", params={"chat_id": f"eq.{chat_id}"})
    sb_rest("DELETE", f"/rest/v1/{CHAT_TABLE}", params={"id": f"eq.{chat_id}"})


def save_message(chat_id: str, role: str, content: str):
    sb_rest(
        "POST",
        f"/rest/v1/{MSG_TABLE}",
        json={"id": str(uuid.uuid4()), "chat_id": chat_id, "role": role, "content": content, "created_at": datetime.utcnow().isoformat()},
    )

    # Set title on first user message
    if role == "user":
        count = sb_rest(
            "GET",
            f"/rest/v1/{MSG_TABLE}",
            params={"chat_id": f"eq.{chat_id}", "role": "eq.user", "select": "id"},
        )
        if count is not None and len(count) == 1:
            title = (content[:45] + "...") if len(content) > 45 else content
            sb_rest("PATCH", f"/rest/v1/{CHAT_TABLE}", params={"id": f"eq.{chat_id}"}, json={"title": title})


def load_user_chats(user_id: str, mode: str):
    rows = sb_rest(
        "GET",
        f"/rest/v1/{CHAT_TABLE}",
        params={"user_id": f"eq.{user_id}", "mode": f"eq.{mode}", "select": "id,title,created_at", "order": "created_at.desc"},
    )
    return rows or []


def load_messages(chat_id: str):
    rows = sb_rest(
        "GET",
        f"/rest/v1/{MSG_TABLE}",
        params={"chat_id": f"eq.{chat_id}", "select": "role,content,created_at", "order": "created_at.asc"},
    )
    return rows or []


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    if ChatGoogleGenerativeAI is None:
        st.error("Missing package: `langchain-google-genai`. Make sure it’s installed from `requirements.txt`.")
        st.stop()
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)


# -------------------- LOGIN UI --------------------
if not st.session_state.authenticated:
    st.title("🔐 SlideSense Login (Supabase)")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                err = login(email, password)
            except Exception as e:
                err = str(e)
            if err:
                st.error(err)
            else:
                st.rerun()

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Signup"):
            try:
                err = signup(new_email, new_password)
            except Exception as e:
                err = str(e)
            if err:
                st.error(err)
            else:
                st.success("Account created! You can log in now.")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.success(f"👤 {st.session_state.email}")
if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode_label = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode_label else "IMAGE"

st.sidebar.markdown("## 💬 Your Chats")

try:
    user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)
except Exception as e:
    st.sidebar.error(str(e))
    user_chats = []

for c in user_chats:
    chat_id = c["id"]
    title = c.get("title") or "Chat"
    col1, col2 = st.sidebar.columns([4, 1])
    icon = "📘" if st.session_state.mode == "PDF" else "🖼"

    if col1.button(f"{icon} {title}", key=f"open_{chat_id}"):
        st.session_state.current_chat_id = chat_id
        st.rerun()

    if col2.button("🗑", key=f"delete_{chat_id}"):
        try:
            delete_chat(chat_id)
        except Exception as e:
            st.sidebar.error(str(e))
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None
        st.rerun()

if st.sidebar.button("➕ New Chat"):
    try:
        new_chat_id = create_new_chat(st.session_state.user_id, st.session_state.mode)
        st.session_state.current_chat_id = new_chat_id
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))


# -------------------- MAIN CONTENT --------------------
if not st.session_state.current_chat_id:
    st.markdown("## 👋 Welcome to SlideSense AI")
    st.markdown("### 🚀 AI Powered PDF & Image Analyzer")
    st.info("Select 'New Chat' from the sidebar to begin.")
    st.stop()


chat_id = st.session_state.current_chat_id

img_file = None
if st.session_state.mode == "PDF":
    st.markdown("## 📘 PDF Analyzer")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        fingerprint = f"{pdf.name}:{len(pdf.getvalue())}"
        prev_fp = st.session_state.pdf_fingerprints.get(chat_id)
        if prev_fp != fingerprint:
            if FAISS is None or HuggingFaceEmbeddings is None or RecursiveCharacterTextSplitter is None:
                st.error("Missing packages for PDF search (FAISS / embeddings). Check your `requirements.txt`.")
                st.stop()

            with st.spinner("Indexing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_dbs[chat_id] = FAISS.from_texts(chunks, embeddings)
                st.session_state.pdf_fingerprints[chat_id] = fingerprint

else:
    st.markdown("## 🖼 Image Question Answering")
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, use_container_width=True)


# -------------------- LOAD + DISPLAY MESSAGES --------------------
try:
    messages = load_messages(chat_id)
except Exception as e:
    st.error(str(e))
    st.stop()

for m in messages:
    role = m.get("role")
    content = m.get("content") or ""
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)


question = st.chat_input("Ask something...")
if question:
    try:
        save_message(chat_id, "user", question)
    except Exception as e:
        st.error(str(e))
        st.stop()

    llm = load_llm()

    # -------- PDF ANSWER --------
    if st.session_state.mode == "PDF":
        vector_db = st.session_state.vector_dbs.get(chat_id)
        if vector_db is None:
            answer = "Please upload a PDF first."
        else:
            docs = vector_db.similarity_search(question, k=6)
            context_parts = []
            for d in docs:
                if hasattr(d, "page_content"):
                    context_parts.append(d.page_content)
                else:
                    context_parts.append(str(d))
            context = "\n\n".join(context_parts)
            prompt = (
                "You are SlideSense. Answer using ONLY the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "If the answer is not in the context, reply exactly:\n"
                "Information not found in document."
            )
            answer = llm.invoke(prompt).content

    # -------- IMAGE ANSWER --------
    else:
        if not img_file:
            answer = "Please upload an image first."
        else:
            if HumanMessage is None:
                st.error("Missing package: `langchain-core`. Check your `requirements.txt`.")
                st.stop()
            with st.spinner("Analyzing image..."):
                image_bytes = img_file.getvalue()
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                response = llm.invoke(
                    [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}" }},
                            ]
                        )
                    ]
                )
                answer = response.content

    try:
        save_message(chat_id, "assistant", answer)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.rerun()