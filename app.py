
import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components

from PyPDF2 import PdfReader
from PIL import Image
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()


# -------------------- AUTH HELPERS (Supabase HTTP) --------------------
def _auth_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }


def _rest_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=10, **kwargs)


def set_session(user: Dict[str, Any]) -> None:
    st.session_state["session"] = {
        "user": user,
    }


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["user"]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."

    # Check if username already exists
    resp = _rest_request(
        "GET",
        f"/rest/v1/{USER_PROFILES_TABLE}"
        f"?select=username&username=eq.{username}",
    )
    if resp.status_code >= 400:
        return resp.text

    existing = resp.json()
    if existing:
        return "Username already exists."

    payload = {
        "username": username,
        "password_hash": _hash_password(password),
    }

    resp = _rest_request(
        "POST",
        f"/rest/v1/{USER_PROFILES_TABLE}",
        json=payload,
    )
    if resp.status_code >= 400:
        return resp.text
    return None


def sign_in(username: str, password: str) -> Optional[str]:
    resp = _rest_request(
        "GET",
        f"/rest/v1/{USER_PROFILES_TABLE}"
        f"?select=id,username,password_hash&username=eq.{username}",
    )
    if resp.status_code >= 400:
        return resp.text

    rows = resp.json()
    if not rows:
        return "Invalid username or password."

    row = rows[0]
    if row.get("password_hash") != _hash_password(password):
        return "Invalid username or password."

    user = {
        "id": row.get("id"),
        "username": row["username"],
    }

    set_session(user)
    return None


def sign_out() -> None:
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


def render_answer_with_copy(answer: str, key_suffix: str) -> None:
    """Render an answer with a Copy button."""
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText({safe_text});"
                style="margin-top:4px;padding:4px 10px;border-radius:4px;border:1px solid #ccc;cursor:pointer;">
            Copy
        </button>
        """,
        height=40,
    )


def get_display_name(user: Dict[str, Any]) -> str:
    """Return a display name for the current user."""
    if user.get("username"):
        return user["username"]
    meta = user.get("user_metadata") or {}
    if meta.get("username"):
        return meta["username"]
    email = user.get("email", "")
    return email.split("@")[0] if "@" in email else email


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
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

# -------------------- AUTH UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"),
            height=300
        )

    with col2:
        type_text("🔐 Welcome to SlideSense")

        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])

        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_in(username, password)
                    if err:
                        st.error(f"Login failed: {err}")
                    else:
                        st.rerun()

        with tab2:
            username = st.text_input("Username", key="signup_username")
            password = st.text_input(
                "Password (min 6 chars)", type="password", key="signup_password"
            )
            if st.button("Create Account"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_up(username, password)
                    if err:
                        st.error(f"Sign-up failed: {err}")
                    else:
                        st.success(
                            "Account created! You can now log in."
                        )

        with tab3:
            st.markdown("Continue without creating an account.")
            if st.button("Continue as guest"):
                st.session_state["guest"] = True
                st.rerun()


# -------------------- CHAT HISTORY PERSISTENCE (Supabase) --------------------
def load_chat_history_from_db() -> None:
    """Load PDF chat history for the current user from Supabase into session_state."""
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    username = user.get("username")
    if not username:
        return

    resp = _rest_request(
        "GET",
        "/rest/v1/chat_history"
        "?select=question,answer,mode,created_at"
        f"&username=eq.{username}"
        "&mode=eq.pdf"
        "&order=created_at.asc",
    )
    if resp.status_code >= 400:
        return

    rows: List[Dict[str, Any]] = resp.json()
    st.session_state.chat_history = [(r["question"], r["answer"]) for r in rows]
    st.session_state.history_loaded = True


def save_chat_to_db(question: str, answer: str) -> None:
    """Append a single PDF QA pair to Supabase chat_history."""
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    username = user.get("username")
    if not username:
        return

    payload = {
        "username": username,
        "mode": "pdf",
        "question": question,
        "answer": answer,
    }
    _rest_request(
        "POST",
        "/rest/v1/chat_history",
        json=payload,
    )

# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}
Convert into one clear sentence. No extra details.
"""
    return llm.invoke(prompt).content

# -------------------- AUTH CHECK --------------------
user = current_user()
if (not user) and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
user = current_user()
if user:
    display_name = get_display_name(user)
    label = f"Logged in as {display_name}"
elif st.session_state.get("guest"):
    label = "Logged in as Guest"
else:
    label = "Not logged in"

st.sidebar.success(label)

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.session_state["guest"] = False
    sign_out()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- SIDEBAR HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

if st.session_state.chat_history:
    # Latest first in the sidebar list
    items = list(reversed(list(enumerate(st.session_state.chat_history, start=1))))
    labels = [f"{idx}. {q[:40]}..." for idx, (q, _) in items]

    selected_label = st.sidebar.selectbox(
        "Select a message",
        options=labels,
        label_visibility="collapsed",
        key="history_select",
    )

    # Map selection back to the corresponding Q/A pair
    if selected_label:
        sel_idx = int(selected_label.split(".")[0])
        q_sel, a_sel = st.session_state.chat_history[sel_idx - 1]

        with st.sidebar.expander("Selected chat", expanded=True):
            st.markdown("**You**")
            st.write(q_sel)
            st.markdown("**Assistant**")
            st.write(a_sel)

    if st.sidebar.button("🧹 Clear History"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.sidebar.caption("No history yet")

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    pdf_col_anim, pdf_col_title = st.columns([1, 3])

    with pdf_col_anim:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
            height=180,
        )

    with pdf_col_title:
        st.markdown("## 📘 PDF Analyzer")
        st.caption("Upload a PDF and ask questions about its content.")

    st.divider()

    pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if (not st.session_state.get("history_loaded")) and (not st.session_state.get("guest")):
            load_chat_history_from_db()

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                page_texts = []

                for page_num, pdf_page in enumerate(reader.pages, start=1):
                    extracted = pdf_page.extract_text()
                    if extracted:
                        page_texts.append((page_num, extracted))

                if not page_texts:
                    st.error("No readable text found in PDF")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )

                all_chunks: List[str] = []
                metadatas: List[Dict[str, Any]] = []
                for page_num, page_text in page_texts:
                    page_chunks = splitter.split_text(page_text)
                    all_chunks.extend(page_chunks)
                    metadatas.extend([{"page": page_num} for _ in page_chunks])

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(
                    all_chunks, embeddings, metadatas=metadatas
                )

        user_q = st.chat_input("Ask a question about this PDF")

        if user_q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(user_q, k=5)

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
            res = chain.invoke({"context": docs, "question": user_q})

            if isinstance(res, dict):
                answer = res.get("output_text", "")
            else:
                answer = res

            st.session_state.chat_history.append((user_q, answer))
            save_chat_to_db(user_q, answer)

            # Store sources (pages and snippets) for the latest answer
            st.session_state["last_sources"] = [
                {
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
                }
                for d in docs
            ]

        # -------- CHAT DISPLAY (ChatGPT-style, latest on top) --------
        st.markdown("## 💬 Conversation")

        chat_container = st.container()
        with chat_container:
            for uq, ua in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(uq)
                with st.chat_message("assistant"):
                    render_answer_with_copy(ua, key_suffix=f"pdf_{uq}")

        # -------- SOURCES FOR LATEST ANSWER --------
        sources = st.session_state.get("last_sources") or []
        if sources:
            st.markdown("### 🔍 Sources used")
            for idx, src in enumerate(sources, start=1):
                page = src.get("page", "?")
                snippet = src.get("snippet", "")
                with st.expander(f"Source {idx} • Page {page}"):
                    st.write(snippet)

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    img_col_anim, img_col_title = st.columns([1, 3])

    with img_col_anim:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
            height=180,
        )

    with img_col_title:
        st.markdown("## 🖼 Image Q&A")
        st.caption("Upload an image and ask questions about it.")

    st.divider()

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analyzing image..."):
                ans = answer_image_question(img, question)
            st.success("Answer:")
            render_answer_with_copy(ans, key_suffix="img_answer")
