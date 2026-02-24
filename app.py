import streamlit as st
import requests, os, time
from typing import Any, Dict, Optional, List

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

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 40%, #000000 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }
    .login-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 18px;
        padding: 1.75rem 1.5rem 1.5rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.25);
        backdrop-filter: blur(20px);
    }
    .login-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: space-between;
    }
    .stTextInput > label {
        font-weight: 500;
    }
    .stButton > button {
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        font-weight: 600;
    }
    .stChatMessage {
        max-width: 760px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- ENV --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. "
        "Set **SUPABASE_URL** and **SUPABASE_ANON_KEY** environment variables."
    )
    st.stop()

if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set — LLM features will fail.")

# -------------------- SESSION DEFAULTS --------------------
# Initialise BEFORE any function that reads session_state
_DEFAULTS: Dict[str, Any] = {
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
    "session": None,
    "pending_verification_email": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- AUTH HELPERS --------------------
def _base_headers() -> Dict[str, str]:
    """Headers for unauthenticated Supabase auth endpoints."""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }


def _authed_headers(access_token: str) -> Dict[str, str]:
    """Headers for REST requests made on behalf of a signed-in user."""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _auth_post(path: str, payload: dict) -> requests.Response:
    return requests.post(
        f"{SUPABASE_URL}{path}",
        headers=_base_headers(),
        json=payload,
        timeout=15,
    )


def _db_request(method: str, path: str, access_token: str, **kwargs) -> requests.Response:
    return requests.request(
        method,
        f"{SUPABASE_URL}{path}",
        headers=_authed_headers(access_token),
        timeout=15,
        **kwargs,
    )


def _parse_error(resp: requests.Response) -> str:
    """Return a human-readable error from a Supabase error response."""
    try:
        data = resp.json()
        return (
            data.get("error_description")
            or data.get("msg")
            or data.get("message")
            or data.get("error")
            or resp.text
        )
    except Exception:
        return resp.text


# -------------------- AUTH ACTIONS --------------------
def set_session(data: dict) -> None:
    st.session_state["session"] = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "user": data["user"],
    }


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    return sess["user"] if sess else None


def _current_token() -> Optional[str]:
    sess = st.session_state.get("session")
    return sess["access_token"] if sess else None


def sign_up(email: str, password: str) -> Optional[str]:
    """Register a new user. Returns error string or None on success."""
    if len(password) < 6:
        return "Password must be at least 6 characters."

    resp = _auth_post("/auth/v1/signup", {"email": email, "password": password})

    if resp.status_code >= 400:
        return _parse_error(resp)

    data = resp.json()
    # Supabase returns identities=[] when email already exists + email confirm on
    if isinstance(data.get("identities"), list) and len(data["identities"]) == 0:
        return "This email is already registered. Please log in or reset your password."

    st.session_state["pending_verification_email"] = email
    return None


def resend_verification(email: str) -> Optional[str]:
    """Resend the confirmation email."""
    resp = _auth_post("/auth/v1/resend", {"type": "signup", "email": email})
    if resp.status_code >= 400:
        return _parse_error(resp)
    return None


def sign_in(email: str, password: str) -> Optional[str]:
    """Sign in. Returns error string or None on success."""
    resp = _auth_post(
        "/auth/v1/token?grant_type=password",
        {"email": email, "password": password},
    )

    if resp.status_code >= 400:
        err = _parse_error(resp)
        if "email not confirmed" in err.lower():
            st.session_state["pending_verification_email"] = email
            return (
                "Email not verified yet. Check your inbox and click the link, "
                "or use the **Resend verification email** button."
            )
        return err

    data = resp.json()
    if not data.get("access_token") or not data.get("user"):
        return "Unexpected response from Supabase. Please try again."

    set_session(data)
    st.session_state["pending_verification_email"] = None
    return None


def refresh_session() -> bool:
    """Silently refresh the access token. Signs out on failure."""
    sess = st.session_state.get("session")
    if not sess or not sess.get("refresh_token"):
        return False

    resp = _auth_post(
        "/auth/v1/token?grant_type=refresh_token",
        {"refresh_token": sess["refresh_token"]},
    )
    if resp.status_code >= 400:
        sign_out()
        return False

    data = resp.json()
    if not data.get("access_token") or not data.get("user"):
        sign_out()
        return False

    set_session(data)
    return True


def sign_out() -> None:
    token = _current_token()
    if token:
        try:
            requests.post(
                f"{SUPABASE_URL}/auth/v1/logout",
                headers=_authed_headers(token),
                timeout=5,
            )
        except Exception:
            pass
    st.session_state["session"] = None


def reset_password(email: str) -> Optional[str]:
    """Send a password-reset email."""
    resp = _auth_post("/auth/v1/recover", {"email": email})
    if resp.status_code >= 400:
        return _parse_error(resp)
    return None


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)


@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device


# -------------------- DB HELPERS --------------------
def _db_get_with_refresh(path: str) -> Optional[List[dict]]:
    token = _current_token()
    if not token:
        return None
    resp = _db_request("GET", path, access_token=token)
    if resp.status_code == 401:
        if not refresh_session():
            return None
        resp = _db_request("GET", path, access_token=_current_token())
    if resp.status_code >= 400:
        return None
    return resp.json()


def _db_post_with_refresh(path: str, payload: dict) -> bool:
    token = _current_token()
    if not token:
        return False
    resp = _db_request("POST", path, access_token=token, json=payload)
    if resp.status_code == 401:
        if not refresh_session():
            return False
        resp = _db_request("POST", path, access_token=_current_token(), json=payload)
    return resp.status_code < 400


# -------------------- CHAT PERSISTENCE --------------------
def load_chat_history_from_db() -> None:
    if st.session_state.get("guest") or st.session_state.get("history_loaded"):
        return
    user = current_user()
    if not user:
        return

    rows = _db_get_with_refresh(
        "/rest/v1/chat_history"
        "?select=question,answer,created_at"
        f"&user_id=eq.{user['id']}"
        "&mode=eq.pdf"
        "&order=created_at.asc"
    )
    # Mark loaded regardless — avoids hammering DB on every rerun
    st.session_state.history_loaded = True
    if rows:
        st.session_state.chat_history = [(r["question"], r["answer"]) for r in rows]


def save_chat_to_db(question: str, answer: str) -> None:
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    _db_post_with_refresh(
        "/rest/v1/chat_history",
        {"user_id": user["id"], "mode": "pdf", "question": question, "answer": answer},
    )


# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question: str) -> str:
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = (
        f"Question: {question}\n"
        f"Vision model answer: {short_answer}\n"
        "Rewrite as one clear, complete sentence. No extra commentary."
    )
    return llm.invoke(prompt).content


# -------------------- AUTH UI --------------------
def _maybe_show_resend() -> None:
    """If we have a pending verification email, show a resend button."""
    pending = st.session_state.get("pending_verification_email")
    if not pending:
        return
    st.info(f"Didn't receive the email? We can resend it to **{pending}**.")
    if st.button("📧 Resend verification email", key="btn_resend"):
        with st.spinner("Resending…"):
            err = resend_verification(pending)
        if err:
            st.error(f"Resend failed: {err}")
        else:
            st.success("Verification email resent! Check your inbox and spam folder.")


def login_ui() -> None:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-title">🔐 Welcome to SlideSense</div>'
        '<div class="login-subtitle">Sign in to analyse PDFs, chat with your documents, and ask questions about images.</div>',
        unsafe_allow_html=True,
    )

    tab_login, tab_signup, tab_guest, tab_reset = st.tabs(
        ["Login", "Sign Up", "Guest mode", "Forgot password"]
    )

    # ---- LOGIN ----
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="btn_login"):
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                with st.spinner("Signing in…"):
                    err = sign_in(email, password)
                if err:
                    st.error(err)
                    _maybe_show_resend()
                else:
                    st.rerun()

    # ---- SIGN UP ----
    with tab_signup:
        su_email = st.text_input("Email", key="signup_email")
        su_pw = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
        su_pw2 = st.text_input("Confirm password", type="password", key="signup_password2")

        if st.button("Create Account", key="btn_signup"):
            if not su_email or not su_pw or not su_pw2:
                st.warning("Please fill in all fields.")
            elif su_pw != su_pw2:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating account…"):
                    err = sign_up(su_email, su_pw)
                if err:
                    st.error(f"Sign-up failed: {err}")
                else:
                    st.success(
                        "✅ Account created! Check your inbox for a **verification email** "
                        "and click the link before logging in."
                    )
                    _maybe_show_resend()

    # ---- GUEST ----
    with tab_guest:
        st.markdown(
            "Continue without an account. "
            "Chat history won't be saved between sessions."
        )
        if st.button("Continue as Guest", key="btn_guest"):
            st.session_state["guest"] = True
            st.rerun()

    # ---- FORGOT PASSWORD ----
    with tab_reset:
        reset_email = st.text_input("Your account email", key="reset_email")
        if st.button("Send reset link", key="btn_reset"):
            if not reset_email:
                st.warning("Enter your email address.")
            else:
                with st.spinner("Sending…"):
                    err = reset_password(reset_email)
                if err:
                    st.error(f"Failed: {err}")
                else:
                    st.success("📧 Password-reset email sent! Check your inbox.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- AUTH GATE --------------------
if not current_user() and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
user = current_user()
if user:
    st.sidebar.success(f"👤 {user.get('email', 'User')}")
else:
    st.sidebar.info("👤 Guest session")

if st.sidebar.button("Logout"):
    sign_out()
    st.cache_resource.clear()
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- SIDEBAR HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

history: List[tuple] = st.session_state.chat_history
if history:
    items = list(reversed(list(enumerate(history, start=1))))
    labels = [f"{idx}. {q[:40]}…" for idx, (q, _) in items]

    selected_label = st.sidebar.selectbox(
        "Select a message",
        options=labels,
        label_visibility="collapsed",
        key="history_select",
    )

    if selected_label:
        sel_idx = int(selected_label.split(".")[0])
        q_sel, a_sel = history[sel_idx - 1]
        with st.sidebar.expander("Selected chat", expanded=True):
            st.markdown("**You**")
            st.write(q_sel)
            st.markdown("**Assistant**")
            st.write(a_sel)

    if st.sidebar.button("🧹 Clear History"):
        st.session_state.chat_history = []
        st.session_state.history_loaded = False
        st.rerun()
else:
    st.sidebar.caption("No history yet.")

# -------------------- HERO --------------------
st.markdown("## 📘 SlideSense AI Platform")
st.markdown("Smart Learning · Smart Vision · Smart AI")
st.divider()

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    pdf = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        # New PDF — reset state
        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.history_loaded = False

        # Load persisted history once per PDF session
        if not st.session_state.history_loaded and not st.session_state.get("guest"):
            load_chat_history_from_db()

        # Build vector DB
        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF…"):
                reader = PdfReader(pdf)
                text = "".join(page.extract_text() or "" for page in reader.pages)

                if not text.strip():
                    st.error("No readable text found in this PDF.")
                    st.stop()

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=80
                ).split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.text_input("Ask a question about the document")

        if q:
            with st.spinner("Thinking…"):
                llm = load_llm()
                docs = st.session_state.vector_db.similarity_search(q, k=5)

                prompt = ChatPromptTemplate.from_template(
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Rules:\n"
                    "- Answer only from the provided context.\n"
                    "- If the answer is not in the context, say: "
                    "'Information not found in the document.'"
                )

                chain = create_stuff_documents_chain(llm, prompt)
                res = chain.invoke({"context": docs, "question": q})
                answer = res.get("output_text", res) if isinstance(res, dict) else res

            st.session_state.chat_history.append((q, answer))
            save_chat_to_db(q, answer)

        if st.session_state.chat_history:
            st.markdown("## 💬 Conversation")
            for uq, ua in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(uq)
                with st.chat_message("assistant"):
                    st.markdown(ua)

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_container_width=True)   # use_column_width is deprecated

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analysing image…"):
                answer = answer_image_question(img, question)
            st.success(answer)
