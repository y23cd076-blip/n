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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Global reset ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .stApp {
        background: #ede9f4;
        font-family: 'DM Sans', sans-serif;
        color: #1a1a2e;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 520px !important;
        padding: 2rem 0 2rem !important;
        margin: 0 auto;
    }

    /* ── Login card ── */
    .login-card {
        background: #fff;
        border-radius: 24px;
        padding: 2rem 1.75rem;
        box-shadow: 0 12px 40px rgba(100,60,200,0.13);
        border: 1px solid #e2d9f3;
    }
    .login-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .login-subtitle {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 1.4rem;
    }

    /* ── Chat widget wrapper ── */
    .chat-widget {
        background: #fff;
        border-radius: 28px;
        box-shadow: 0 20px 60px rgba(100,60,200,0.18), 0 4px 16px rgba(0,0,0,0.08);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        max-height: 82vh;
        border: 1px solid #e2d9f3;
    }

    /* ── Header bar ── */
    .chat-header {
        background: #fff;
        padding: 1rem 1.25rem 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        border-bottom: 1px solid #f0ebfa;
        flex-shrink: 0;
    }
    .chat-header-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(124,58,237,0.35);
    }
    .chat-header-info { flex: 1; }
    .chat-header-name {
        font-weight: 700; font-size: 0.95rem; color: #1a1a2e; line-height: 1.2;
    }
    .chat-header-status {
        font-size: 0.75rem; color: #6b7280; display: flex; align-items: center; gap: 4px;
    }
    .status-dot {
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%,100% { opacity:1; }
        50% { opacity:0.4; }
    }
    .chat-header-actions {
        display: flex; gap: 0.5rem; align-items: center;
    }
    .header-btn {
        width: 30px; height: 30px;
        border: none; background: #f5f3ff;
        border-radius: 50%; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; color: #7c3aed;
        transition: background 0.15s;
    }
    .header-btn:hover { background: #ede9f4; }

    /* ── Date separator ── */
    .date-separator {
        text-align: center; font-size: 0.72rem;
        color: #9ca3af; padding: 0.75rem 0 0.4rem;
        letter-spacing: 0.04em;
    }

    /* ── Messages area ── */
    .chat-messages {
        flex: 1; overflow-y: auto;
        padding: 0.5rem 1.1rem 0.75rem;
        display: flex; flex-direction: column; gap: 0.55rem;
        scroll-behavior: smooth;
    }
    .chat-messages::-webkit-scrollbar { width: 4px; }
    .chat-messages::-webkit-scrollbar-thumb { background: #d1c4e9; border-radius: 4px; }

    /* Bot bubble */
    .msg-bot {
        align-self: flex-start;
        background: #f5f3ff;
        color: #1a1a2e;
        border-radius: 18px 18px 18px 4px;
        padding: 0.65rem 0.9rem;
        font-size: 0.9rem;
        max-width: 82%;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        animation: bubble-in 0.2s ease-out;
    }
    /* User bubble */
    .msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.65rem 0.9rem;
        font-size: 0.9rem;
        max-width: 82%;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(124,58,237,0.3);
        animation: bubble-in 0.2s ease-out;
    }
    @keyframes bubble-in {
        from { opacity:0; transform:translateY(6px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* ── Quick reply buttons ── */
    .quick-replies {
        display: flex; flex-wrap: wrap; gap: 0.45rem;
        padding: 0.3rem 1.1rem 0.6rem;
        justify-content: flex-end;
    }
    .quick-btn-filled {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: #fff; border: none;
        border-radius: 999px; padding: 0.45rem 1.1rem;
        font-size: 0.83rem; font-weight: 600;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: transform 0.1s, box-shadow 0.15s;
        box-shadow: 0 2px 8px rgba(124,58,237,0.3);
    }
    .quick-btn-filled:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(124,58,237,0.4); }
    .quick-btn-outline {
        background: #fff; color: #7c3aed;
        border: 1.5px solid #c4b5fd;
        border-radius: 999px; padding: 0.43rem 1.1rem;
        font-size: 0.83rem; font-weight: 600;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: background 0.15s;
    }
    .quick-btn-outline:hover { background: #f5f3ff; }

    /* ── Suggested reply ── */
    .suggested-reply {
        margin: 0 1.1rem 0.65rem;
        align-self: flex-end;
    }
    .suggested-btn {
        background: #fff; color: #7c3aed;
        border: 1.5px solid #c4b5fd;
        border-radius: 999px; padding: 0.5rem 1.2rem;
        font-size: 0.83rem; font-weight: 500;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: background 0.15s;
        display: inline-block;
    }
    .suggested-btn:hover { background: #f5f3ff; }

    /* ── Input bar ── */
    .chat-input-bar {
        border-top: 1px solid #f0ebfa;
        padding: 0.7rem 1rem;
        display: flex; align-items: center; gap: 0.5rem;
        background: #fff;
        flex-shrink: 0;
    }
    .input-icons {
        display: flex; gap: 0.35rem; align-items: center;
    }
    .icon-btn {
        width: 32px; height: 32px; border: none;
        background: transparent; cursor: pointer;
        font-size: 1.1rem; color: #9ca3af;
        border-radius: 50%; display:flex; align-items:center; justify-content:center;
        transition: color 0.15s, background 0.15s;
    }
    .icon-btn:hover { color: #7c3aed; background: #f5f3ff; }

    /* Override Streamlit text input inside bar */
    .chat-input-bar .stTextInput { flex: 1; }
    .chat-input-bar .stTextInput > div > div > input {
        border: none !important;
        background: #f5f3ff !important;
        border-radius: 999px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.88rem !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #1a1a2e !important;
        box-shadow: none !important;
    }
    .chat-input-bar .stTextInput > div > div > input::placeholder { color: #9ca3af; }
    .chat-input-bar .stTextInput label { display: none !important; }

    /* Send button */
    .send-btn {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        border: none; border-radius: 50%;
        color: #fff; font-size: 1rem;
        cursor: pointer; display:flex; align-items:center; justify-content:center;
        box-shadow: 0 2px 8px rgba(124,58,237,0.35);
        transition: transform 0.1s;
        flex-shrink: 0;
    }
    .send-btn:hover { transform: scale(1.07); }

    /* ── Tabs & form overrides ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #f0ebfa; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600; font-size: 0.85rem; color: #6b7280 !important;
        padding: 0.4rem 0.75rem !important;
    }
    .stTabs [aria-selected="true"] { color: #7c3aed !important; }

    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e2d9f3 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    }

    .stButton > button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.1s, box-shadow 0.15s !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(124,58,237,0.4) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #fff !important;
        border-right: 1px solid #f0ebfa !important;
    }
    section[data-testid="stSidebar"] * { color: #1a1a2e !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c4b5fd !important;
        border-radius: 14px !important;
        background: #faf8ff !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #7c3aed !important; }

    /* Scrollbar global */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-thumb { background: #d1c4e9; border-radius: 4px; }
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


def sign_up(email: str, password: str) -> tuple:
    """
    Register a new user.
    Returns (error_str, auto_logged_in).
    auto_logged_in=True when Supabase returns a session immediately
    (i.e. email confirmation is disabled in dashboard).
    """
    if len(password) < 6:
        return "Password must be at least 6 characters.", False

    resp = _auth_post("/auth/v1/signup", {"email": email, "password": password})

    if resp.status_code >= 400:
        return _parse_error(resp), False

    data = resp.json()

    # Email already exists (email confirm ON returns identities=[])
    if isinstance(data.get("identities"), list) and len(data["identities"]) == 0:
        return "This email is already registered. Please log in or reset your password.", False

    # Supabase returned a full session — email confirmation is OFF, auto-login
    if data.get("access_token") and data.get("user"):
        set_session(data)
        return None, True

    # Confirmation email sent — user must verify before logging in
    st.session_state["pending_verification_email"] = email
    return None, False


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
                    err, auto_logged_in = sign_up(su_email, su_pw)
                if err:
                    st.error(f"Sign-up failed: {err}")
                elif auto_logged_in:
                    st.rerun()  # email confirm OFF — already logged in
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

import datetime as _dt

# ==================== CHAT WIDGET HEADER ====================
def render_chat_header(title: str, subtitle: str, emoji: str = "📘") -> None:
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-icon">{emoji}</div>
        <div class="chat-header-info">
            <div class="chat-header-name">{title}</div>
            <div class="chat-header-status">
                <span class="status-dot"></span> {subtitle}
            </div>
        </div>
        <div class="chat-header-actions">
            <button class="header-btn" title="Refresh">↺</button>
            <button class="header-btn" title="Minimize">✕</button>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_messages(history) -> None:
    today = _dt.date.today().strftime("%B %d, %Y")
    st.markdown(f'<div class="date-separator">{today}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for uq, ua in history:
        st.markdown(f'<div class="msg-user">{uq}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-bot">{ua}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    st.markdown('<div class="chat-widget">', unsafe_allow_html=True)
    render_chat_header("SlideSense Bot", "We're online ...", "📘")

    # ---- Upload area (outside widget when no PDF yet) ----
    st.markdown('</div>', unsafe_allow_html=True)  # close widget temporarily

    pdf = st.file_uploader("Upload a PDF to start chatting", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.history_loaded = False

        if not st.session_state.history_loaded and not st.session_state.get("guest"):
            load_chat_history_from_db()

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

        # ---- Reopen widget ----
        st.markdown('<div class="chat-widget">', unsafe_allow_html=True)
        render_chat_header("SlideSense Bot", "We're online ...", "📘")

        # Messages
        if st.session_state.chat_history:
            render_messages(st.session_state.chat_history)
        else:
            st.markdown("""
            <div class="chat-messages">
                <div class="msg-bot">
                    Hi there! 👋 I've loaded your PDF.<br>
                    Ask me anything about the document!
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Quick reply suggestions (shown when chat is empty)
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="quick-replies">
                <span style="font-size:0.78rem;color:#9ca3af;align-self:center;margin-right:4px;">Try:</span>
            </div>
            """, unsafe_allow_html=True)

        # Input bar
        st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
        col_emoji, col_input, col_attach, col_send = st.columns([0.08, 0.72, 0.08, 0.12])
        with col_emoji:
            st.markdown('<button class="icon-btn" title="Emoji">🙂</button>', unsafe_allow_html=True)
        with col_input:
            q = st.text_input("msg", placeholder="Enter message", label_visibility="collapsed", key="pdf_q")
        with col_attach:
            st.markdown('<button class="icon-btn" title="Attach">📎</button>', unsafe_allow_html=True)
        with col_send:
            send = st.button("➤", key="pdf_send")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close widget

        # Process question
        if (q and send) or q:
            if q.strip():
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
                st.rerun()

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    st.markdown('<div class="chat-widget">', unsafe_allow_html=True)
    render_chat_header("Vision Bot", "We're online ...", "🖼")

    st.markdown('</div>', unsafe_allow_html=True)

    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")

        st.markdown('<div class="chat-widget">', unsafe_allow_html=True)
        render_chat_header("Vision Bot", "We're online ...", "🖼")

        # Show image + message history
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        st.markdown('<div class="msg-bot">Image loaded! Ask me anything about it 👆</div>', unsafe_allow_html=True)

        if "img_history" not in st.session_state:
            st.session_state.img_history = []
        for uq, ua in st.session_state.img_history:
            st.markdown(f'<div class="msg-user">{uq}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-bot">{ua}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Image preview (small, inside widget)
        st.image(img, use_container_width=True)

        # Input bar
        st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
        col_emoji, col_input, col_attach, col_send = st.columns([0.08, 0.72, 0.08, 0.12])
        with col_emoji:
            st.markdown('<button class="icon-btn">🙂</button>', unsafe_allow_html=True)
        with col_input:
            iq = st.text_input("imgmsg", placeholder="Ask about the image...", label_visibility="collapsed", key="img_q")
        with col_attach:
            st.markdown('<button class="icon-btn">📎</button>', unsafe_allow_html=True)
        with col_send:
            isend = st.button("➤", key="img_send")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close widget

        if iq and iq.strip():
            with st.spinner("Analysing image…"):
                ans = answer_image_question(img, iq)
            st.session_state.img_history.append((iq, ans))
            st.rerun()
