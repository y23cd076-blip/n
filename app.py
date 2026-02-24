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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

    /* ═══════════════════════════════════════════════
       CSS CUSTOM PROPERTIES — auto light / dark
    ═══════════════════════════════════════════════ */
    :root {
        --accent:        #7c3aed;
        --accent-light:  #a855f7;
        --accent-soft:   rgba(124,58,237,0.10);
        --accent-glow:   rgba(124,58,237,0.30);

        /* surfaces */
        --bg:            #f0ebfa;
        --surface:       rgba(255,255,255,0.88);
        --surface-solid: #ffffff;
        --border:        rgba(196,181,253,0.35);
        --border-solid:  #e2d9f3;

        /* text */
        --text:          #18122b;
        --text-muted:    #6b6483;
        --text-faint:    #a09ab8;

        /* chat */
        --bubble-bot-bg: rgba(245,243,255,0.95);
        --bubble-bot-border: rgba(196,181,253,0.28);
        --input-bg:      rgba(245,243,255,0.9);

        --shadow-card:   0 20px 56px rgba(100,60,200,0.16), 0 3px 10px rgba(0,0,0,0.06);
        --shadow-hover:  0 28px 72px rgba(100,60,200,0.22), 0 5px 16px rgba(0,0,0,0.08);
    }

    /* Dark mode overrides — respects OS preference AND Streamlit dark theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg:            #0f0a1e;
            --surface:       rgba(28,18,48,0.90);
            --surface-solid: #1c1230;
            --border:        rgba(124,58,237,0.22);
            --border-solid:  rgba(124,58,237,0.18);

            --text:          #ede9f4;
            --text-muted:    #a09ab8;
            --text-faint:    #6b6483;

            --bubble-bot-bg: rgba(38,24,64,0.95);
            --bubble-bot-border: rgba(124,58,237,0.20);
            --input-bg:      rgba(38,24,64,0.8);

            --shadow-card:   0 20px 56px rgba(0,0,0,0.5), 0 3px 10px rgba(0,0,0,0.3);
            --shadow-hover:  0 28px 72px rgba(0,0,0,0.6), 0 5px 16px rgba(0,0,0,0.4);
        }
    }

    /* Streamlit injects [data-theme="dark"] on the root element */
    [data-theme="dark"] {
        --bg:            #0f0a1e;
        --surface:       rgba(28,18,48,0.90);
        --surface-solid: #1c1230;
        --border:        rgba(124,58,237,0.22);
        --border-solid:  rgba(124,58,237,0.18);
        --text:          #ede9f4;
        --text-muted:    #a09ab8;
        --text-faint:    #6b6483;
        --bubble-bot-bg: rgba(38,24,64,0.95);
        --bubble-bot-border: rgba(124,58,237,0.20);
        --input-bg:      rgba(38,24,64,0.8);
        --shadow-card:   0 20px 56px rgba(0,0,0,0.5), 0 3px 10px rgba(0,0,0,0.3);
        --shadow-hover:  0 28px 72px rgba(0,0,0,0.6), 0 5px 16px rgba(0,0,0,0.4);
    }

    /* ═══════════════════════════════════════════════
       GLOBAL
    ═══════════════════════════════════════════════ */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .stApp {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
        background: var(--bg);
        min-height: 100vh;
        transition: background 0.4s ease, color 0.4s ease;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 540px !important;
        padding: 2rem 0.75rem 3rem !important;
        margin: 0 auto;
        position: relative; z-index: 1;
    }

    /* ═══════════════════════════════════════════════
       ANIMATIONS — keyframes
    ═══════════════════════════════════════════════ */
    @keyframes rise      { from{opacity:0;transform:translateY(22px) scale(0.97)} to{opacity:1;transform:none} }
    @keyframes fade-in   { from{opacity:0} to{opacity:1} }
    @keyframes slide-x   { from{opacity:0;transform:translateX(-12px)} to{opacity:1;transform:none} }
    @keyframes slide-xr  { from{opacity:0;transform:translateX(12px)}  to{opacity:1;transform:none} }
    @keyframes pop       { from{transform:scale(0) rotate(-18deg)} to{transform:scale(1) rotate(0)} }
    @keyframes dot-pulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(.8);opacity:.6} }
    @keyframes dot-ring  { 0%{transform:translate(-50%,-50%) scale(1);opacity:.6} 100%{transform:translate(-50%,-50%) scale(3.2);opacity:0} }
    @keyframes bounce3   { 0%,60%,100%{transform:translateY(0);opacity:.45} 30%{transform:translateY(-6px);opacity:1} }
    @keyframes msg-in-l  { from{opacity:0;transform:translateX(-12px) scale(.96)} to{opacity:1;transform:none} }
    @keyframes msg-in-r  { from{opacity:0;transform:translateX(12px)  scale(.96)} to{opacity:1;transform:none} }
    @keyframes glow-pulse{ 0%,100%{box-shadow:0 3px 10px var(--accent-glow)} 50%{box-shadow:0 4px 20px rgba(124,58,237,.55)} }
    @keyframes bar-up    { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }

    /* ═══════════════════════════════════════════════
       CHAT WIDGET  (clean, theme-aware)
    ═══════════════════════════════════════════════ */
    .chat-widget {
        background: var(--surface);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        box-shadow: var(--shadow-card);
        border: 1px solid var(--border);
        overflow: hidden;
        display: flex; flex-direction: column;
        max-height: 80vh;
        animation: rise .5s cubic-bezier(.22,1,.36,1) both;
        transition: box-shadow .3s ease;
    }
    .chat-widget:hover { box-shadow: var(--shadow-hover); }

    /* Header */
    .chat-header {
        background: var(--surface);
        padding: .9rem 1.2rem;
        display: flex; align-items: center; gap: .6rem;
        border-bottom: 1px solid var(--border);
        flex-shrink: 0;
        animation: fade-in .4s ease .1s both;
    }
    .chat-header-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; flex-shrink: 0;
        box-shadow: 0 2px 10px var(--accent-glow);
        animation: pop .45s cubic-bezier(.34,1.56,.64,1) .15s both;
    }
    .chat-header-info { flex: 1; }
    .chat-header-name {
        font-weight: 700; font-size: .92rem;
        color: var(--text); line-height: 1.2;
    }
    .chat-header-status {
        font-size: .73rem; color: var(--text-muted);
        display: flex; align-items: center; gap: 4px;
        animation: fade-in .5s ease .35s both;
    }
    .status-dot {
        width: 7px; height: 7px; background: #22c55e;
        border-radius: 50%; display: inline-block; position: relative;
        animation: dot-pulse 2.4s ease-in-out infinite;
    }
    .status-dot::after {
        content: ''; position: absolute;
        top: 50%; left: 50%;
        width: 7px; height: 7px; background: #22c55e;
        border-radius: 50%;
        animation: dot-ring 2.4s ease-out infinite;
    }
    .chat-header-actions { display: flex; gap: .45rem; align-items: center; }
    .header-btn {
        width: 28px; height: 28px; border: none;
        background: var(--accent-soft); border-radius: 50%;
        cursor: pointer; display: flex; align-items: center; justify-content: center;
        font-size: .82rem; color: var(--accent);
        transition: background .2s, transform .2s cubic-bezier(.34,1.56,.64,1);
    }
    .header-btn:hover { background: var(--border); transform: scale(1.14) rotate(8deg); }

    /* Date separator */
    .date-separator {
        text-align: center; font-size: .7rem; color: var(--text-faint);
        padding: .7rem 0 .35rem; letter-spacing: .05em;
        animation: fade-in .45s ease .25s both;
    }

    /* Messages scroll area */
    .chat-messages {
        flex: 1; overflow-y: auto;
        padding: .5rem 1rem .7rem;
        display: flex; flex-direction: column; gap: .55rem;
        scroll-behavior: smooth;
    }
    .chat-messages::-webkit-scrollbar { width: 3px; }
    .chat-messages::-webkit-scrollbar-thumb {
        background: var(--accent); border-radius: 3px; opacity: .4;
    }

    /* Bubbles */
    .msg-bot {
        align-self: flex-start;
        background: var(--bubble-bot-bg);
        color: var(--text);
        border: 1px solid var(--bubble-bot-border);
        border-radius: 16px 16px 16px 3px;
        padding: .65rem .9rem;
        font-size: .88rem; max-width: 82%; line-height: 1.55;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
        animation: msg-in-l .35s cubic-bezier(.22,1,.36,1) both;
    }
    .msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        color: #fff;
        border-radius: 16px 16px 3px 16px;
        padding: .65rem .9rem;
        font-size: .88rem; max-width: 82%; line-height: 1.55;
        box-shadow: 0 3px 12px var(--accent-glow);
        animation: msg-in-r .35s cubic-bezier(.22,1,.36,1) both;
    }

    /* Typing dots */
    .typing-indicator {
        align-self: flex-start;
        background: var(--bubble-bot-bg);
        border: 1px solid var(--bubble-bot-border);
        border-radius: 16px 16px 16px 3px;
        padding: .7rem .9rem;
        display: flex; gap: 5px; align-items: center;
        animation: msg-in-l .3s ease both;
    }
    .typing-dot {
        width: 7px; height: 7px; background: var(--accent);
        border-radius: 50%; opacity: .5;
        animation: bounce3 1.3s ease-in-out infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: .18s; }
    .typing-dot:nth-child(3) { animation-delay: .36s; }

    /* Input bar */
    .chat-input-bar {
        border-top: 1px solid var(--border);
        padding: .65rem .9rem;
        display: flex; align-items: center; gap: .45rem;
        background: var(--surface);
        flex-shrink: 0;
        animation: bar-up .4s cubic-bezier(.22,1,.36,1) .2s both;
    }
    .icon-btn {
        width: 32px; height: 32px; border: none;
        background: transparent; cursor: pointer;
        font-size: 1.1rem; color: var(--text-faint);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        transition: color .2s, background .2s, transform .2s cubic-bezier(.34,1.56,.64,1);
    }
    .icon-btn:hover { color: var(--accent); background: var(--accent-soft); transform: scale(1.15) rotate(-6deg); }

    /* Input inside bar */
    .chat-input-bar .stTextInput > div > div > input {
        border: 1.5px solid transparent !important;
        background: var(--input-bg) !important;
        border-radius: 999px !important;
        padding: .48rem 1rem !important;
        font-size: .86rem !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text) !important;
        box-shadow: none !important;
        transition: border-color .25s, box-shadow .25s, background .25s !important;
    }
    .chat-input-bar .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
        background: var(--surface-solid) !important;
    }
    .chat-input-bar .stTextInput > div > div > input::placeholder { color: var(--text-faint); }
    .chat-input-bar .stTextInput label { display: none !important; }

    /* ═══════════════════════════════════════════════
       GLOBAL STREAMLIT OVERRIDES (theme-aware)
    ═══════════════════════════════════════════════ */

    /* All buttons */
    .stButton > button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
        color: #fff !important; border: none !important;
        font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
        padding: .48rem 1.4rem !important;
        transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s !important;
        box-shadow: 0 2px 8px var(--accent-glow) !important;
        animation: glow-pulse 3s ease infinite !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.04) !important;
        box-shadow: 0 6px 18px rgba(124,58,237,.45) !important;
    }
    .stButton > button:active { transform: scale(.96) !important; }

    /* Text inputs */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid var(--border-solid) !important;
        background: var(--surface-solid) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color .25s, box-shadow .25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }
    .stTextInput > label { color: var(--text-muted) !important; font-size: .8rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px; border-bottom: 2px solid var(--border-solid);
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600; font-size: .84rem; color: var(--text-muted) !important;
        padding: .38rem .7rem !important;
        transition: color .2s, background .2s !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: var(--text) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-solid) !important;
        border-radius: 14px !important;
        background: var(--input-bg) !important;
        transition: border-color .25s, background .25s !important;
        animation: fade-in .5s ease .1s both;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* Alerts */
    .stAlert { border-radius: 12px !important; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; opacity: .4; }
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
    "username": None,          # display name chosen at signup
    "auth_mode": "login",      # "login" | "signup" | "guest"
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
def set_session(data: dict, username: Optional[str] = None) -> None:
    st.session_state["session"] = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "user": data["user"],
    }
    if username:
        st.session_state["username"] = username
    elif not st.session_state.get("username"):
        # Derive a display name from the email stored in the user object
        email = data.get("user", {}).get("email", "")
        st.session_state["username"] = email.split("@")[0] if email else "User"


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    return sess["user"] if sess else None


def _current_token() -> Optional[str]:
    sess = st.session_state.get("session")
    return sess["access_token"] if sess else None


def sign_up(email: str, password: str, username: str = "") -> tuple:
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
        set_session(data, username=username or email.split("@")[0])
        return None, True

    # Confirmation email sent — user must verify before logging in
    st.session_state["pending_verification_email"] = email
    if username:
        st.session_state["username"] = username
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

    set_session(data)   # derives username from email if not already set
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
    st.session_state["username"] = None


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
    pass  # No email verification needed


def login_ui() -> None:
    mode = st.session_state.get("auth_mode", "login")

    st.markdown("""
    <style>
    .auth-header { text-align:center; margin-bottom: 1.6rem; }
    .auth-logo   {
        font-size: 2.4rem; display: block; margin-bottom: .3rem;
    }
    .auth-title  {
        font-size: 1.5rem; font-weight: 800; color: #1a1a2e;
        letter-spacing: -.02em;
    }
    .auth-sub    { font-size: .82rem; color: #7c6fa0; margin-top: .15rem; }

    /* Tab strip */
    .auth-tabs   {
        display: flex; border-bottom: 2px solid #ede9f4;
        margin-bottom: 1.4rem; gap: 0;
    }
    .auth-tab    {
        flex: 1; text-align: center;
        padding: .55rem 0; font-size: .85rem; font-weight: 600;
        color: #9ca3af; border-bottom: 2px solid transparent;
        margin-bottom: -2px; cursor: pointer;
        transition: color .2s, border-color .2s;
    }
    .auth-tab.active { color: #7c3aed; border-bottom-color: #7c3aed; }
    </style>
    """, unsafe_allow_html=True)

    # Brand
    st.markdown("""
    <div class="auth-header">
      <span class="auth-logo">📘</span>
      <div class="auth-title">SlideSense</div>
      <div class="auth-sub">Smart Learning · Smart Vision · Smart AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Tab strip (visual)
    tl = "active" if mode == "login"  else ""
    ts = "active" if mode == "signup" else ""
    tg = "active" if mode == "guest"  else ""
    st.markdown(f"""
    <div class="auth-tabs">
      <div class="auth-tab {tl}">Login</div>
      <div class="auth-tab {ts}">Sign Up</div>
      <div class="auth-tab {tg}">Guest</div>
    </div>""", unsafe_allow_html=True)

    # Functional tab buttons (invisible drivers)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Login", key="tab_login", use_container_width=True):
            st.session_state["auth_mode"] = "login"; st.rerun()
    with c2:
        if st.button("Sign Up", key="tab_signup", use_container_width=True):
            st.session_state["auth_mode"] = "signup"; st.rerun()
    with c3:
        if st.button("Guest", key="tab_guest", use_container_width=True):
            st.session_state["auth_mode"] = "guest"; st.rerun()

    st.write("")  # small spacer

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    if mode == "login":
        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
        if st.button("Sign In", key="btn_login", use_container_width=True):
            if not username or not password:
                st.warning("Please fill in all fields.")
            else:
                with st.spinner("Signing in…"):
                    err = sign_in(username, password)
                if err:
                    st.error(err)
                else:
                    st.rerun()

    # ── SIGN UP ───────────────────────────────────────────────────────────────
    elif mode == "signup":
        su_user = st.text_input("Username", placeholder="Choose a username", key="signup_user")
        su_pw   = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_pw")
        su_pw2  = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="signup_pw2")
        if st.button("Create Account", key="btn_signup", use_container_width=True):
            if not su_user or not su_pw or not su_pw2:
                st.warning("Please fill in all fields.")
            elif su_pw != su_pw2:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating account…"):
                    err, logged_in = sign_up(su_user, su_pw)
                if err:
                    st.error(err)
                elif logged_in:
                    st.rerun()
                else:
                    st.success("Account created! Go to Login to sign in.")

    # ── GUEST ─────────────────────────────────────────────────────────────────
    else:
        st.info("Continue without an account. Your chat history won't be saved.")
        st.write("")
        if st.button("Continue as Guest", key="btn_guest", use_container_width=True):
            st.session_state["guest"] = True
            st.session_state["username"] = "Guest"
            st.rerun()
# -------------------- AUTH GATE --------------------
if not current_user() and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
user = current_user()
username = st.session_state.get("username") or (
    user.get("user_metadata", {}).get("username") or
    user.get("email", "user").split("@")[0]
    if user else "Guest"
)
if user:
    st.sidebar.success(f"👤 {username}")
else:
    st.sidebar.info(f"👤 {username}")

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


def render_messages(history, thinking: bool = False) -> None:
    today = _dt.date.today().strftime("%B %d, %Y")
    st.markdown(f'<div class="date-separator">{today}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for i, (uq, ua) in enumerate(history):
        delay_u = i * 0.04
        delay_a = i * 0.04 + 0.08
        st.markdown(
            f'<div class="msg-user" style="animation-delay:{delay_u:.2f}s">{uq}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="msg-bot" style="animation-delay:{delay_a:.2f}s">{ua}</div>',
            unsafe_allow_html=True,
        )
    if thinking:
        st.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    # Auto-scroll JS
    st.markdown(
        "<script>const msgs=document.querySelector('.chat-messages');"
        "if(msgs)msgs.scrollTop=msgs.scrollHeight;</script>",
        unsafe_allow_html=True,
    )


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
