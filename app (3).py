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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    /* ── Global reset ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── Animated background ── */
    .stApp {
        font-family: 'DM Sans', sans-serif;
        color: #1a1a2e;
        background: linear-gradient(135deg, #e8e0f5 0%, #ede9f4 40%, #ddd6f3 100%);
        background-size: 400% 400%;
        animation: bg-shift 12s ease infinite;
        min-height: 100vh;
    }
    @keyframes bg-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating orbs behind widget */
    .stApp::before, .stApp::after {
        content: '';
        position: fixed;
        border-radius: 50%;
        filter: blur(70px);
        pointer-events: none;
        z-index: 0;
        animation: orb-float 8s ease-in-out infinite alternate;
    }
    .stApp::before {
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(167,139,250,0.22) 0%, transparent 70%);
        top: -100px; left: -80px;
    }
    .stApp::after {
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
        bottom: -60px; right: -40px;
        animation-delay: -4s;
    }
    @keyframes orb-float {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(30px, 20px) scale(1.08); }
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 520px !important;
        padding: 2rem 0 2rem !important;
        margin: 0 auto;
        position: relative; z-index: 1;
    }

    /* ── Widget entrance ── */
    .chat-widget, .login-card {
        animation: widget-rise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    @keyframes widget-rise {
        from { opacity: 0; transform: translateY(28px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0)   scale(1);    }
    }

    /* ── Login card ── */
    .login-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 2rem 1.75rem;
        box-shadow: 0 12px 40px rgba(100,60,200,0.13), 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid rgba(226,217,243,0.8);
    }
    .login-title {
        font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.3rem;
    }
    .login-subtitle {
        font-size: 0.88rem; color: #6b7280; margin-bottom: 1.4rem;
    }

    /* ── Chat widget ── */
    .chat-widget {
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border-radius: 28px;
        box-shadow:
            0 24px 64px rgba(100,60,200,0.20),
            0 4px 18px rgba(0,0,0,0.07),
            inset 0 1px 0 rgba(255,255,255,0.9);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        max-height: 82vh;
        border: 1px solid rgba(226,217,243,0.7);
        transition: box-shadow 0.3s ease;
    }
    .chat-widget:hover {
        box-shadow:
            0 32px 80px rgba(100,60,200,0.25),
            0 6px 24px rgba(0,0,0,0.09),
            inset 0 1px 0 rgba(255,255,255,0.9);
    }

    /* ── Header ── */
    .chat-header {
        background: rgba(255,255,255,0.95);
        padding: 1rem 1.25rem 0.9rem;
        display: flex; align-items: center; gap: 0.65rem;
        border-bottom: 1px solid rgba(240,235,250,0.9);
        flex-shrink: 0;
        animation: header-slide 0.45s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
    }
    @keyframes header-slide {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Spinning gradient ring on icon */
    .chat-header-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #7c3aed, #a855f7, #6d28d9);
        background-size: 200% 200%;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
        box-shadow: 0 3px 12px rgba(124,58,237,0.45);
        animation: icon-gradient 4s ease infinite, icon-pop 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.2s both;
    }
    @keyframes icon-gradient {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes icon-pop {
        from { transform: scale(0) rotate(-20deg); }
        to   { transform: scale(1) rotate(0deg); }
    }

    .chat-header-info { flex: 1; }
    .chat-header-name {
        font-weight: 700; font-size: 0.95rem; color: #1a1a2e; line-height: 1.2;
    }
    .chat-header-status {
        font-size: 0.75rem; color: #6b7280;
        display: flex; align-items: center; gap: 4px;
        animation: fade-in 0.6s ease 0.4s both;
    }
    @keyframes fade-in {
        from { opacity: 0; } to { opacity: 1; }
    }

    /* Pulsing status dot with ripple */
    .status-dot {
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        position: relative;
        animation: dot-pulse 2.4s ease-in-out infinite;
    }
    .status-dot::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        animation: dot-ripple 2.4s ease-out infinite;
    }
    @keyframes dot-pulse {
        0%, 100% { transform: scale(1);   opacity: 1;   }
        50%       { transform: scale(0.85); opacity: 0.7; }
    }
    @keyframes dot-ripple {
        0%   { transform: translate(-50%,-50%) scale(1);   opacity: 0.6; }
        100% { transform: translate(-50%,-50%) scale(3);   opacity: 0;   }
    }

    .chat-header-actions { display: flex; gap: 0.5rem; align-items: center; }
    .header-btn {
        width: 30px; height: 30px;
        border: none; background: #f5f3ff;
        border-radius: 50%; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; color: #7c3aed;
        transition: background 0.2s, transform 0.2s cubic-bezier(0.34,1.56,0.64,1);
    }
    .header-btn:hover { background: #ede9f4; transform: scale(1.15) rotate(10deg); }

    /* ── Date separator ── */
    .date-separator {
        text-align: center; font-size: 0.72rem;
        color: #9ca3af; padding: 0.75rem 0 0.4rem;
        letter-spacing: 0.05em;
        animation: fade-in 0.5s ease 0.3s both;
        position: relative;
    }
    .date-separator::before, .date-separator::after {
        content: '';
        position: absolute;
        top: 50%; width: 28%;
        height: 1px;
        background: linear-gradient(to right, transparent, #e2d9f3);
        margin-top: 4px;
    }
    .date-separator::before { left: 4%; }
    .date-separator::after  { right: 4%; background: linear-gradient(to left, transparent, #e2d9f3); }

    /* ── Messages area ── */
    .chat-messages {
        flex: 1; overflow-y: auto;
        padding: 0.5rem 1.1rem 0.75rem;
        display: flex; flex-direction: column; gap: 0.6rem;
        scroll-behavior: smooth;
    }
    .chat-messages::-webkit-scrollbar { width: 3px; }
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #c4b5fd, #a78bfa);
        border-radius: 4px;
    }

    /* ── Message bubbles ── */
    .msg-bot {
        align-self: flex-start;
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9f4 100%);
        color: #1a1a2e;
        border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem; max-width: 82%; line-height: 1.55;
        box-shadow: 0 2px 8px rgba(124,58,237,0.10), 0 1px 2px rgba(0,0,0,0.04);
        animation: msg-bot-in 0.38s cubic-bezier(0.22, 1, 0.36, 1) both;
        border: 1px solid rgba(196,181,253,0.25);
    }
    @keyframes msg-bot-in {
        from { opacity: 0; transform: translateX(-14px) scale(0.95); }
        to   { opacity: 1; transform: translateX(0)     scale(1);    }
    }

    .msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #a855f7 100%);
        background-size: 200% 200%;
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem; max-width: 82%; line-height: 1.55;
        box-shadow: 0 4px 14px rgba(124,58,237,0.38), 0 1px 4px rgba(0,0,0,0.08);
        animation: msg-user-in 0.38s cubic-bezier(0.22, 1, 0.36, 1) both,
                   msg-gradient 5s ease infinite 0.4s;
    }
    @keyframes msg-user-in {
        from { opacity: 0; transform: translateX(14px) scale(0.95); }
        to   { opacity: 1; transform: translateX(0)    scale(1);    }
    }
    @keyframes msg-gradient {
        0%,100% { background-position: 0% 50%;   }
        50%      { background-position: 100% 50%; }
    }

    /* Typing indicator */
    .typing-indicator {
        align-self: flex-start;
        background: linear-gradient(135deg, #f5f3ff, #ede9f4);
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1rem;
        display: flex; gap: 5px; align-items: center;
        animation: msg-bot-in 0.3s ease both;
        border: 1px solid rgba(196,181,253,0.25);
    }
    .typing-dot {
        width: 7px; height: 7px;
        background: #a78bfa;
        border-radius: 50%;
        animation: typing-bounce 1.3s ease-in-out infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.18s; }
    .typing-dot:nth-child(3) { animation-delay: 0.36s; }
    @keyframes typing-bounce {
        0%,60%,100% { transform: translateY(0);    opacity: 0.5; }
        30%          { transform: translateY(-6px); opacity: 1;   }
    }

    /* ── Quick reply buttons ── */
    .quick-replies {
        display: flex; flex-wrap: wrap; gap: 0.45rem;
        padding: 0.3rem 1.1rem 0.6rem;
        justify-content: flex-end;
        animation: fade-in 0.5s ease 0.2s both;
    }
    .quick-btn-filled {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: #fff; border: none;
        border-radius: 999px; padding: 0.48rem 1.15rem;
        font-size: 0.83rem; font-weight: 600;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(124,58,237,0.3);
    }
    .quick-btn-filled:hover {
        transform: translateY(-2px) scale(1.04);
        box-shadow: 0 6px 16px rgba(124,58,237,0.45);
    }
    .quick-btn-filled:active { transform: scale(0.97); }
    .quick-btn-outline {
        background: rgba(255,255,255,0.8); color: #7c3aed;
        border: 1.5px solid #c4b5fd;
        border-radius: 999px; padding: 0.46rem 1.15rem;
        font-size: 0.83rem; font-weight: 600;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: background 0.2s, transform 0.2s cubic-bezier(0.34,1.56,0.64,1);
    }
    .quick-btn-outline:hover { background: #f5f3ff; transform: translateY(-2px) scale(1.03); }

    /* ── Input bar ── */
    .chat-input-bar {
        border-top: 1px solid rgba(240,235,250,0.9);
        padding: 0.75rem 1rem;
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(255,255,255,0.96);
        flex-shrink: 0;
        animation: input-bar-rise 0.45s cubic-bezier(0.22,1,0.36,1) 0.25s both;
    }
    @keyframes input-bar-rise {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    .icon-btn {
        width: 34px; height: 34px; border: none;
        background: transparent; cursor: pointer;
        font-size: 1.15rem; color: #9ca3af;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        transition: color 0.2s, background 0.2s,
                    transform 0.2s cubic-bezier(0.34,1.56,0.64,1);
    }
    .icon-btn:hover {
        color: #7c3aed; background: #f5f3ff;
        transform: scale(1.18) rotate(-8deg);
    }
    .icon-btn:active { transform: scale(0.92); }

    /* Input field */
    .chat-input-bar .stTextInput { flex: 1; }
    .chat-input-bar .stTextInput > div > div > input {
        border: 1.5px solid transparent !important;
        background: #f5f3ff !important;
        border-radius: 999px !important;
        padding: 0.52rem 1.1rem !important;
        font-size: 0.88rem !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #1a1a2e !important;
        box-shadow: none !important;
        transition: border-color 0.25s, box-shadow 0.25s, background 0.25s !important;
    }
    .chat-input-bar .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        background: #fff !important;
        box-shadow: 0 0 0 4px rgba(167,139,250,0.18) !important;
    }
    .chat-input-bar .stTextInput > div > div > input::placeholder { color: #b0a8c4; }
    .chat-input-bar .stTextInput label { display: none !important; }

    /* Send button */
    .stButton > button[kind="primary"],
    .chat-input-bar .stButton > button {
        width: 38px !important; height: 38px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        background-size: 200% 200% !important;
        border: none !important; color: #fff !important;
        font-size: 1rem !important;
        box-shadow: 0 3px 10px rgba(124,58,237,0.4) !important;
        transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1),
                    box-shadow 0.2s, background-position 0.4s !important;
        animation: send-btn-glow 3s ease infinite !important;
    }
    @keyframes send-btn-glow {
        0%,100% { box-shadow: 0 3px 10px rgba(124,58,237,0.4); }
        50%      { box-shadow: 0 4px 18px rgba(124,58,237,0.62); }
    }
    .stButton > button:hover {
        transform: scale(1.12) rotate(-5deg) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.55) !important;
        background-position: 100% 50% !important;
    }
    .stButton > button:active { transform: scale(0.95) !important; }

    /* ── General buttons (login/auth) ── */
    .stButton > button {
        border-radius: 999px !important;
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: #fff !important; border: none !important;
        font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1),
                    box-shadow 0.2s !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 6px 18px rgba(124,58,237,0.42) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 2px solid #f0ebfa;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600; font-size: 0.85rem; color: #6b7280 !important;
        padding: 0.4rem 0.75rem !important;
        transition: color 0.2s, background 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        color: #7c3aed !important;
        background: rgba(124,58,237,0.05) !important;
    }

    /* ── Text inputs (forms) ── */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e2d9f3 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.92) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(240,235,250,0.9) !important;
        animation: sidebar-slide 0.5s cubic-bezier(0.22,1,0.36,1) both !important;
    }
    @keyframes sidebar-slide {
        from { transform: translateX(-20px); opacity: 0; }
        to   { transform: translateX(0);     opacity: 1; }
    }
    section[data-testid="stSidebar"] * { color: #1a1a2e !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c4b5fd !important;
        border-radius: 16px !important;
        background: rgba(245,243,255,0.6) !important;
        transition: border-color 0.25s, background 0.25s !important;
        animation: fade-in 0.5s ease 0.15s both;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #7c3aed !important;
        background: rgba(245,243,255,0.9) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #7c3aed !important; }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #c4b5fd, #a78bfa);
        border-radius: 4px;
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
    "username": None,          # display name chosen at signup
    "auth_mode": "login",      # "login" | "signup" — controls which panel shows
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
    mode = st.session_state.get("auth_mode", "login")

    # ── Animated full-page shell ──────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Particles canvas */
    #particles-canvas {
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 0;
    }

    /* Login page layout */
    .auth-page {
        position: relative; z-index: 1;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 90vh; gap: 0;
    }

    /* Brand header above card */
    .brand-header {
        text-align: center;
        margin-bottom: 1.6rem;
        animation: brand-drop 0.6s cubic-bezier(0.22,1,0.36,1) both;
    }
    @keyframes brand-drop {
        from { opacity:0; transform: translateY(-22px) scale(0.94); }
        to   { opacity:1; transform: translateY(0)     scale(1);    }
    }
    .brand-logo {
        width: 56px; height: 56px;
        background: linear-gradient(135deg, #7c3aed, #a855f7, #6d28d9);
        background-size: 200% 200%;
        border-radius: 18px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.7rem;
        box-shadow: 0 8px 28px rgba(124,58,237,0.42);
        margin-bottom: 0.6rem;
        animation: logo-spin-gradient 5s ease infinite, logo-bounce 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.1s both;
    }
    @keyframes logo-spin-gradient {
        0%,100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }
    @keyframes logo-bounce {
        from { transform: scale(0) rotate(-25deg); }
        to   { transform: scale(1) rotate(0deg);   }
    }
    .brand-name {
        font-size: 1.65rem; font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em; line-height: 1.1;
        animation: fade-in 0.7s ease 0.25s both;
    }
    .brand-tagline {
        font-size: 0.82rem; color: #7c6fa0; margin-top: 0.2rem;
        letter-spacing: 0.06em; text-transform: uppercase;
        animation: fade-in 0.7s ease 0.4s both;
    }

    /* Card */
    .auth-card {
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border-radius: 26px;
        padding: 2rem 1.8rem 1.6rem;
        width: 100%; max-width: 420px;
        box-shadow: 0 20px 60px rgba(100,60,200,0.18),
                    0 4px 16px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.9);
        border: 1px solid rgba(226,217,243,0.75);
        animation: card-rise 0.55s cubic-bezier(0.22,1,0.36,1) 0.1s both;
        transition: box-shadow 0.3s ease;
    }
    .auth-card:hover {
        box-shadow: 0 28px 72px rgba(100,60,200,0.22),
                    0 6px 20px rgba(0,0,0,0.08),
                    inset 0 1px 0 rgba(255,255,255,0.9);
    }
    @keyframes card-rise {
        from { opacity:0; transform: translateY(30px) scale(0.96); }
        to   { opacity:1; transform: translateY(0)    scale(1);    }
    }

    /* Toggle pills */
    .auth-toggle {
        display: flex; background: #f0ebfa;
        border-radius: 999px; padding: 4px; gap: 2px;
        margin-bottom: 1.5rem;
        animation: fade-in 0.5s ease 0.3s both;
    }
    .auth-toggle-btn {
        flex: 1; border: none; cursor: pointer;
        border-radius: 999px; padding: 0.45rem 0;
        font-size: 0.85rem; font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        transition: background 0.25s, color 0.25s,
                    transform 0.2s cubic-bezier(0.34,1.56,0.64,1),
                    box-shadow 0.25s;
        color: #7c6fa0;
        background: transparent;
    }
    .auth-toggle-btn.active {
        background: #fff;
        color: #7c3aed;
        box-shadow: 0 2px 10px rgba(124,58,237,0.18);
        transform: scale(1.03);
    }
    .auth-toggle-btn:hover:not(.active) {
        background: rgba(255,255,255,0.5);
        transform: scale(1.01);
    }

    /* Form title */
    .auth-form-title {
        font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
        margin-bottom: 0.25rem;
        animation: title-slide 0.4s cubic-bezier(0.22,1,0.36,1) both;
    }
    .auth-form-sub {
        font-size: 0.82rem; color: #7c6fa0;
        margin-bottom: 1.2rem;
        animation: title-slide 0.4s cubic-bezier(0.22,1,0.36,1) 0.06s both;
    }
    @keyframes title-slide {
        from { opacity:0; transform: translateX(-10px); }
        to   { opacity:1; transform: translateX(0);     }
    }

    /* Field labels */
    .field-label {
        font-size: 0.78rem; font-weight: 600;
        color: #5b4d80; margin-bottom: 0.25rem;
        letter-spacing: 0.03em; text-transform: uppercase;
    }

    /* Divider */
    .auth-divider {
        display: flex; align-items: center; gap: 0.75rem;
        margin: 1rem 0; color: #b0a8c4; font-size: 0.78rem;
    }
    .auth-divider::before, .auth-divider::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, transparent, #e2d9f3, transparent);
    }

    /* Guest button */
    .guest-btn {
        width: 100%; border: 1.5px solid #c4b5fd;
        background: transparent; border-radius: 999px;
        padding: 0.52rem; font-size: 0.85rem;
        font-weight: 600; color: #7c3aed;
        cursor: pointer; font-family: 'DM Sans', sans-serif;
        transition: background 0.2s, transform 0.2s cubic-bezier(0.34,1.56,0.64,1);
        margin-top: 0.1rem;
    }
    .guest-btn:hover { background: #f5f3ff; transform: translateY(-1px); }

    /* Input fields — staggered entrance */
    .stTextInput:nth-child(1) { animation: field-in 0.35s ease 0.2s both; }
    .stTextInput:nth-child(2) { animation: field-in 0.35s ease 0.28s both; }
    .stTextInput:nth-child(3) { animation: field-in 0.35s ease 0.36s both; }
    .stTextInput:nth-child(4) { animation: field-in 0.35s ease 0.44s both; }
    @keyframes field-in {
        from { opacity:0; transform: translateY(8px); }
        to   { opacity:1; transform: translateY(0);   }
    }

    /* Shimmer on active card border */
    .auth-card::after {
        content: '';
        position: absolute; inset: 0;
        border-radius: 26px;
        background: linear-gradient(135deg,
            rgba(167,139,250,0.12) 0%,
            transparent 50%,
            rgba(124,58,237,0.08) 100%);
        pointer-events: none;
    }
    .auth-card { position: relative; overflow: hidden; }

    /* Floating sparkles */
    .sparkle {
        position: fixed; pointer-events: none; z-index: 0;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: rgba(167,139,250,0.55);
        animation: sparkle-float linear infinite;
    }
    @keyframes sparkle-float {
        0%   { transform: translateY(0) scale(1);   opacity: 0.7; }
        50%  { opacity: 1; }
        100% { transform: translateY(-120vh) scale(0.3); opacity: 0; }
    }
    </style>

    <!-- Floating sparkle particles -->
    <canvas id="particles-canvas"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('particles-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        window.addEventListener('resize', () => {
            canvas.width  = window.innerWidth;
            canvas.height = window.innerHeight;
        });
        const particles = Array.from({length: 38}, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2.8 + 0.8,
            dx: (Math.random() - 0.5) * 0.35,
            dy: -(Math.random() * 0.55 + 0.2),
            o: Math.random() * 0.5 + 0.2,
            hue: Math.random() > 0.5 ? 270 : 290,
        }));
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${p.hue},70%,70%,${p.o})`;
                ctx.fill();
                p.x += p.dx; p.y += p.dy;
                p.o -= 0.0015;
                if (p.y < -10 || p.o <= 0) {
                    p.x = Math.random() * canvas.width;
                    p.y = canvas.height + 10;
                    p.o = Math.random() * 0.5 + 0.25;
                    p.r = Math.random() * 2.8 + 0.8;
                }
            });
            requestAnimationFrame(draw);
        }
        draw();
    })();
    </script>
    """, unsafe_allow_html=True)

    # ── Brand header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">📘</div><br>
        <div class="brand-name">SlideSense</div>
        <div class="brand-tagline">Smart Learning · Smart Vision · Smart AI</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Card ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    # Toggle: Login / Sign Up
    toggle_html = f"""
    <div class="auth-toggle">
        <button class="auth-toggle-btn {"active" if mode == "login" else ""}"
            onclick="window.parent.document.querySelector('[data-testid=\"stForm\"]')"
        >Sign In</button>
        <button class="auth-toggle-btn {"active" if mode == "signup" else ""}"
        >Create Account</button>
    </div>"""
    st.markdown(toggle_html, unsafe_allow_html=True)

    # Streamlit toggle buttons (invisible, drive state)
    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("Sign In", key="toggle_login",
                     type="primary" if mode == "login" else "secondary"):
            st.session_state["auth_mode"] = "login"
            st.rerun()
    with col_r:
        if st.button("Create Account", key="toggle_signup",
                     type="primary" if mode == "signup" else "secondary"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

    st.markdown("---", unsafe_allow_html=False)

    # ── LOGIN FORM ────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown('<div class="auth-form-title">Welcome back 👋</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-form-sub">Sign in to continue to SlideSense</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="cooluser42", key="login_user")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")

        if st.button("Sign In →", key="btn_login", use_container_width=True):
            if not username or not password:
                st.warning("Please fill in all fields.")
            else:
                with st.spinner("Signing in…"):
                    err = sign_in(username, password)
                if err:
                    st.error(err)
                else:
                    st.rerun()

        st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)
        if st.button("Continue as Guest 👤", key="btn_guest_login", use_container_width=True):
            st.session_state["guest"] = True
            st.session_state["username"] = "Guest"
            st.rerun()

    # ── SIGN UP FORM ──────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="auth-form-title">Create account ✨</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-form-sub">Join SlideSense — it\'s free</div>', unsafe_allow_html=True)

        su_username = st.text_input("Username", placeholder="cooluser42", key="signup_username")
        su_pw       = st.text_input("Password (min 6 chars)", type="password", placeholder="••••••••", key="signup_password")
        su_pw2      = st.text_input("Confirm password", type="password", placeholder="••••••••", key="signup_password2")

        if st.button("Create Account →", key="btn_signup", use_container_width=True):
            if not su_username or not su_pw or not su_pw2:
                st.warning("Please fill in all fields.")
            elif su_pw != su_pw2:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating account…"):
                    err, auto_logged_in = sign_up(su_username, su_pw)
                if err:
                    st.error(f"Sign-up failed: {err}")
                elif auto_logged_in:
                    st.rerun()
                else:
                    st.success("✅ Account created! You can now sign in.")

        st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)
        if st.button("Continue as Guest 👤", key="btn_guest_signup", use_container_width=True):
            st.session_state["guest"] = True
            st.session_state["username"] = "Guest"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close auth-card


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
