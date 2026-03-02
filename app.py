import streamlit as st
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    def st_lottie(*args, **kwargs): pass

from datetime import datetime
import uuid
import requests
import time
import json
from PyPDF2 import PdfReader
from PIL import Image
import base64
import os
import streamlit.components.v1 as components

import firebase_admin
from firebase_admin import credentials, auth, firestore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="SlideSense AI", page_icon="🧠", layout="wide")


# -------------------- LOGO --------------------
def get_logo_base64():
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

LOGO_B64 = get_logo_base64()

def logo_img_tag(size=52):
    if LOGO_B64:
        return f'<img src="data:image/png;base64,{LOGO_B64}" width="{size}" height="{size}" style="border-radius:10px;object-fit:contain;">'
    return f'<span style="font-size:{size//2}px;">🧠</span>'


# -------------------- CSS --------------------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

.logo-bar {display:flex;align-items:center;gap:14px;padding:18px 0 10px 0;}
.logo-text {
    font-size:2.2rem;font-weight:900;letter-spacing:5px;
    background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin:0;line-height:1.1;
}
.logo-tagline {font-size:0.7rem;color:#888;letter-spacing:2.5px;text-transform:uppercase;margin:0;}

.sidebar-logo {display:flex;align-items:center;gap:10px;padding:8px 0 14px 0;
    border-bottom:1px solid rgba(108,99,255,0.2);margin-bottom:12px;}
.sidebar-logo-text {font-size:1rem;font-weight:800;letter-spacing:3px;
    background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
</style>
""", unsafe_allow_html=True)


# -------------------- FIREBASE --------------------
if not firebase_admin._apps:
    raw_key = st.secrets["firebase"]["private_key"]
    private_key = raw_key.replace("\\n", "\n")
    firebase_config = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": private_key,
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    }
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# -------------------- SESSION --------------------
defaults = {
    "authenticated": False,
    "is_guest": False,
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "vector_db": None,
    "guest_messages": [],
    "logo_typed": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- HELPERS --------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def type_text_logo():
    if st.session_state.logo_typed:
        st.markdown(f"""
            <div class="logo-bar">
                {logo_img_tag(56)}
                <div>
                    <p class="logo-text">SLIDESENSE</p>
                    <p class="logo-tagline">AI · PDF · Image Analyzer</p>
                </div>
            </div>""", unsafe_allow_html=True)
        return
    placeholder = st.empty()
    full = "SLIDESENSE"
    out = ""
    for c in full:
        out += c
        placeholder.markdown(f"""
            <div class="logo-bar">
                {logo_img_tag(56)}
                <div>
                    <p class="logo-text">{out}</p>
                    <p class="logo-tagline">AI · PDF · Image Analyzer</p>
                </div>
            </div>""", unsafe_allow_html=True)
        time.sleep(0.07)
    st.session_state.logo_typed = True


def render_logo():
    st.markdown(f"""
        <div class="logo-bar">
            {logo_img_tag(56)}
            <div>
                <p class="logo-text">SLIDESENSE</p>
                <p class="logo-tagline">AI · PDF · Image Analyzer</p>
            </div>
        </div>""", unsafe_allow_html=True)


def render_answer_with_copy(answer: str) -> None:
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(f"""
        <script>
        function copyText() {{
            var text = {safe_text};
            try {{
                var el = document.createElement('textarea');
                el.value = text;
                el.setAttribute('readonly','');
                el.style.position='absolute';
                el.style.left='-9999px';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                var btn = document.getElementById('copybtn');
                btn.innerText = '✅ Copied!';
                btn.style.borderColor='#48CAE4';
                btn.style.color='#48CAE4';
                setTimeout(function(){{
                    btn.innerText='📋 Copy';
                    btn.style.borderColor='#6C63FF';
                    btn.style.color='#6C63FF';
                }}, 2000);
            }} catch(e) {{ window.prompt("Copy:", text); }}
        }}
        </script>
        <button id="copybtn" onclick="copyText();"
            style="margin-top:4px;padding:5px 14px;border-radius:6px;
                   border:1px solid #6C63FF;color:#6C63FF;
                   background:transparent;cursor:pointer;font-size:12px;">
            📋 Copy
        </button>""", height=45)


# -------------------- AUTH --------------------
def signup(email, password):
    try:
        return auth.create_user(email=email, password=password)
    except auth.EmailAlreadyExistsError:
        st.error("An account with this email already exists.")
        return None
    except Exception as e:
        st.error(f"Signup error: {e}")
        return None


def login(email, password):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            st.error("FIREBASE_WEB_API_KEY missing from secrets.")
            return None
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        data = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True}).json()
        if "error" in data:
            msg = data["error"]["message"]
            st.error("❌ Invalid email or password." if msg in ("EMAIL_NOT_FOUND","INVALID_LOGIN_CREDENTIALS","INVALID_PASSWORD") else f"Login failed: {msg}")
            return None
        return auth.get_user_by_email(email)
    except Exception as e:
        st.error(f"Login error: {e}")
        return None


def reset_password(email):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            return False
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"
        data = requests.post(url, json={"requestType": "PASSWORD_RESET", "email": email}).json()
        if "error" in data:
            st.error(f"Reset failed: {data['error']['message']}")
            return False
        return True
    except Exception as e:
        st.error(f"Reset error: {e}")
        return False


# -------------------- FIRESTORE --------------------
def get_next_chat_number(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats").where("mode","==",mode).stream()
    return sum(1 for _ in chats) + 1


def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    num = get_next_chat_number(user_id, mode)
    icon = "📘" if mode == "PDF" else "🖼"
    title = f"{icon} Chat {num}"
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode, "created_at": datetime.utcnow(), "title": title
    })
    return chat_id


def update_chat_title(user_id, chat_id, first_question):
    short = first_question.strip()[:35]
    if len(first_question.strip()) > 35:
        short += "..."
    db.collection("users").document(user_id).collection("chats").document(chat_id).update({"title": short})


def save_message(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role, "content": content, "timestamp": datetime.utcnow()
        })


def load_user_chats(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode","==",mode) \
        .order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    return [(doc.id, doc.to_dict()["title"]) for doc in chats]


def load_messages(user_id, chat_id):
    msgs = db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages") \
        .order_by("timestamp").stream()
    return [(doc.to_dict()["role"], doc.to_dict()["content"]) for doc in msgs]


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )


# ==================== AUTH SCREEN ====================
if not st.session_state.authenticated and not st.session_state.is_guest:

    col_anim, col_form = st.columns([1, 1], gap="large")

    with col_anim:
        type_text_logo()
        lottie_data = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
        if lottie_data:
            st_lottie(lottie_data, height=360, key="login_anim")
        else:
            st.markdown(f'<div style="text-align:center;padding-top:60px">{logo_img_tag(120)}</div>', unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#888;font-size:0.85rem;margin-top:8px;'>Analyze PDFs & Images using AI</div>", unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='padding-top:55px'></div>", unsafe_allow_html=True)
        tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "📝 Sign Up", "👤 Guest"])

        with tab_login:
            st.markdown("#### Welcome back!")
            le = st.text_input("Email", key="login_email", placeholder="you@example.com")
            lp = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
            cb, cf = st.columns(2)
            with cb:
                if st.button("Login", use_container_width=True, type="primary"):
                    if not le or not lp:
                        st.warning("Please fill in all fields.")
                    else:
                        with st.spinner("Logging in..."):
                            user = login(le, lp)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.is_guest = False
                            st.session_state.user_id = user.uid
                            st.session_state.email = le
                            st.rerun()
            with cf:
                if st.button("Forgot Password?", use_container_width=True):
                    if not le:
                        st.warning("Enter your email above first.")
                    else:
                        with st.spinner("Sending..."):
                            sent = reset_password(le)
                        if sent:
                            st.success("📧 Reset email sent!")

        with tab_signup:
            st.markdown("#### Create your account")
            ne = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            np_ = st.text_input("Password", type="password", key="signup_password", placeholder="Min 6 characters")
            cp = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Re-enter password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not ne or not np_ or not cp:
                    st.warning("Please fill in all fields.")
                elif len(np_) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif np_ != cp:
                    st.error("❌ Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        user = signup(ne, np_)
                    if user:
                        st.success("✅ Account created! Go to the Login tab.")

        with tab_guest:
            st.markdown("#### Continue without an account")
            st.info("**Guest mode:** Full PDF & Image AI — no account needed.\n\n❌ Chats not saved, clears on refresh.")
            if st.button("👤 Continue as Guest", use_container_width=True, type="primary"):
                st.session_state.is_guest = True
                st.session_state.user_id = "guest"
                st.session_state.email = "Guest"
                st.session_state.guest_messages = []
                st.rerun()

    st.stop()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-logo">
            {logo_img_tag(36)}
            <span class="sidebar-logo-text">SLIDESENSE</span>
        </div>""", unsafe_allow_html=True)

    if st.session_state.is_guest:
        st.warning("👤 Guest Mode")
        st.caption("Chats are not saved.")
    else:
        st.success(f"👤 {st.session_state.email}")

    if st.button("🚪 Logout", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

    st.divider()
    mode = st.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
    st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"
    st.divider()

    if st.session_state.is_guest:
        st.markdown("### 💬 Session Chat")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.guest_messages = []
            st.session_state.vector_db = None
            st.rerun()
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = "guest_session"
    else:
        st.markdown("### 💬 Your Chats")
        user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)
        for chat_id, title in user_chats:
            c1, c2 = st.columns([4, 1])
            if c1.button(title, key=f"open_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.session_state.vector_db = None
                st.rerun()
            if c2.button("🗑", key=f"del_{chat_id}"):
                db.collection("users").document(st.session_state.user_id) \
                  .collection("chats").document(chat_id).delete()
                if st.session_state.current_chat_id == chat_id:
                    st.session_state.current_chat_id = None
                st.rerun()
        if st.button("➕ New Chat", use_container_width=True):
            cid = create_new_chat(st.session_state.user_id, st.session_state.mode)
            st.session_state.current_chat_id = cid
            st.session_state.vector_db = None
            st.rerun()


# ==================== WELCOME SCREEN (after login, before chat) ====================
if not st.session_state.current_chat_id:
    st.markdown("""
    <style>
    .ss-wrap {
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        padding:60px 0 40px 0;
    }
    .ss-svg { width:220px; height:220px; }

    .slide-back { animation: ssSlideIn 0.8s ease-out forwards; }
    .slide-front { animation: ssSlideIn 0.8s 0.2s ease-out forwards; opacity:0; }

    .ss-node { animation: ssPulse 2s infinite ease-in-out; }
    .ss-node-1 { animation-delay:0.5s; }
    .ss-node-2 { animation-delay:0.8s; }
    .ss-node-3 { animation-delay:1.1s; }

    .ss-arrow {
        stroke-dasharray:100; stroke-dashoffset:100;
        animation: ssDraw 1s 1.2s cubic-bezier(0.16,1,0.3,1) forwards;
    }
    .ss-brand {
        font-size:44px; font-weight:900; letter-spacing:5px;
        background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; margin-top:14px;
        opacity:0; animation: ssFadeUp 0.8s 1.5s forwards;
    }
    .ss-tagline {
        font-size:14px; color:#888; letter-spacing:2px;
        text-transform:uppercase; margin-top:6px;
        opacity:0; animation: ssFade 1s 2s forwards;
    }
    .ss-hint {
        margin-top:28px; font-size:14px; color:#6C63FF;
        opacity:0; animation: ssFade 1s 2.5s forwards;
        background:rgba(108,99,255,0.08);
        border:1px solid rgba(108,99,255,0.3);
        border-radius:10px; padding:10px 24px;
    }

    @keyframes ssSlideIn {
        from { transform:translateX(-30px); opacity:0; }
        to   { transform:translateX(0);     opacity:1; }
    }
    @keyframes ssPulse {
        0%,100% { fill:#0ea5e9; }
        50%      { fill:#38bdf8; filter:drop-shadow(0 0 3px #38bdf8); }
    }
    @keyframes ssDraw { to { stroke-dashoffset:0; } }
    @keyframes ssFadeUp {
        from { opacity:0; transform:translateY(12px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes ssFade { to { opacity:1; } }
    </style>

    <div class="ss-wrap">
        <svg class="ss-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <path class="slide-back"
                d="M30 20 H65 A5 5 0 0 1 70 25 V75 A5 5 0 0 1 65 80 H30 A5 5 0 0 1 25 75 V25 A5 5 0 0 1 30 20"
                fill="#0369a1"/>
            <path class="slide-front"
                d="M35 25 H70 A5 5 0 0 1 75 30 V80 A5 5 0 0 1 70 85 H35 A5 5 0 0 1 30 80 V30 A5 5 0 0 1 35 25"
                fill="#0ea5e9"/>
            <circle cx="55" cy="55" r="22" fill="white" stroke="#0c4a6e" stroke-width="3"/>
            <circle class="ss-node ss-node-1" cx="48" cy="52" r="2" fill="#0ea5e9"/>
            <circle class="ss-node ss-node-2" cx="55" cy="48" r="2" fill="#0ea5e9"/>
            <circle class="ss-node ss-node-3" cx="53" cy="58" r="2" fill="#0ea5e9"/>
            <line x1="48" y1="52" x2="55" y2="48" stroke="#cbd5e1" stroke-width="0.5"/>
            <line x1="55" y1="48" x2="53" y2="58" stroke="#cbd5e1" stroke-width="0.5"/>
            <path class="ss-arrow"
                d="M45 65 L75 35 M75 35 L68 35 M75 35 L75 42"
                stroke="#0ea5e9" stroke-width="4"
                stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
        <div class="ss-brand">SLIDESENSE</div>
        <div class="ss-tagline">PDF &amp; Image Q&amp;A</div>
        <div class="ss-hint">👈 Select ➕ New Chat from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)


# ==================== CHAT SCREEN ====================
else:
    img_file = None
    camera_file = None

    if st.session_state.mode == "PDF":
        ac, tc = st.columns([1, 4])
        with ac:
            components.html("""<!DOCTYPE html>
<html><head><style>
body {
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  background:transparent; margin:0; padding:8px 0;
  font-family:'Segoe UI',sans-serif;
}
.pdf-card {
  position:relative; width:90px; height:120px;
  background:rgba(255,255,255,0.05);
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:10px;
  box-shadow:0 8px 32px rgba(0,0,0,0.37);
  overflow:hidden;
  display:flex; flex-direction:column;
  padding:14px 10px; box-sizing:border-box;
}
.skeleton-line {
  height:6px; background:#334155;
  border-radius:4px; margin-bottom:9px;
  position:relative; overflow:hidden; width:100%;
}
.skeleton-line.short { width:55%; }
.skeleton-line::after {
  content:""; position:absolute;
  top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.22),transparent);
  animation:shimmer 1.5s infinite;
}
.scanner-beam {
  position:absolute; top:-50px; left:0;
  width:100%; height:40px;
  background:linear-gradient(to bottom,transparent,rgba(56,189,248,0.5));
  border-bottom:2px solid #38bdf8;
  box-shadow:0 5px 15px rgba(56,189,248,0.4);
  animation:scan 2.5s infinite ease-in-out alternate;
  z-index:10;
}
.badge {
  position:absolute; bottom:8px; right:8px;
  background:linear-gradient(135deg,#ef4444,#b91c1c);
  color:white; font-size:8px; letter-spacing:1px;
  font-weight:700; padding:3px 6px;
  border-radius:5px;
  box-shadow:0 4px 10px rgba(239,68,68,0.3);
}
@keyframes scan {
  0%  { top:-50px; opacity:0; }
  10% { opacity:1; }
  90% { opacity:1; }
  100%{ top:100%; opacity:0; }
}
@keyframes shimmer { 100%{ left:100%; } }
</style></head>
<body>
  <div class="pdf-card">
    <div class="scanner-beam"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line short"></div>
    <div class="skeleton-line"></div>
    <div class="badge">PDF</div>
  </div>
</body></html>""", height=140)
        with tc:
            st.markdown("## 📘 PDF Analyzer")
            st.caption("Upload a PDF and ask questions about its content.")
        st.divider()

        pdf = st.file_uploader("Upload PDF", type="pdf")
        if pdf and st.session_state.vector_db is None:
            # Show custom PDF scanning animation using components.html (supports CSS animations)
            anim_slot = st.empty()
            with anim_slot:
                components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  body {
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:transparent; margin:0; padding:20px 0;
    font-family:'Segoe UI',Roboto,sans-serif;
  }
  .pdf-card {
    position:relative; width:140px; height:190px;
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:12px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);
    overflow:hidden;
    display:flex; flex-direction:column;
    padding:20px 15px; box-sizing:border-box;
  }
  .skeleton-line {
    height:8px; background:#334155;
    border-radius:4px; margin-bottom:12px;
    position:relative; overflow:hidden; width:100%;
  }
  .skeleton-line.short { width:60%; }
  .skeleton-line::after {
    content:""; position:absolute;
    top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25),transparent);
    animation:shimmer 1.5s infinite;
  }
  .scanner-beam {
    position:absolute; top:-50px; left:0;
    width:100%; height:50px;
    background:linear-gradient(to bottom,transparent,rgba(56,189,248,0.6));
    border-bottom:2px solid #38bdf8;
    box-shadow:0 5px 20px rgba(56,189,248,0.5);
    animation:scan 2.5s infinite ease-in-out alternate;
    z-index:10;
  }
  .pdf-badge {
    position:absolute; bottom:12px; right:12px;
    background:linear-gradient(135deg,#ef4444,#b91c1c);
    color:white; font-size:10px; letter-spacing:1px;
    font-weight:700; padding:4px 8px;
    border-radius:6px;
    box-shadow:0 4px 10px rgba(239,68,68,0.4);
  }

  @keyframes scan {
    0%   { top:-50px; opacity:0; }
    10%  { opacity:1; }
    90%  { opacity:1; }
    100% { top:100%; opacity:0; }
  }
  @keyframes shimmer { 100% { left:100%; } }

</style>
</head>
<body>
  <div class="pdf-card">
    <div class="scanner-beam"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line short"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line short"></div>
    <div class="skeleton-line"></div>
    <div class="pdf-badge">PDF</div>
  </div>
</body>
</html>
""", height=300)

            reader = PdfReader(pdf)
            text = "".join(p.extract_text() or "" for p in reader.pages)
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_text(text)
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_db = FAISS.from_texts(chunks, emb)

            # Clear animation and show success
            anim_slot.empty()
            st.success("✅ PDF processed! Ask your questions below.")

    else:
        ac, tc = st.columns([1, 4])
        with ac:
            components.html("""<!DOCTYPE html>
<html><head><style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  display:flex; align-items:center; justify-content:center;
  background:transparent;
  font-family:'Segoe UI', sans-serif;
  height:130px; overflow:hidden;
}
.scene {
  position:relative; width:200px; height:120px;
}

/* Image card */
.img-card {
  position:absolute; left:0; top:10px;
  width:65px; height:90px;
  border-radius:10px;
  background:linear-gradient(160deg,#60a5fa 0%,#3b82f6 40%,#1e40af 100%);
  box-shadow:0 6px 20px rgba(59,130,246,0.4);
  overflow:hidden;
  animation:cardFloat 3s ease-in-out infinite;
}
/* mountain svg inside card */
.img-card svg { width:100%; height:100%; }

/* Search bar */
.search-bar {
  position:absolute; left:38px; top:4px;
  width:150px; height:28px;
  background:white;
  border-radius:20px;
  box-shadow:0 4px 15px rgba(0,0,0,0.15);
  display:flex; align-items:center;
  padding:0 8px; gap:5px;
  animation:slideInDown 0.6s ease-out forwards;
  opacity:0;
}
.search-icon { color:#9ca3af; font-size:11px; flex-shrink:0; }
.search-text {
  font-size:9px; color:#374151; font-weight:500;
  white-space:nowrap; overflow:hidden;
  border-right:1.5px solid #374151;
  width:0;
  animation:typeText 1s 0.8s steps(12) forwards;
}
.search-btn {
  margin-left:auto; width:20px; height:20px;
  background:#3b82f6; border-radius:6px;
  display:flex; align-items:center; justify-content:center;
  font-size:9px; color:white; flex-shrink:0;
  animation:btnPop 0.3s 1.8s ease-out both;
  transform:scale(0);
}

/* Response bubble */
.response-bubble {
  position:absolute; right:0; bottom:0;
  width:100px; height:48px;
  background:linear-gradient(135deg,#34d399,#10b981);
  border-radius:10px 10px 10px 2px;
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 6px 20px rgba(16,185,129,0.35);
  animation:bubblePop 0.5s 2s cubic-bezier(0.175,0.885,0.32,1.275) both;
  transform:scale(0); transform-origin:bottom left;
}
.response-bubble::after {
  content:"";
  position:absolute; bottom:-8px; left:12px;
  width:0; height:0;
  border-left:8px solid transparent;
  border-right:0px solid transparent;
  border-top:8px solid #10b981;
}
.response-inner {
  background:rgba(255,255,255,0.25);
  border-radius:6px; padding:5px 10px;
  font-size:9px; font-weight:600; color:white;
  letter-spacing:0.3px;
  animation:fadeIn 0.3s 2.4s ease-out both; opacity:0;
}

@keyframes cardFloat {
  0%,100% { transform:translateY(0px); }
  50%      { transform:translateY(-5px); }
}
@keyframes slideInDown {
  from { opacity:0; transform:translateY(-10px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes typeText {
  from { width:0; }
  to   { width:72px; border-right:none; }
}
@keyframes btnPop {
  to { transform:scale(1); }
}
@keyframes bubblePop {
  to { transform:scale(1); }
}
@keyframes fadeIn {
  to { opacity:1; }
}
</style></head>
<body>
<div class="scene">
  <!-- Image Card with mountain SVG -->
  <div class="img-card">
    <svg viewBox="0 0 65 90" xmlns="http://www.w3.org/2000/svg">
      <rect width="65" height="90" fill="#bfdbfe"/>
      <polygon points="10,80 32,30 54,80" fill="#1e3a8a" opacity="0.9"/>
      <polygon points="0,80 18,45 36,80" fill="#1e40af" opacity="0.8"/>
      <polygon points="28,80 45,42 65,80" fill="#1d4ed8" opacity="0.7"/>
      <polygon points="24,42 32,28 40,42" fill="white" opacity="0.9"/>
      <rect y="60" width="65" height="30" fill="#1e3a8a" opacity="0.5"/>
      <!-- clouds -->
      <ellipse cx="12" cy="18" rx="8" ry="4" fill="white" opacity="0.7"/>
      <ellipse cx="50" cy="14" rx="6" ry="3" fill="white" opacity="0.6"/>
    </svg>
  </div>

  <!-- Search Bar -->
  <div class="search-bar">
    <span class="search-icon">🔍</span>
    <span class="search-text">What is this?</span>
    <div class="search-btn">🔍</div>
  </div>

  <!-- Response Bubble -->
  <div class="response-bubble">
    <div class="response-inner">A Mountain</div>
  </div>
</div>
</body></html>""", height=140)
        with tc:
            st.markdown("## 🖼 Image Q&A")
            st.caption("Upload an image or use your live camera.")
        st.divider()

        img_source = st.radio("Image Source", ["📁 Upload Image", "📷 Live Camera"],
                              horizontal=True, key="img_source")

        if img_source == "📁 Upload Image":
            img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
            if img_file:
                st.image(Image.open(img_file).convert("RGB"), use_container_width=True)
        else:
            st.info("📷 Point your camera and click **'Take Photo'** to capture.")
            camera_file = st.camera_input("Take a photo")
            if camera_file:
                st.image(Image.open(camera_file).convert("RGB"), use_container_width=True)

    # ---- Messages ----
    messages = st.session_state.guest_messages if st.session_state.is_guest else \
               load_messages(st.session_state.user_id, st.session_state.current_chat_id)

    st.markdown("### 💬 Conversation")
    for role, content in messages:
        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "assistant":
                render_answer_with_copy(content)
            else:
                st.markdown(content)

    question = st.chat_input("Ask something...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("user", question))
        else:
            existing_msgs = load_messages(st.session_state.user_id, st.session_state.current_chat_id)
            if len(existing_msgs) == 0:
                update_chat_title(st.session_state.user_id, st.session_state.current_chat_id, question)
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "user", question)

        if st.session_state.mode == "PDF":
            if st.session_state.vector_db is None:
                answer = "⚠️ Please upload a PDF first."
            else:
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_db.similarity_search(question, k=6)
                    chain = create_stuff_documents_chain(load_llm(), ChatPromptTemplate.from_template(
                        "Context:\n{context}\n\nQuestion:\n{input}\n\nIf not found say: Information not found in document."
                    ))
                    result = chain.invoke({"context": docs, "input": question})
                    answer = result if isinstance(result, str) else result.get("output_text", str(result))
        else:
            active_image = img_file or camera_file
            if not active_image:
                answer = "⚠️ Please upload an image or take a photo first."
            else:
                with st.spinner("🖼 Analyzing image..."):
                    image_bytes = active_image.getvalue()
                    encoded = base64.b64encode(image_bytes).decode("utf-8")
                    mime = "image/jpeg" if (camera_file and not img_file) else \
                           ("image/png" if img_file.name.lower().endswith(".png") else "image/jpeg")
                    response = load_llm().invoke([HumanMessage(content=[
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ])])
                    answer = response.content

        with st.chat_message("assistant"):
            render_answer_with_copy(answer)

        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("assistant", answer))
        else:
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "assistant", answer)

        st.rerun()
