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

# Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="SlideSense AI", page_icon="assets/logo.png", layout="wide")

# -------------------- LOAD LOGO --------------------
def get_logo_base64():
    """Load logo from assets/logo.png and return base64 string."""
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

LOGO_B64 = get_logo_base64()

def logo_img_tag(size=52):
    """Returns an <img> tag for the logo, or fallback emoji."""
    if LOGO_B64:
        return f'<img src="data:image/png;base64,{LOGO_B64}" width="{size}" height="{size}" style="border-radius:10px; object-fit:contain;">'
    return f'<span style="font-size:{size//2}px;">📊</span>'


st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .logo-bar {
        display: flex; align-items: center; gap: 14px;
        padding: 18px 0 10px 0;
    }
    .logo-text {
        font-size: 2.2rem; font-weight: 900; letter-spacing: 5px;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0; line-height: 1.1;
    }
    .logo-tagline {
        font-size: 0.7rem; color: #888; letter-spacing: 2.5px;
        text-transform: uppercase; margin: 0;
    }
    .sidebar-logo {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 0 14px 0;
        border-bottom: 1px solid rgba(108,99,255,0.2);
        margin-bottom: 12px;
    }
    .sidebar-logo-text {
        font-size: 1rem; font-weight: 800; letter-spacing: 3px;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- FIREBASE INIT --------------------
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


# -------------------- SESSION INIT --------------------
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
    """Animated typing logo for auth page."""
    if st.session_state.logo_typed:
        st.markdown(f"""
            <div class="logo-bar">
                {logo_img_tag(56)}
                <div>
                    <p class="logo-text">SLIDESENSE</p>
                    <p class="logo-tagline">AI · PDF · Image Analyzer</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
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
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.07)
    st.session_state.logo_typed = True


def render_logo():
    """Static logo for main area."""
    st.markdown(f"""
        <div class="logo-bar">
            {logo_img_tag(56)}
            <div>
                <p class="logo-text">SLIDESENSE</p>
                <p class="logo-tagline">AI · PDF · Image Analyzer</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_answer_with_copy(answer: str) -> None:
    st.markdown(answer)
    safe_text = json.dumps(answer)
    # Use postMessage to escape the iframe sandbox and copy to clipboard
    components.html(
        f"""
        <script>
        function copyText() {{
            var text = {safe_text};
            // Try modern clipboard API first (works outside iframe via postMessage)
            try {{
                var el = document.createElement('textarea');
                el.value = text;
                el.setAttribute('readonly', '');
                el.style.position = 'absolute';
                el.style.left = '-9999px';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                var btn = document.getElementById('copybtn');
                btn.innerText = '✅ Copied!';
                btn.style.borderColor = '#48CAE4';
                btn.style.color = '#48CAE4';
                setTimeout(function() {{
                    btn.innerText = '📋 Copy';
                    btn.style.borderColor = '#6C63FF';
                    btn.style.color = '#6C63FF';
                }}, 2000);
            }} catch(e) {{
                // Fallback: open prompt with text selected
                window.prompt("Copy this text:", text);
            }}
        }}
        </script>
        <button id="copybtn" onclick="copyText();"
            style="margin-top:4px;padding:5px 14px;border-radius:6px;
                   border:1px solid #6C63FF;color:#6C63FF;
                   background:transparent;cursor:pointer;font-size:12px;
                   transition: all 0.2s ease;">
            📋 Copy
        </button>
        """,
        height=45,
    )


# -------------------- AUTH FUNCTIONS --------------------
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
    """Count existing chats to auto-number new ones."""
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode", "==", mode).stream()
    return sum(1 for _ in chats) + 1


def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    num = get_next_chat_number(user_id, mode)
    icon = "📘" if mode == "PDF" else "🖼"
    title = f"{icon} Chat {num}"
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode,
        "created_at": datetime.utcnow(),
        "title": title
    })
    return chat_id


def update_chat_title(user_id, chat_id, first_question):
    """Rename chat to first 35 chars of the first question asked."""
    short = first_question.strip()[:35]
    if len(first_question.strip()) > 35:
        short += "..."
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).update({"title": short})


def save_message(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role, "content": content, "timestamp": datetime.utcnow()
        })


def load_user_chats(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode", "==", mode) \
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
            st.markdown("<div style='text-align:center;padding-top:60px'>", unsafe_allow_html=True)
            if LOGO_B64:
                st.markdown(f'<div style="text-align:center">{logo_img_tag(120)}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
            st.info("**Guest mode:** Full PDF & Image AI — no account needed.\n\n❌ Chats are not saved and clear on refresh.")
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
        </div>
    """, unsafe_allow_html=True)

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


# ==================== MAIN CONTENT ====================
if not st.session_state.current_chat_id:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px 0;
        }
        .logo-container { width: 260px; text-align: center; }

        .slide-back { animation: slideIn 0.8s ease-out forwards; }
        .slide-front { animation: slideIn 0.8s 0.2s ease-out forwards; opacity: 0; }

        .node { animation: pulseNode 2s infinite ease-in-out; }
        .node-1 { animation-delay: 0.5s; }
        .node-2 { animation-delay: 0.8s; }
        .node-3 { animation-delay: 1.1s; }

        .arrow {
            stroke-dasharray: 100;
            stroke-dashoffset: 100;
            animation: drawArrow 1s 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .brand-name {
            font-size: 38px;
            font-weight: 800;
            color: #0c4a6e;
            margin-top: 8px;
            opacity: 0;
            animation: fadeInUp 0.8s 1.5s forwards;
        }
        .tagline {
            font-size: 14px;
            color: #64748b;
            letter-spacing: 1px;
            opacity: 0;
            animation: fadeIn 1s 2s forwards;
            margin-bottom: 16px;
        }
        .hint {
            font-size: 13px;
            color: #6C63FF;
            opacity: 0;
            animation: fadeIn 1s 2.4s forwards;
            background: rgba(108,99,255,0.08);
            border: 1px solid rgba(108,99,255,0.25);
            border-radius: 8px;
            padding: 8px 16px;
        }

        @keyframes slideIn {
            from { transform: translateX(-30px); opacity: 0; }
            to   { transform: translateX(0);     opacity: 1; }
        }
        @keyframes pulseNode {
            0%,100% { fill: #0ea5e9; r: 2; }
            50%      { fill: #38bdf8; r: 3.5; filter: drop-shadow(0 0 2px #38bdf8); }
        }
        @keyframes drawArrow { to { stroke-dashoffset: 0; } }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn { to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="logo-container">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <path class="slide-back" d="M30 20 H65 A5 5 0 0 1 70 25 V75 A5 5 0 0 1 65 80 H30 A5 5 0 0 1 25 75 V25 A5 5 0 0 1 30 20" fill="#0369a1" />
            <path class="slide-front" d="M35 25 H70 A5 5 0 0 1 75 30 V80 A5 5 0 0 1 70 85 H35 A5 5 0 0 1 30 80 V30 A5 5 0 0 1 35 25" fill="#0ea5e9" />
            <circle cx="55" cy="55" r="22" fill="white" stroke="#0c4a6e" stroke-width="3" />
            <circle class="node node-1" cx="48" cy="52" r="2" fill="#0ea5e9" />
            <circle class="node node-2" cx="55" cy="48" r="2" fill="#0ea5e9" />
            <circle class="node node-3" cx="53" cy="58" r="2" fill="#0ea5e9" />
            <line x1="48" y1="52" x2="55" y2="48" stroke="#cbd5e1" stroke-width="0.5" />
            <line x1="55" y1="48" x2="53" y2="58" stroke="#cbd5e1" stroke-width="0.5" />
            <path class="arrow" d="M45 65 L75 35 M75 35 L68 35 M75 35 L75 42" stroke="#0ea5e9" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" fill="none" />
        </svg>
        <div class="brand-name">SlideSense</div>
        <div class="tagline">PDF and Image Q&amp;A</div>
        <div class="hint">👈 Select ➕ New Chat from the sidebar to begin</div>
    </div>
</body>
</html>
        """, height=400)

else:
    if st.session_state.mode == "PDF":
        ac, tc = st.columns([1, 4])
        with ac:
            lp = load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
            if lp:
                st_lottie(lp, height=110, key="pdf_anim")
        with tc:
            st.markdown("## 📘 PDF Analyzer")
            st.caption("Upload a PDF and ask questions about its content.")
        st.divider()

        pdf = st.file_uploader("Upload PDF", type="pdf")
        if pdf and st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = "".join(p.extract_text() or "" for p in reader.pages)
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(text)
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_texts(chunks, emb)
                st.success("✅ PDF processed! Ask your questions below.")
    else:
        ac, tc = st.columns([1, 4])
        with ac:
            li = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
            if li:
                st_lottie(li, height=110, key="img_anim")
        with tc:
            st.markdown("## 🖼 Image Q&A")
            st.caption("Upload an image or use your live camera to ask questions.")
        st.divider()

        img_source = st.radio(
            "Image Source",
            ["📁 Upload Image", "📷 Live Camera"],
            horizontal=True,
            key="img_source"
        )

        img_file = None
        camera_file = None

        if img_source == "📁 Upload Image":
            img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if img_file:
                st.image(Image.open(img_file).convert("RGB"), use_container_width=True)
        else:
            st.info("📷 Point your camera and click **\'Take Photo\'** to capture.")
            camera_file = st.camera_input("Take a photo")
            if camera_file:
                st.image(Image.open(camera_file).convert("RGB"), use_container_width=True)

    # Messages
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
            # Rename chat to first question if it still has default "Chat N" title
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
                    # camera_input is always jpeg; uploaded files use their extension
                    if camera_file and not img_file:
                        mime = "image/jpeg"
                    else:
                        mime = "image/png" if img_file.name.lower().endswith(".png") else "image/jpeg"
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
