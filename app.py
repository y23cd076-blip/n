import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib, uuid
from PyPDF2 import PdfReader
from PIL import Image
import torch

try:
    from supabase import create_client
except ImportError:
    create_client = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")
USERS_FILE = "users.json"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_USERS_TABLE = os.getenv("SUPABASE_USERS_TABLE", "users")


# -------------------- HELPERS --------------------
@st.cache_resource
def get_supabase_client():
    if create_client is None:
        return None
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception:
        return None


def load_users():
    supabase = get_supabase_client()
    if supabase is not None:
        try:
            resp = supabase.table(SUPABASE_USERS_TABLE).select(
                "username,password_hash"
            ).execute()
            data = getattr(resp, "data", None)
            if data:
                return {row["username"]: row["password_hash"] for row in data}
            return {}
        except Exception as e:
            st.warning(f"Falling back to local user store: {e}")

    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    supabase = get_supabase_client()
    if supabase is not None:
        rows = [{"username": u, "password_hash": pw} for u, pw in users.items()]
        try:
            if rows:
                supabase.table(SUPABASE_USERS_TABLE).upsert(rows).execute()
        except Exception as e:
            st.warning(f"Could not save users to Supabase: {e}")

    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def type_text(text, speed=0.02):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=2048,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def invoke_llm(prompt_text: str):
    """Invoke Gemini with a text prompt; returns response content or raises a clear error."""
    llm = load_llm()
    try:
        msg = llm.invoke([HumanMessage(content=prompt_text)])
        return msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        if type(e).__name__ == "ChatGoogleGenerativeAIError":
            raise ValueError(
                "Gemini API error. Set GOOGLE_API_KEY in Streamlit Cloud secrets (or env) and ensure the key is valid."
            ) from e
        err = str(e).lower()
        if "api_key" in err or "invalid" in err or "403" in err:
            raise ValueError(
                "Google API error: Set GOOGLE_API_KEY in your environment (e.g. Streamlit Cloud secrets)."
            ) from e
        raise

@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    return processor, model, device


# -------------------- SESSION DEFAULTS --------------------
def _default_chats():
    return [{"id": "0", "title": "New chat", "messages": []}]

defaults = {
    "authenticated": False,
    "guest": False,
    "welcome_done": False,
    "users": load_users(),
    "vector_db": None,
    "current_pdf_id": None,
    "chats": _default_chats(),
    "current_chat_id": "0",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_current_chat():
    """Return the chat dict for current_chat_id; ensure it exists."""
    cid = st.session_state.current_chat_id
    for c in st.session_state.chats:
        if c["id"] == cid:
            return c
    if st.session_state.chats:
        st.session_state.current_chat_id = st.session_state.chats[0]["id"]
        return st.session_state.chats[0]
    st.session_state.chats = _default_chats()
    st.session_state.current_chat_id = "0"
    return st.session_state.chats[0]

def add_message_to_current_chat(question: str, answer: str):
    chat = get_current_chat()
    chat["messages"].append((question, answer))
    if chat["title"] == "New chat" and question:
        chat["title"] = question[:40] + ("…" if len(question) > 40 else "")


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
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == hash_password(p):
                    st.session_state.authenticated = True
                    st.session_state.guest = False
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = hash_password(np)
                    save_users(st.session_state.users)
                    st.success("Account created")

        with tab3:
            st.info("Use SlideSense without an account. Your session won't be saved.")
            if st.button("Continue as Guest"):
                st.session_state.authenticated = True
                st.session_state.guest = True
                st.rerun()


# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=30,
        num_beams=5,
        early_stopping=True
    )

    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    prompt = f"""
Question: {question}
Vision Answer: {short_answer}

Convert into one clear and complete sentence.
"""
    return invoke_llm(prompt)


# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()


# -------------------- WELCOME PAGE (after sign in) --------------------
if not st.session_state.get("welcome_done"):
    st.sidebar.success("Logged in ✅" if not st.session_state.get("guest") else "Guest 👤")
    if st.sidebar.button("Logout"):
        st.cache_resource.clear()
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 📘 Welcome to SlideSense")
        st.markdown("---")
        st.markdown(
            """
            **SlideSense** is an AI-powered learning platform that helps you understand documents and images.

            - **📘 PDF Analyzer** — Upload a PDF (notes, textbooks, slides). Ask questions and get answers grounded in the document.
            - **🖼 Image Q&A** — Upload an image and ask anything about it. The AI describes and answers using vision.

            Click **New Chat** below to start. You can then choose PDF Analyzer or Image Q&A and switch between chats from the sidebar.
            """
        )
        st.markdown("---")
        if st.button("🆕 New Chat", type="primary", use_container_width=True):
            st.session_state.welcome_done = True
            st.rerun()
    st.stop()


# -------------------- SIDEBAR (after New Chat from welcome) --------------------
if st.session_state.get("guest"):
    st.sidebar.success("Guest 👤")
else:
    st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

st.sidebar.divider()
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    new_id = str(uuid.uuid4())[:8]
    st.session_state.chats.append({"id": new_id, "title": "New chat", "messages": []})
    st.session_state.current_chat_id = new_id
    st.session_state.vector_db = None
    st.session_state.current_pdf_id = None
    st.rerun()

st.sidebar.markdown("**💬 Chats**")
chats = st.session_state.chats
current_id = st.session_state.current_chat_id
for c in chats:
    label = c["title"] or "New chat"
    if st.sidebar.button(
        label[:35] + ("…" if len(label) > 35 else ""),
        key="chat_" + c["id"],
        use_container_width=True,
        type="primary" if c["id"] == current_id else "secondary",
    ):
        st.session_state.current_chat_id = c["id"]
        st.rerun()

if st.sidebar.button("🗑️ Delete chat", use_container_width=True):
    chats = st.session_state.chats
    current_id = st.session_state.current_chat_id
    if len(chats) <= 1:
        get_current_chat()["messages"] = []
        get_current_chat()["title"] = "New chat"
    else:
        idx = next((i for i, c in enumerate(chats) if c["id"] == current_id), 0)
        st.session_state.chats.pop(idx)
        st.session_state.current_chat_id = st.session_state.chats[max(0, idx - 1)]["id"]
    st.rerun()

if st.sidebar.button("🗑 Clear history", use_container_width=True):
    get_current_chat()["messages"] = []
    get_current_chat()["title"] = "New chat"
    st.rerun()

st.sidebar.divider()
mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])


# -------------------- HERO --------------------
col1, col2 = st.columns([1, 2])

with col1:
    st_lottie(
        load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
        height=250
    )

with col2:
    type_text("📘 SlideSense AI Platform")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            get_current_chat()["messages"] = []

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""

                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )

                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = InMemoryVectorStore.from_texts(chunks, embeddings)

        question = st.chat_input("Ask a question about the PDF")

        if question:
            with st.spinner("Thinking..."):
                docs = st.session_state.vector_db.similarity_search(question, k=8)
                llm = load_llm()

                context = "\n\n".join(doc.page_content for doc in docs)
                prompt_text = f"""You are analyzing a structured academic document.

Context:
{context}

Question:
{question}

Instructions:
- Provide complete information from context.
- If question refers to a section (like Unit 3),
  explain the full section clearly.
- Do not cut off important points.
- If not found say:
  "Information not found in the document."
"""
                try:
                    answer = invoke_llm(prompt_text)
                except ValueError as e:
                    st.error(str(e))
                    answer = None
                if answer:
                    add_message_to_current_chat(question, answer)

        # Display Chat
        st.markdown("## 💬 Conversation")
        messages = get_current_chat()["messages"]
        for q, a in reversed(messages):
            st.markdown(f"🧑 **You:** {q}")
            st.markdown(f"🤖 **AI:** {a}")
            st.divider()


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.chat_input("Ask a question about the image")

        if question:
            with st.spinner("Analyzing image..."):
                try:
                    answer = answer_image_question(img, question)
                    st.success(answer)
                    add_message_to_current_chat(question, answer)
                except ValueError as e:
                    st.error(str(e))
