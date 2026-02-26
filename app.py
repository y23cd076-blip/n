import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch

try:
    from supabase import create_client
except ImportError:
    create_client = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
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
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=2048
    )

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
    "authenticated": False,
    "guest": False,
    "users": load_users(),
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None
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

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}

Convert into one clear and complete sentence.
"""
    return llm.invoke(prompt).content


# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()


# -------------------- SIDEBAR --------------------
if st.session_state.get("guest"):
    st.sidebar.success("Guest 👤")
else:
    st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

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
            st.session_state.chat_history = []

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

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        question = st.chat_input("Ask a question about the PDF")

        if question:
            with st.spinner("Thinking..."):
                docs = st.session_state.vector_db.similarity_search(question, k=8)
                llm = load_llm()

                prompt = ChatPromptTemplate.from_template("""
You are analyzing a structured academic document.

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
""")

                chain = create_stuff_documents_chain(llm, prompt)

                result = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = result.get("output_text", "") \
                    if isinstance(result, dict) else result

                st.session_state.chat_history.append((question, answer))

        # Display Chat
        st.markdown("## 💬 Conversation")

        for q, a in reversed(st.session_state.chat_history):
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
                answer = answer_image_question(img, question)
                st.success(answer)