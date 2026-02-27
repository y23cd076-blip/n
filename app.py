import streamlit as st
from datetime import datetime
import uuid
import requests
from PyPDF2 import PdfReader
from PIL import Image
import base64

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


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SlideSense AI", layout="wide")


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
    # Guest-only in-memory chat history (not saved to Firestore)
    "guest_messages": [],
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- AUTH FUNCTIONS --------------------
def signup(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user
    except auth.EmailAlreadyExistsError:
        st.error("An account with this email already exists.")
        return None
    except Exception as e:
        st.error(f"Signup error: {e}")
        return None


def login(email, password):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            st.error("⚠️ FIREBASE_WEB_API_KEY is missing from Streamlit secrets.")
            return None

        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload)
        data = response.json()

        if "error" in data:
            msg = data["error"]["message"]
            if msg in ("EMAIL_NOT_FOUND", "INVALID_LOGIN_CREDENTIALS", "INVALID_PASSWORD"):
                st.error("❌ Invalid email or password.")
            else:
                st.error(f"Login failed: {msg}")
            return None

        user = auth.get_user_by_email(email)
        return user

    except Exception as e:
        st.error(f"Login error: {e}")
        return None


def reset_password(email):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            st.error("⚠️ FIREBASE_WEB_API_KEY is missing from Streamlit secrets.")
            return False

        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"
        payload = {"requestType": "PASSWORD_RESET", "email": email}
        response = requests.post(url, json=payload)
        data = response.json()

        if "error" in data:
            st.error(f"Reset failed: {data['error']['message']}")
            return False
        return True

    except Exception as e:
        st.error(f"Reset error: {e}")
        return False


# -------------------- FIRESTORE --------------------
def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode,
        "created_at": datetime.utcnow(),
        "title": "New Chat"
    })
    return chat_id


def save_message(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })


def load_user_chats(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode", "==", mode) \
        .order_by("created_at", direction=firestore.Query.DESCENDING) \
        .stream()
    return [(doc.id, doc.to_dict()["title"]) for doc in chats]


def load_messages(user_id, chat_id):
    messages = db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages") \
        .order_by("timestamp").stream()
    return [(doc.to_dict()["role"], doc.to_dict()["content"]) for doc in messages]


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )


# ==================== AUTH SCREEN ====================
if not st.session_state.authenticated and not st.session_state.is_guest:

    # Hero header
    st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size:3rem;'>🧠 SlideSense AI</h1>
            <p style='font-size:1.2rem; color:gray;'>AI-Powered PDF & Image Analyzer</p>
        </div>
    """, unsafe_allow_html=True)

    col_space1, col_main, col_space2 = st.columns([1, 2, 1])

    with col_main:
        tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "📝 Sign Up", "👤 Guest"])

        # ---- LOGIN TAB ----
        with tab_login:
            st.markdown("### Welcome back!")
            login_email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")

            col_btn, col_forgot = st.columns([1, 1])
            with col_btn:
                if st.button("Login", use_container_width=True, type="primary"):
                    if not login_email or not login_password:
                        st.warning("Please fill in all fields.")
                    else:
                        with st.spinner("Logging in..."):
                            user = login(login_email, login_password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.is_guest = False
                            st.session_state.user_id = user.uid
                            st.session_state.email = login_email
                            st.success("✅ Logged in!")
                            st.rerun()

            with col_forgot:
                if st.button("Forgot Password?", use_container_width=True):
                    if not login_email:
                        st.warning("Enter your email above first.")
                    else:
                        with st.spinner("Sending reset email..."):
                            sent = reset_password(login_email)
                        if sent:
                            st.success("📧 Password reset email sent! Check your inbox.")

        # ---- SIGN UP TAB ----
        with tab_signup:
            st.markdown("### Create an account")
            new_email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            new_password = st.text_input("Password", type="password", key="signup_password", placeholder="Min 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Re-enter password")

            if st.button("Create Account", use_container_width=True, type="primary"):
                if not new_email or not new_password or not confirm_password:
                    st.warning("Please fill in all fields.")
                elif len(new_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        user = signup(new_email, new_password)
                    if user:
                        st.success("✅ Account created! Please go to the Login tab.")

        # ---- GUEST TAB ----
        with tab_guest:
            st.markdown("### Continue as Guest")
            st.info("""
**Guest mode lets you:**
- 📘 Analyze PDFs
- 🖼️ Ask questions about images
- 💬 Chat with AI instantly

**Limitations:**
- ❌ Chat history is not saved
- ❌ No access to previous sessions
- ❌ History clears on page refresh
""")
            st.markdown("---")
            if st.button("👤 Continue as Guest", use_container_width=True, type="primary"):
                st.session_state.is_guest = True
                st.session_state.authenticated = False
                st.session_state.user_id = "guest"
                st.session_state.email = "Guest"
                st.session_state.guest_messages = []
                st.rerun()

    st.stop()


# ==================== SIDEBAR ====================
if st.session_state.is_guest:
    st.sidebar.warning("👤 Guest Mode")
    st.sidebar.caption("Your chats won't be saved.")
else:
    st.sidebar.success(f"👤 {st.session_state.email}")

if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

# ---- Guest: no persistent chat history ----
if st.session_state.is_guest:
    st.sidebar.markdown("## 💬 Session Chat")
    st.sidebar.info("Guest chats are temporary and not saved.")

    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.guest_messages = []
        st.session_state.vector_db = None
        st.rerun()

    # Auto-create a single guest chat session
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = "guest_session"

# ---- Authenticated: full chat management ----
else:
    st.sidebar.markdown("## 💬 Your Chats")
    user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)

    for chat_id, title in user_chats:
        col1, col2 = st.sidebar.columns([4, 1])
        icon = "📘" if st.session_state.mode == "PDF" else "🖼"

        if col1.button(f"{icon} {title}", key=f"open_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.session_state.vector_db = None
            st.rerun()

        if col2.button("🗑", key=f"delete_{chat_id}"):
            db.collection("users").document(st.session_state.user_id) \
              .collection("chats").document(chat_id).delete()
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
            st.rerun()

    if st.sidebar.button("➕ New Chat"):
        new_chat_id = create_new_chat(st.session_state.user_id, st.session_state.mode)
        st.session_state.current_chat_id = new_chat_id
        st.session_state.vector_db = None
        st.rerun()


# ==================== MAIN CONTENT ====================
if not st.session_state.current_chat_id:
    st.markdown("## 👋 Welcome to SlideSense AI")
    st.markdown("### 🚀 AI Powered PDF & Image Analyzer")
    st.info("Select '➕ New Chat' from the sidebar to begin.")

else:

    if st.session_state.mode == "PDF":
        st.markdown("## 📘 PDF Analyzer")
        pdf = st.file_uploader("Upload PDF", type="pdf")

        if pdf and st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
                st.success("✅ PDF processed! Ask your questions below.")

    else:
        st.markdown("## 🖼 Image Question Answering")
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, use_container_width=True)

    # -------------------- LOAD & DISPLAY MESSAGES --------------------
    if st.session_state.is_guest:
        messages = st.session_state.guest_messages
    else:
        messages = load_messages(st.session_state.user_id, st.session_state.current_chat_id)

    for role, content in messages:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # -------------------- CHAT INPUT --------------------
    question = st.chat_input("Ask something...")

    if question:

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Save user message
        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("user", question))
        else:
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "user", question)

        # -------- PDF ANSWER --------
        if st.session_state.mode == "PDF":
            if st.session_state.vector_db is None:
                answer = "⚠️ Please upload a PDF first."
            else:
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_db.similarity_search(question, k=6)
                    llm = load_llm()
                    prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{input}

If the answer is not found in the context, say: Information not found in document.
""")
                    chain = create_stuff_documents_chain(llm, prompt)
                    result = chain.invoke({"context": docs, "input": question})
                    answer = result if isinstance(result, str) else result.get("output_text", str(result))

        # -------- IMAGE ANSWER --------
        else:
            if not img_file:
                answer = "⚠️ Please upload an image first."
            else:
                with st.spinner("🖼 Analyzing image..."):
                    llm = load_llm()
                    image_bytes = img_file.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                    mime_type = "image/png" if img_file.name.lower().endswith(".png") else "image/jpeg"

                    response = llm.invoke([
                        HumanMessage(content=[
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}},
                        ])
                    ])
                    answer = response.content

        # Save & display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("assistant", answer))
        else:
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "assistant", answer)

        st.rerun()
