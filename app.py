import streamlit as st
from datetime import datetime
import uuid
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
    firebase_config = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
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
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "vector_db": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- AUTH --------------------
# NOTE: Firebase Admin SDK does not support password verification directly.
# Password check is done via a REST call to Firebase Auth REST API.
import requests

def signup(email, password):
    try:
        return auth.create_user(email=email, password=password)
    except Exception as e:
        st.error(f"Signup error: {e}")
        return None


def login(email, password):
    """
    Firebase Admin SDK cannot verify passwords.
    We use the Firebase Auth REST API for sign-in.
    Requires FIREBASE_WEB_API_KEY in st.secrets.
    """
    try:
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload)
        data = response.json()

        if "error" in data:
            st.error(f"Login failed: {data['error']['message']}")
            return None

        # Get the Firebase user object via Admin SDK
        user = auth.get_user_by_email(email)
        return user
    except Exception as e:
        st.error(f"Login error: {e}")
        return None


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


# -------------------- LOGIN UI --------------------
if not st.session_state.authenticated:

    st.title("🔐 SlideSense Login")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            user = login(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user.uid
                st.session_state.email = email
                st.rerun()

    with tab2:
        new_email = st.text_input("New Email", key="signup_email")
        new_password = st.text_input("New Password", type="password", key="signup_password")

        if st.button("Signup"):
            user = signup(new_email, new_password)
            if user:
                st.success("Account created! Please log in.")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.success(f"👤 {st.session_state.email}")

if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

st.sidebar.markdown("## 💬 Your Chats")

user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)

for chat_id, title in user_chats:
    col1, col2 = st.sidebar.columns([4, 1])
    icon = "📘" if st.session_state.mode == "PDF" else "🖼"

    if col1.button(f"{icon} {title}", key=f"open_{chat_id}"):
        st.session_state.current_chat_id = chat_id
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


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]  # ✅ Direct access, not .get()
    )


# -------------------- MAIN CONTENT --------------------
if not st.session_state.current_chat_id:

    st.markdown("## 👋 Welcome to SlideSense AI")
    st.markdown("### 🚀 AI Powered PDF & Image Analyzer")
    st.info("Select 'New Chat' from the sidebar to begin.")

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

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )

                chunks = splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
                st.success("PDF processed! Ask your questions below.")

    else:

        st.markdown("## 🖼 Image Question Answering")
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, use_container_width=True)  # ✅ Fixed deprecated param

    # -------------------- LOAD MESSAGES --------------------
    messages = load_messages(
        st.session_state.user_id,
        st.session_state.current_chat_id
    )

    for role, content in messages:
        if role == "user":
            st.markdown(f"🧑 {content}")
        else:
            st.markdown(f"🤖 {content}")

    question = st.chat_input("Ask something...")

    if question:

        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "user",
            question
        )

        # -------- PDF ANSWER --------
        if st.session_state.mode == "PDF":
            if st.session_state.vector_db is None:
                answer = "Please upload a PDF first."
            else:
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_db.similarity_search(question, k=6)
                    llm = load_llm()

                    prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{input}

If not found say:
Information not found in document.
""")

                    chain = create_stuff_documents_chain(llm, prompt)

                    result = chain.invoke({
                        "context": docs,
                        "input": question  # ✅ create_stuff_documents_chain uses "input" key
                    })

                    # result is a string in newer LangChain versions
                    answer = result if isinstance(result, str) else result.get("output_text", str(result))

        # -------- IMAGE ANSWER --------
        else:
            if not img_file:
                answer = "Please upload an image first."
            else:
                with st.spinner("🖼 Analyzing image..."):

                    llm = load_llm()
                    image_bytes = img_file.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                    # Detect MIME type from filename
                    filename = img_file.name.lower()
                    mime_type = "image/png" if filename.endswith(".png") else "image/jpeg"

                    response = llm.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": question},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{encoded_image}"
                                        },
                                    },
                                ]
                            )
                        ]
                    )

                    answer = response.content

        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "assistant",
            answer
        )

        st.rerun()
