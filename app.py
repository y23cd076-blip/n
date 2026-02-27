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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense AI", layout="wide")

# ✅ SAFE FIREBASE INIT
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()


# -------------------- SESSION --------------------
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
def signup(email, password):
    try:
        return auth.create_user(email=email, password=password)
    except Exception as e:
        st.error(str(e))
        return None


def login(email):
    try:
        return auth.get_user_by_email(email)
    except:
        return None


# -------------------- FIRESTORE --------------------
def create_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode,
        "created_at": datetime.utcnow(),
        "title": "New Chat"
    })
    return chat_id


def save_msg(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })


def load_msgs(user_id, chat_id):
    docs = db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages") \
        .order_by("timestamp").stream()

    return [(d.to_dict()["role"], d.to_dict()["content"]) for d in docs]


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=st.secrets.get("GOOGLE_API_KEY")
    )


# -------------------- LOGIN --------------------
if not st.session_state.authenticated:
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login(email)
        if user:
            st.session_state.authenticated = True
            st.session_state.user_id = user.uid
            st.session_state.email = email
            st.rerun()
        else:
            st.error("User not found")

    if st.button("Signup"):
        user = signup(email, password)
        if user:
            st.success("Account created")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.write(st.session_state.email)

mode = st.sidebar.radio("Mode", ["PDF", "IMAGE"])
st.session_state.mode = mode

if st.sidebar.button("New Chat"):
    st.session_state.current_chat_id = create_chat(
        st.session_state.user_id, mode
    )
    st.session_state.vector_db = None
    st.rerun()


# -------------------- MAIN --------------------
if not st.session_state.current_chat_id:
    st.write("Create a chat to start")
    st.stop()


# -------------------- PDF MODE --------------------
if mode == "PDF":

    pdf = st.file_uploader("Upload PDF")

    if pdf and st.session_state.vector_db is None:
        reader = PdfReader(pdf)
        text = ""

        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

# -------------------- IMAGE MODE --------------------
else:
    img_file = st.file_uploader("Upload Image")

    if img_file:
        st.image(img_file)


# -------------------- CHAT --------------------
msgs = load_msgs(
    st.session_state.user_id,
    st.session_state.current_chat_id
)

for r, c in msgs:
    st.write(f"{r}: {c}")

q = st.chat_input("Ask")

if q:

    save_msg(st.session_state.user_id,
             st.session_state.current_chat_id,
             "user", q)

    llm = load_llm()

    # PDF
    if mode == "PDF":
        if st.session_state.vector_db is None:
            ans = "Upload PDF first"
        else:
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}
""")

            chain = create_stuff_documents_chain(llm, prompt)

            res = chain.invoke({
                "context": docs,
                "question": q
            })

            ans = res.get("output_text", "")

    # IMAGE
    else:
        if not img_file:
            ans = "Upload image first"
        else:
            img_bytes = img_file.getvalue()
            encoded = base64.b64encode(img_bytes).decode()

            res = llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": q},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ])
            ])

            ans = res.content

    save_msg(st.session_state.user_id,
             st.session_state.current_chat_id,
             "assistant", ans)

    st.rerun()
