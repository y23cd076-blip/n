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
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def type_text(text, speed=0.01):
    st.markdown(f"### {text}")


# -------------------- SAFE API KEY --------------------
def get_google_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY")


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    key = get_google_key()
    if not key:
        st.error("❌ GOOGLE_API_KEY missing in Streamlit secrets")
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=2048,
        google_api_key=key,
    )


def invoke_llm(prompt_text: str):
    llm = load_llm()
    try:
        msg = llm.invoke([HumanMessage(content=prompt_text)])
        return msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None


# -------------------- SAFE BLIP (LAZY LOAD) --------------------
@st.cache_resource
def load_blip():
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(device)
        return processor, model, device
    except Exception as e:
        st.error("⚠️ Image model failed to load (transformers issue)")
        return None, None, None


def answer_image_question(image, question):
    processor, model, device = load_blip()
    if processor is None:
        return "Image model not available."

    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=30)

    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    return invoke_llm(f"Question: {question}\nAnswer: {short_answer}")


# -------------------- BASIC UI (to prevent blank screen) --------------------
st.title("📘 SlideSense")

st.success("✅ App Loaded Successfully")


mode = st.radio("Choose Mode", ["PDF", "Image"])

# -------------------- PDF --------------------
if mode == "PDF":
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        question = st.text_input("Ask question")

        if question:
            answer = invoke_llm(text[:3000] + "\n\n" + question)
            if answer:
                st.write(answer)


# -------------------- IMAGE --------------------
if mode == "Image":
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img)

        question = st.text_input("Ask about image")

        if question:
            answer = answer_image_question(img, question)
            st.write(answer)
