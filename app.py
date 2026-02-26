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
        # ✅ FIXED (no silent crash anymore)
        google_api_key=(
            st.secrets["GOOGLE_API_KEY"]
            if "GOOGLE_API_KEY" in st.secrets
            else os.getenv("GOOGLE_API_KEY")
        ),
    )


def invoke_llm(prompt_text: str):
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
