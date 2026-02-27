# ================= SlideSense AI - FIXED SUPABASE VERSION =================

import base64, hashlib, os, uuid
from datetime import datetime
import requests
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SlideSense AI", layout="wide")

def _secret(name, default=""):
    try:
        v = st.secrets.get(name)
        if v: return str(v)
    except: pass
    return str(os.getenv(name, default) or default)

SUPABASE_URL = _secret("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = _secret("SUPABASE_ANON_KEY") or _secret("SUPABASE_KEY")
GOOGLE_API_KEY = _secret("GOOGLE_API_KEY")
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-1.5-flash")

USER_TABLE = "user_profiles"
CHAT_TABLE = "chat_sessions"
MSG_TABLE = "chat_messages"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in secrets")
    st.stop()

# 🔥 FIX: correct REST base
REST_BASE = f"{SUPABASE_URL}/rest/v1"

def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def sb_rest(method, table, *, params=None, json=None):
    url = f"{REST_BASE}/{table}"
    r = requests.request(method, url, headers=sb_headers(), params=params, json=json, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase error {r.status_code}: {r.text}")
    return r.json() if r.text else None

def hash_pw(pw): 
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- SESSION ----------------
for k,v in {
    "authenticated":False,
    "user_id":None,
    "username":None,
    "mode":"PDF",
    "current_chat_id":None,
    "vector_dbs":{},
    "pdf_fingerprints":{}
}.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ---------------- AUTH ----------------
def signup(username,password):
    r = sb_rest("GET", USER_TABLE, params={"username":f"eq.{username}","select":"id"})
    if r: return "User exists"
    uid=str(uuid.uuid4())
    sb_rest("POST", USER_TABLE, json={
        "id":uid,"username":username,
        "password_hash":hash_pw(password),
        "created_at":datetime.utcnow().isoformat()
    })
    return None

def login(username,password):
    r = sb_rest("GET", USER_TABLE, params={"username":f"eq.{username}","select":"*"})
    if not r: return "Invalid login"
    if r[0]["password_hash"]!=hash_pw(password): return "Invalid login"
    st.session_state.authenticated=True
    st.session_state.user_id=r[0]["id"]
    st.session_state.username=r[0]["username"]
    return None

def login_as_guest():
    uid=str(uuid.uuid4())
    uname=f"guest_{uid[:6]}"
    try:
        sb_rest("POST", USER_TABLE, json={
            "id":uid,"username":uname,"password_hash":None,
            "created_at":datetime.utcnow().isoformat()
        })
    except: pass
    st.session_state.authenticated=True
    st.session_state.user_id=uid
    st.session_state.username=uname

# ---------------- CHAT DB ----------------
def create_new_chat(uid,mode):
    cid=str(uuid.uuid4())
    sb_rest("POST", CHAT_TABLE, json={
        "id":cid,"user_id":uid,"mode":mode,
        "title":"New Chat","created_at":datetime.utcnow().isoformat()
    })
    return cid

def save_message(cid,role,content):
    sb_rest("POST", MSG_TABLE, json={
        "id":str(uuid.uuid4()),
        "chat_id":cid,
        "role":role,
        "content":content,
        "created_at":datetime.utcnow().isoformat()
    })

def load_user_chats(uid,mode):
    return sb_rest("GET", CHAT_TABLE, params={
        "user_id":f"eq.{uid}",
        "mode":f"eq.{mode}",
        "select":"id,title,created_at",
        "order":"created_at.desc"
    }) or []

def load_messages(cid):
    return sb_rest("GET", MSG_TABLE, params={
        "chat_id":f"eq.{cid}",
        "select":"role,content,created_at",
        "order":"created_at.asc"
    }) or []

# ---------------- UI ----------------
if not st.session_state.authenticated:
    st.title("🔐 SlideSense Login")
    t1,t2,t3=st.tabs(["Login","Signup","Guest"])

    with t1:
        u=st.text_input("Username")
        p=st.text_input("Password",type="password")
        if st.button("Login"):
            err=login(u,p)
            if err: st.error(err)
            else: st.rerun()

    with t2:
        u=st.text_input("New Username")
        p=st.text_input("New Password",type="password")
        if st.button("Signup"):
            err=signup(u,p)
            if err: st.error(err)
            else: st.success("Account created")

    with t3:
        if st.button("Continue as Guest"):
            login_as_guest()
            st.rerun()
    st.stop()

st.sidebar.success(f"👤 {st.session_state.username}")
mode = st.sidebar.radio("Mode",["📘 PDF","🖼 IMAGE"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

if st.sidebar.button("➕ New Chat"):
    cid=create_new_chat(st.session_state.user_id, st.session_state.mode)
    st.session_state.current_chat_id=cid
    st.rerun()

chats=load_user_chats(st.session_state.user_id, st.session_state.mode)
for c in chats:
    if st.sidebar.button(c["title"],key=c["id"]):
        st.session_state.current_chat_id=c["id"]
        st.rerun()

if not st.session_state.current_chat_id:
    st.info("Create a new chat to start")
    st.stop()

cid=st.session_state.current_chat_id

msgs=load_messages(cid)
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question=st.chat_input("Ask something...")
if question:
    save_message(cid,"user",question)

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL,temperature=0.3)

    if st.session_state.mode=="PDF":
        answer="PDF mode active"
    else:
        answer="IMAGE mode active"

    save_message(cid,"assistant",answer)
    st.rerun()
