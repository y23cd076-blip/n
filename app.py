# ================================
# SlideSense AI - Full Stable App
# ================================

import os
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from PIL import Image
from PyPDF2 import PdfReader
import numpy as np

# -------- Optional Vision --------
try:
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor
    _BLIP_AVAILABLE = True
except Exception:
    _BLIP_AVAILABLE = False

# -------- LangChain / Gemini -----
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

logger = logging.getLogger("slidesense")
logging.basicConfig(level=logging.INFO)

# ---- Streamlit Secrets ----
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

USER_TABLE = "users"
CHATS_TABLE = "chats"
MESSAGES_TABLE = "messages"
CHAT_META_TABLE = "chat_metadata"

MAX_CONTEXT_TOKENS = 2500
SUMMARIZE_AFTER_MESSAGES = 24


# =========================================================
# SUPABASE CLIENT
# =========================================================

def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def sb_request(method: str, path: str, params=None, json_body=None):
    url = f"{SUPABASE_URL}{path}"
    resp = requests.request(
        method,
        url,
        headers=_supabase_headers(),
        params=params,
        json=json_body,
        timeout=20,
    )
    return resp


def sb_get(path: str, params=None):
    r = sb_request("GET", path, params=params)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() or []


def sb_post(path: str, json_body):
    r = sb_request("POST", path, json_body=json_body)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() or []


def sb_patch(path: str, json_body):
    r = sb_request("PATCH", path, json_body=json_body)
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() or []


# =========================================================
# AUTH
# =========================================================

import hashlib

def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def get_user():
    return st.session_state.get("user")

def set_user(u):
    st.session_state["user"] = u

def clear_session():
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)


def sign_up(username, password):
    rows = sb_get(f"/rest/v1/{USER_TABLE}", {
        "select": "id",
        "username": f"eq.{username}"
    })
    if rows:
        return "User already exists"

    sb_post(f"/rest/v1/{USER_TABLE}", {
        "username": username,
        "password_hash": hash_password(password)
    })
    return None


def sign_in(username, password):
    rows = sb_get(f"/rest/v1/{USER_TABLE}", {
        "select": "id,username,password_hash",
        "username": f"eq.{username}"
    })
    if not rows:
        return "Invalid credentials"

    if rows[0]["password_hash"] != hash_password(password):
        return "Invalid credentials"

    set_user({"id": rows[0]["id"], "username": rows[0]["username"]})
    return None


def sign_out():
    clear_session()


# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

@st.cache_resource
def load_blip():
    if not _BLIP_AVAILABLE:
        raise RuntimeError("BLIP not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    m = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return p, m, device


# =========================================================
# CHAT SYSTEM (SAFE MODE - NO JOINS)
# =========================================================

def auto_title(q: str):
    q = q.replace("\n", " ").strip()
    return (q[:40] + "...") if len(q) > 40 else q


def estimate_tokens(t: str):
    return max(1, len(t)//4)


def create_chat(user_id, mode, first_q):
    now = datetime.now(timezone.utc).isoformat()
    chat = sb_post(f"/rest/v1/{CHATS_TABLE}", {
        "user_id": user_id,
        "title": auto_title(first_q),
        "mode": mode,
        "is_pinned": False,
        "created_at": now,
        "updated_at": now
    })[0]

    sb_post(f"/rest/v1/{CHAT_META_TABLE}", {
        "chat_id": chat["id"],
        "summary": "",
        "token_count": 0,
        "last_message_at": now,
        "last_message_preview": first_q[:120]
    })
    return chat


def append_message(chat_id, role, content):
    sb_post(f"/rest/v1/{MESSAGES_TABLE}", {
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "token_estimate": estimate_tokens(content)
    })


def fetch_chat_messages(chat_id):
    return sb_get(f"/rest/v1/{MESSAGES_TABLE}", {
        "select": "*",
        "chat_id": f"eq.{chat_id}",
        "order": "created_at.asc"
    })


# 🔥 FIXED VERSION (NO JOIN)
def fetch_user_chats(user_id: str):
    chats = sb_get(
        f"/rest/v1/{CHATS_TABLE}",
        {
            "select": "*",
            "user_id": f"eq.{user_id}",
            "order": "is_pinned.desc,updated_at.desc",
        },
    )

    chats = [c for c in chats if c.get("deleted_at") is None]

    if not chats:
        return []

    chat_ids = [c["id"] for c in chats]

    meta_rows = sb_get(
        f"/rest/v1/{CHAT_META_TABLE}",
        {
            "select": "*",
            "chat_id": f"in.({','.join(chat_ids)})",
        },
    )

    meta_map = {m["chat_id"]: m for m in meta_rows}

    for c in chats:
        c["chat_metadata"] = meta_map.get(c["id"], {})

    return chats


def update_metadata(chat_id, answer, tokens):
    now = datetime.now(timezone.utc).isoformat()

    sb_patch(f"/rest/v1/{CHATS_TABLE}?id=eq.{chat_id}", {
        "updated_at": now
    })

    rows = sb_get(f"/rest/v1/{CHAT_META_TABLE}", {
        "select": "*",
        "chat_id": f"eq.{chat_id}"
    })

    current_tokens = rows[0].get("token_count", 0) if rows else 0

    sb_patch(f"/rest/v1/{CHAT_META_TABLE}?chat_id=eq.{chat_id}", {
        "last_message_at": now,
        "last_message_preview": answer[:150],
        "token_count": current_tokens + tokens
    })


# =========================================================
# PDF RAG
# =========================================================

class SimpleVectorIndex:
    def __init__(self, texts, vectors):
        self.texts = texts
        self.vectors = vectors.astype("float32")
        self.vectors /= (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)

    def similarity_search(self, q, k=5):
        emb = load_embeddings().embed_query(q)
        qv = np.array(emb, dtype="float32")
        qv /= (np.linalg.norm(qv)+1e-9)
        sims = np.dot(self.vectors, qv)
        idxs = np.argsort(-sims)[:k]

        class Doc:
            def __init__(self, c): self.page_content = c

        return [Doc(self.texts[i]) for i in idxs]


def build_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_text(text)
    embs = [load_embeddings().embed_query(c) for c in chunks]
    return SimpleVectorIndex(chunks, np.array(embs))


def answer_pdf(index, question):
    docs = index.similarity_search(question, k=5)
    llm = load_llm()

    prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from context
- If not found say: Information not found in the document.
""")

    chain = create_stuff_documents_chain(llm, prompt)
    res = chain.invoke({"context": docs, "question": question})

    ans = res.get("output_text","") if isinstance(res,dict) else str(res)

    return ans, docs


# =========================================================
# IMAGE QA
# =========================================================

def answer_image(img, q):
    p, m, d = load_blip()
    inputs = p(img, q, return_tensors="pt").to(d)
    out = m.generate(**inputs, max_length=10)
    short = p.decode(out[0], skip_special_tokens=True)

    llm = load_llm()
    return llm.invoke(f"Rewrite clearly: {short}").content


# =========================================================
# UI
# =========================================================

def inject_css():
    st.markdown("""
    <style>
    body {background:#020617;color:#e5e7eb;}
    .chat-card{background:#0f172a;border-radius:10px;padding:8px;margin-bottom:6px;}
    .chat-card:hover{background:#111827;}
    </style>
    """, unsafe_allow_html=True)


def typing(text):
    box = st.empty()
    out=""
    for c in text:
        out+=c
        box.markdown(out)
        time.sleep(0.008)


# =========================================================
# MAIN
# =========================================================

def main():
    inject_css()

    user = get_user()

    if not user:
        st.title("🔐 SlideSense Login")
        t1,t2 = st.tabs(["Login","Signup"])

        with t1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                err = sign_in(u,p)
                if err: st.error(err)
                else: st.rerun()

        with t2:
            u = st.text_input("New Username")
            p = st.text_input("New Password", type="password")
            if st.button("Create"):
                err = sign_up(u,p)
                if err: st.error(err)
                else: st.success("Account created")

        return

    # Sidebar
    st.sidebar.markdown(f"👤 **{user['username']}**")
    if st.sidebar.button("Logout"):
        sign_out()
        st.rerun()

    st.sidebar.markdown("## 💬 Chats")

    chats = fetch_user_chats(user["id"])

    active_chat = st.session_state.get("active_chat")

    for c in chats:
        meta = c.get("chat_metadata",{})
        title = c.get("title","Chat")
        preview = meta.get("last_message_preview","")
        if st.sidebar.button(f"{title}\n{preview[:30]}", key=c["id"]):
            st.session_state["active_chat"] = c["id"]
            st.session_state["mode"] = c["mode"]
            st.rerun()

    if st.sidebar.button("➕ New PDF Chat"):
        st.session_state["active_chat"] = None
        st.session_state["mode"] = "pdf"

    if st.sidebar.button("➕ New Image Chat"):
        st.session_state["active_chat"] = None
        st.session_state["mode"] = "image"

    mode = st.session_state.get("mode")

    if not mode:
        st.title("🚀 Start a new chat")
        return

    # ---------------- PDF MODE ----------------
    if mode=="pdf":
        st.title("📘 PDF Analyzer")

        pdf = st.file_uploader("Upload PDF", type="pdf")

        if "pdf_index" not in st.session_state:
            st.session_state["pdf_index"] = None

        if pdf and st.session_state["pdf_index"] is None:
            reader = PdfReader(pdf)
            text=""
            for p in reader.pages:
                if p.extract_text():
                    text+=p.extract_text()+"\n"
            st.session_state["pdf_index"] = build_index(text)

        if active_chat:
            msgs = fetch_chat_messages(active_chat)
            for m in msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        q = st.chat_input("Ask your PDF")

        if q:
            if not active_chat:
                chat = create_chat(user["id"], "pdf", q)
                active_chat = chat["id"]
                st.session_state["active_chat"]=active_chat

            append_message(active_chat,"user",q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans,_ = answer_pdf(st.session_state["pdf_index"], q)
                    typing(ans)

            append_message(active_chat,"assistant",ans)
            update_metadata(active_chat, ans, estimate_tokens(q)+estimate_tokens(ans))
            st.rerun()

    # ---------------- IMAGE MODE ----------------
    if mode=="image":
        st.title("🖼 Image Q&A")
        img_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

        if active_chat:
            msgs = fetch_chat_messages(active_chat)
            for m in msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        q = st.chat_input("Ask about image")

        if q:
            if not img_file:
                st.error("Upload image first")
                return

            if not active_chat:
                chat = create_chat(user["id"], "image", q)
                active_chat = chat["id"]
                st.session_state["active_chat"]=active_chat

            img = Image.open(img_file).convert("RGB")

            append_message(active_chat,"user",q)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    ans = answer_image(img,q)
                    typing(ans)

            append_message(active_chat,"assistant",ans)
            update_metadata(active_chat, ans, estimate_tokens(q)+estimate_tokens(ans))
            st.rerun()


if __name__=="__main__":
    main()
