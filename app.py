import os
import time
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from PyPDF2 import PdfReader
import numpy as np

try:
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor
    _BLIP_AVAILABLE = True
except Exception:
    _BLIP_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# =========================================================
# CONFIG & GLOBALS
# =========================================================

st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

logger = logging.getLogger("slidesense")
logging.basicConfig(level=logging.INFO)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
USER_TABLE = "users"
CHATS_TABLE = "chats"
MESSAGES_TABLE = "messages"
CHAT_META_TABLE = "chat_metadata"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()

# Memory / token limits (rough, character-based approximation)
MAX_CONTEXT_TOKENS = 2500
SUMMARIZE_AFTER_MESSAGES = 24


# =========================================================
# SUPABASE LOW-LEVEL CLIENT
# =========================================================

def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def sb_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=_supabase_headers(),
            params=params,
            json=json_body,
            timeout=15,
        )
        return resp
    except Exception as e:
        logger.exception("Supabase request failed")
        raise RuntimeError(f"Supabase request failed: {e}") from e


def sb_get(path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    resp = sb_request("GET", path, params=params)
    if resp.status_code >= 400:
        logger.error("Supabase GET error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(resp.text)
    try:
        data = resp.json()
    except Exception:
        data = []
    return data or []


def sb_post(path: str, json_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    resp = sb_request("POST", path, json_body=json_body)
    if resp.status_code >= 400:
        logger.error("Supabase POST error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(resp.text)
    try:
        data = resp.json()
    except Exception:
        data = []
    return data or []


def sb_patch(path: str, json_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    resp = sb_request("PATCH", path, json_body=json_body)
    if resp.status_code >= 400:
        logger.error("Supabase PATCH error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(resp.text)
    try:
        data = resp.json()
    except Exception:
        data = []
    return data or []


# =========================================================
# AUTH & USERS
# =========================================================

import hashlib


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("user")


def set_current_user(user: Dict[str, Any]) -> None:
    st.session_state["user"] = user


def clear_session() -> None:
    for k in list(st.session_state.keys()):
        if k not in ("_session",):
            st.session_state.pop(k, None)


def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."

    # Check existing user
    rows = sb_get(
        f"/rest/v1/{USER_TABLE}",
        params={"select": "id,username", "username": f"eq.{username}"},
    )
    if rows:
        return "Username already exists."

    new_user = {
        "username": username,
        "password_hash": hash_password(password),
    }
    created = sb_post(f"/rest/v1/{USER_TABLE}", json_body=new_user)
    if not created:
        return "Failed to create user."
    return None


def sign_in(username: str, password: str) -> Optional[str]:
    rows = sb_get(
        f"/rest/v1/{USER_TABLE}",
        params={
            "select": "id,username,password_hash",
            "username": f"eq.{username}",
        },
    )
    if not rows:
        return "Invalid username or password."

    row = rows[0]
    if row["password_hash"] != hash_password(password):
        return "Invalid username or password."

    user = {"id": row["id"], "username": row["username"]}
    set_current_user(user)
    return None


def change_password(old_password: str, new_password: str) -> Optional[str]:
    user = get_current_user()
    if not user:
        return "Not logged in."

    if len(new_password) < 6:
        return "New password must be at least 6 characters."

    rows = sb_get(
        f"/rest/v1/{USER_TABLE}",
        params={
            "select": "id,username,password_hash",
            "id": f"eq.{user['id']}",
        },
    )
    if not rows:
        return "User not found."

    row = rows[0]
    if row["password_hash"] != hash_password(old_password):
        return "Current password is incorrect."

    sb_patch(
        f"/rest/v1/{USER_TABLE}?id=eq.{user['id']}",
        json_body={"password_hash": hash_password(new_password)},
    )
    return None


def sign_out() -> None:
    clear_session()


# =========================================================
# LLM & MODELS
# =========================================================

@st.cache_resource
def load_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@st.cache_resource
def load_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Uses Google Gemini embeddings via GOOGLE_API_KEY
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


@st.cache_resource
def load_blip():
    if not _BLIP_AVAILABLE:
        raise RuntimeError(
            "Image Q&A requires torch and transformers (BLIP), "
            "which are not available in this environment."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    return processor, model, device


# =========================================================
# CHAT & MEMORY STORE (Supabase)
# =========================================================

def auto_chat_title_from_question(q: str) -> str:
    title = q.strip().replace("\n", " ")
    # Remove weird symbols
    keep_chars = []
    for ch in title:
        if ch.isalnum() or ch.isspace():
            keep_chars.append(ch)
    clean = "".join(keep_chars)
    clean = " ".join(clean.split())
    if len(clean) > 40:
        clean = clean[:37].rstrip() + "..."
    return clean or "New chat"


def estimate_tokens(text: str) -> int:
    # Rough: ~4 chars per token
    return max(1, len(text) // 4)


def create_chat(
    user_id: str,
    mode: str,
    first_question: str,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    title = auto_chat_title_from_question(first_question)
    chat_row = {
        "user_id": user_id,
        "title": title,
        "mode": mode,
        "is_pinned": False,
        "created_at": now,
        "updated_at": now,
    }
    created = sb_post(f"/rest/v1/{CHATS_TABLE}", json_body=chat_row)
    chat = created[0]
    # Create metadata shell
    sb_post(
        f"/rest/v1/{CHAT_META_TABLE}",
        json_body={
            "chat_id": chat["id"],
            "summary": "",
            "token_count": 0,
            "last_message_at": now,
            "last_message_preview": first_question[:120],
        },
    )
    return chat


def update_chat_title(chat_id: str, new_title: str) -> None:
    sb_patch(
        f"/rest/v1/{CHATS_TABLE}?id=eq.{chat_id}",
        json_body={"title": new_title[:80]},
    )


def soft_delete_chat(chat_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sb_patch(
        f"/rest/v1/{CHATS_TABLE}?id=eq.{chat_id}",
        json_body={"deleted_at": now},
    )


def toggle_pin_chat(chat_id: str, current: bool) -> None:
    sb_patch(
        f"/rest/v1/{CHATS_TABLE}?id=eq.{chat_id}",
        json_body={"is_pinned": not current},
    )


def update_chat_metadata_after_message(
    chat_id: str,
    last_message: str,
    delta_tokens: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    # Update chats.updated_at
    sb_patch(
        f"/rest/v1/{CHATS_TABLE}?id=eq.{chat_id}",
        json_body={"updated_at": now},
    )
    # Update metadata
    sb_patch(
        f"/rest/v1/{CHAT_META_TABLE}?chat_id=eq.{chat_id}",
        json_body={
            "last_message_at": now,
            "last_message_preview": last_message[:150],
        },
    )
    # Increment token_count
    rows = sb_get(
        f"/rest/v1/{CHAT_META_TABLE}",
        params={"select": "token_count", "chat_id": f"eq.{chat_id}"},
    )
    if rows:
        current_tokens = rows[0].get("token_count") or 0
        sb_patch(
            f"/rest/v1/{CHAT_META_TABLE}?chat_id=eq.{chat_id}",
            json_body={"token_count": current_tokens + delta_tokens},
        )


def save_pdf_metadata(chat_id: str, pdf_name: str, pdf_text: str) -> None:
    sb_patch(
        f"/rest/v1/{CHAT_META_TABLE}?chat_id=eq.{chat_id}",
        json_body={
            "pdf_name": pdf_name,
            "pdf_text": pdf_text,
        },
    )


def load_pdf_metadata(chat_id: str) -> Optional[Dict[str, Any]]:
    rows = sb_get(
        f"/rest/v1/{CHAT_META_TABLE}",
        params={
            "select": "pdf_name,pdf_text",
            "chat_id": f"eq.{chat_id}",
        },
    )
    if not rows:
        return None
    return rows[0]


def append_message(
    chat_id: str,
    role: str,
    content: str,
) -> None:
    msg = {
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "token_estimate": estimate_tokens(content),
    }
    sb_post(f"/rest/v1/{MESSAGES_TABLE}", json_body=msg)


def fetch_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    rows = sb_get(
        f"/rest/v1/{MESSAGES_TABLE}",
        params={
            "select": "id,role,content,created_at,token_estimate",
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.asc",
        },
    )
    return rows


def fetch_user_chats(user_id: str) -> List[Dict[str, Any]]:
    # Join with metadata to reduce roundtrips
    params = {
        "select": "id,user_id,title,mode,is_pinned,created_at,updated_at,deleted_at,"
                  f"{CHAT_META_TABLE}(last_message_at,last_message_preview)",
        "user_id": f"eq.{user_id}",
        "order": "is_pinned.desc,updated_at.desc",
    }
    rows = sb_get(f"/rest/v1/{CHATS_TABLE}", params=params)
    # Filter out soft-deleted chats
    chats = [c for c in rows if c.get("deleted_at") is None]
    return chats


def summarize_conversation_if_needed(chat_id: str, messages: List[Dict[str, Any]]) -> None:
    if len(messages) < SUMMARIZE_AFTER_MESSAGES:
        return

    # Check current stored summary
    rows = sb_get(
        f"/rest/v1/{CHAT_META_TABLE}",
        params={
            "select": "summary,token_count",
            "chat_id": f"eq.{chat_id}",
        },
    )
    if not rows:
        return

    meta = rows[0]
    current_summary = meta.get("summary") or ""
    token_count = meta.get("token_count") or 0
    if token_count < MAX_CONTEXT_TOKENS:
        return

    llm = load_llm()
    convo_text = ""
    for m in messages[-40:]:
        prefix = "User" if m["role"] == "user" else "Assistant"
        convo_text += f"{prefix}: {m['content']}\n"

    prompt = (
        "You are SlideSense's summarization helper.\n"
        "Summarize the following conversation in a concise paragraph that preserves key context.\n\n"
        f"Existing summary (may be empty):\n{current_summary}\n\n"
        f"Conversation:\n{convo_text}\n\n"
        "New merged summary:"
    )
    try:
        summary = load_llm().invoke(prompt).content
    except Exception as e:
        logger.error("Summarization failed: %s", e)
        return

    new_tokens = estimate_tokens(summary)
    sb_patch(
        f"/rest/v1/{CHAT_META_TABLE}?chat_id=eq.{chat_id}",
        json_body={"summary": summary, "token_count": new_tokens},
    )


def build_context_messages(chat_id: str, mode: str) -> List[Dict[str, str]]:
    """Return trimmed chat history (with summary) for LLM prompting."""
    messages = fetch_chat_messages(chat_id)
    summarize_conversation_if_needed(chat_id, messages)

    rows = sb_get(
        f"/rest/v1/{CHAT_META_TABLE}",
        params={"select": "summary", "chat_id": f"eq.{chat_id}"},
    )
    summary = rows[0].get("summary") if rows else ""

    context_parts: List[str] = []
    if summary:
        context_parts.append(f"Conversation summary so far:\n{summary}\n")

    # Add recent messages until token budget is filled
    running_tokens = estimate_tokens("\n".join(context_parts))
    trimmed = []
    for m in reversed(messages):
        t = estimate_tokens(m["content"])
        if running_tokens + t > MAX_CONTEXT_TOKENS:
            break
        running_tokens += t
        trimmed.append(m)
    trimmed.reverse()

    ctx_messages: List[Dict[str, str]] = []
    if context_parts:
        ctx_messages.append(
            {
                "role": "system",
                "content": "\n".join(context_parts),
            }
        )
    ctx_messages.extend(
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in trimmed
    )
    return ctx_messages


# =========================================================
# RAG: PDF ANALYZER
# =========================================================


class SimpleVectorIndex:
    """Lightweight in-memory vector index using cosine similarity."""

    def __init__(self, texts: List[str], embeddings: np.ndarray):
        self.texts = texts
        self.embeddings = embeddings.astype("float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings = self.embeddings / norms

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        embed_model = load_embeddings()
        q_emb = np.array(embed_model.embed_query(query), dtype="float32")
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        sims = np.dot(self.embeddings, q_emb)
        top_idxs = np.argsort(-sims)[:k]

        class Doc:
            def __init__(self, page_content: str):
                self.page_content = page_content

        return [Doc(self.texts[i]) for i in top_idxs]


def build_pdf_vectorstore_from_text(pdf_text: str) -> SimpleVectorIndex:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_text(pdf_text)
    embed_model = load_embeddings()
    vectors = [embed_model.embed_query(c) for c in chunks]
    return SimpleVectorIndex(chunks, np.array(vectors))


def ensure_pdf_index_for_chat(chat_id: str) -> Optional[SimpleVectorIndex]:
    if "pdf_indices" not in st.session_state:
        st.session_state["pdf_indices"] = {}
    indices = st.session_state["pdf_indices"]

    if chat_id in indices:
        return indices[chat_id]

    meta = load_pdf_metadata(chat_id)
    if not meta or not meta.get("pdf_text"):
        return None

    vector_db = build_pdf_vectorstore_from_text(meta["pdf_text"])
    indices[chat_id] = vector_db
    return vector_db


def ingest_pdf_for_chat(chat_id: str, pdf_file) -> Optional[SimpleVectorIndex]:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        st.error("No readable text found in PDF")
        return None

    save_pdf_metadata(chat_id, pdf_name=pdf_file.name, pdf_text=text)
    vector_db = build_pdf_vectorstore_from_text(text)
    if "pdf_indices" not in st.session_state:
        st.session_state["pdf_indices"] = {}
    st.session_state["pdf_indices"][chat_id] = vector_db
    return vector_db


def answer_pdf_question(
    chat_id: str,
    question: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    vector_db = ensure_pdf_index_for_chat(chat_id)
    if vector_db is None:
        raise RuntimeError("PDF is not loaded for this chat.")

    llm = load_llm()
    docs = vector_db.similarity_search(question, k=5)

    prompt = ChatPromptTemplate.from_template(
        """
You are SlideSense, a persistent conversational AI assistant for analyzing PDFs.

Context:
{context}

Question:
{question}

Rules:
- Answer only from the PDF document context.
- If the information is not in the document, say: "Information not found in the document."
- Be clear and concise.
"""
    )
    chain = create_stuff_documents_chain(llm, prompt)
    res = chain.invoke({"context": docs, "question": question})

    if isinstance(res, dict):
        answer = res.get("output_text", "")
    else:
        answer = str(res)

    sources = []
    for d in docs:
        snippet = d.page_content[:400]
        if len(d.page_content) > 400:
            snippet += "..."
        sources.append(
            {
                "snippet": snippet,
            }
        )
    return answer, sources


# =========================================================
# VISION: IMAGE Q&A
# =========================================================

def answer_image_question(image, question: str) -> str:
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = (
        "You are SlideSense, a helpful vision assistant.\n"
        "User question: {q}\n"
        "Initial vision model answer: {a}\n\n"
        "Rewrite the answer as one clear, natural sentence. No extra details."
    ).format(q=question, a=short_answer)

    return llm.invoke(prompt).content


# =========================================================
# UI HELPERS
# =========================================================

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        body {
            background: radial-gradient(circle at top left, #0f172a, #020617 55%, #020617);
            color: #e5e7eb;
        }
        .main {
            padding-top: 0rem;
        }
        .slidesense-topbar {
            position: sticky;
            top: 0;
            z-index: 999;
            padding: 0.6rem 1.2rem;
            border-radius: 0 0 16px 16px;
            background: rgba(15,23,42,0.85);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(148,163,184,0.35);
        }
        .slidesense-badge {
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            border: 1px solid rgba(148,163,184,0.5);
            background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(147,51,234,0.15));
        }
        .slidesense-chat-card:hover {
            background: rgba(15,23,42,0.9) !important;
            border-color: rgba(148,163,184,0.8) !important;
            transform: translateY(-1px);
        }
        .slidesense-chat-card {
            transition: all 0.15s ease-out;
        }
        .slidesense-input-container {
            position: sticky;
            bottom: 0;
            z-index: 998;
            padding: 0.5rem 0.25rem 0.75rem 0.25rem;
            background: linear-gradient(to top, rgba(15,23,42,0.95), transparent);
        }
        .slidesense-floating-box {
            padding: 0.4rem 0.7rem 0.6rem 0.7rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.5);
            box-shadow: 0 18px 40px rgba(15,23,42,0.7);
        }
        .slidesense-shimmer {
            position: relative;
            overflow: hidden;
            background: linear-gradient(
              90deg,
              rgba(30,41,59,0.9) 0%,
              rgba(15,23,42,0.9) 40%,
              rgba(30,41,59,0.9) 80%
            );
        }
        .slidesense-shimmer::after {
            content: "";
            position: absolute;
            top: 0;
            left: -150px;
            height: 100%;
            width: 150px;
            background: linear-gradient(
                to right,
                rgba(148,163,184,0.05),
                rgba(148,163,184,0.18),
                rgba(148,163,184,0.05)
            );
            animation: shimmer 1.4s infinite;
        }
        @keyframes shimmer {
            0% { transform: translateX(0); }
            100% { transform: translateX(350px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(user: Optional[Dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="slidesense-topbar">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
              <div style="width:30px;height:30px;border-radius:999px;background:radial-gradient(circle at 30% 20%, #38bdf8, #4f46e5 48%, #0f172a 90%);display:flex;align-items:center;justify-content:center;font-size:0.9rem;">📘</div>
              <div>
                <div style="font-weight:600;font-size:0.95rem;">SlideSense</div>
                <div style="font-size:0.75rem;color:#9ca3af;">AI PDF & Vision Workspace</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:0.6rem;">
              <span class="slidesense-badge">Persistent AI Assistant</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_with_copy(answer: str) -> None:
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText({safe_text});"
                style="margin-top:4px;padding:4px 10px;border-radius:999px;border:1px solid rgba(148,163,184,0.6);
                       cursor:pointer;background:rgba(15,23,42,0.9);color:#e5e7eb;font-size:12px;">
            Copy
        </button>
        """,
        height=40,
    )


def render_typing_animation(answer: str, delay: float = 0.01) -> None:
    box = st.empty()
    out = ""
    for ch in answer:
        out += ch
        box.markdown(out)
        time.sleep(delay)


def group_chats_by_date(chats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    now = datetime.now(timezone.utc)
    groups: Dict[str, List[Dict[str, Any]]] = {"Today": [], "Yesterday": [], "Last 7 Days": [], "Older": []}

    for chat in chats:
        # Prefer last_message_at if available
        last_meta = (chat.get(CHAT_META_TABLE) or {}) or {}
        ts_str = last_meta.get("last_message_at") or chat.get("updated_at") or chat.get("created_at")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            groups["Older"].append(chat)
            continue

        delta = now.date() - ts.date()
        if delta.days == 0:
            groups["Today"].append(chat)
        elif delta.days == 1:
            groups["Yesterday"].append(chat)
        elif delta.days <= 7:
            groups["Last 7 Days"].append(chat)
        else:
            groups["Older"].append(chat)

    return groups


def render_sidebar_chat_system(user: Dict[str, Any]) -> Optional[str]:
    st.sidebar.markdown("### 💬 Chats")

    search = st.sidebar.text_input("Search chats", placeholder="Search by title or content...")
    chats = fetch_user_chats(user["id"])

    # Filter by search
    if search:
        s = search.lower()
        filtered = []
        for c in chats:
            title = (c.get("title") or "").lower()
            meta = c.get(CHAT_META_TABLE) or {}
            preview = (meta.get("last_message_preview") or "").lower()
            if s in title or s in preview:
                filtered.append(c)
        chats = filtered

    groups = group_chats_by_date(chats)

    active_chat_id = st.session_state.get("active_chat_id")

    def render_chat_row(chat: Dict[str, Any]) -> None:
        mode_icon = "📘" if chat.get("mode") == "pdf" else "🖼"
        meta = chat.get(CHAT_META_TABLE) or {}
        preview = meta.get("last_message_preview") or ""
        title = chat.get("title") or "Untitled"
        is_pinned = bool(chat.get("is_pinned"))
        created = chat.get("created_at")
        ts_label = ""
        if created:
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                ts_label = ts.strftime("%H:%M")
            except Exception:
                pass

        selected = active_chat_id == chat["id"]
        card_bg = "background:rgba(15,23,42,0.85);" if selected else "background:rgba(15,23,42,0.65);"

        with st.sidebar.container():
            st.markdown(
                f"""
                <div class="slidesense-chat-card"
                     style="{card_bg}border-radius:12px;border:1px solid rgba(51,65,85,0.9);padding:0.45rem 0.55rem;margin-bottom:0.4rem;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.4rem;">
                    <div style="flex:1;">
                      <div style="display:flex;align-items:center;gap:0.35rem;font-size:0.8rem;">
                        <span>{mode_icon}</span>
                        <span style="font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</span>
                        {"<span>📌</span>" if is_pinned else ""}
                      </div>
                      <div style="font-size:0.7rem;color:#9ca3af;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {preview}
                      </div>
                    </div>
                    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.15rem;font-size:0.65rem;color:#9ca3af;">
                      <span>{ts_label}</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.sidebar.columns([1, 1, 1, 2])
            with cols[0]:
                if st.button("✏️", key=f"rename_{chat['id']}"):
                    new_title = st.text_input(
                        "New chat title",
                        value=title,
                        key=f"rename_input_{chat['id']}",
                    )
                    if st.button("Save", key=f"rename_save_{chat['id']}"):
                        update_chat_title(chat["id"], new_title)
                        st.experimental_rerun()
            with cols[1]:
                if st.button("📌" if not is_pinned else "📍", key=f"pin_{chat['id']}"):
                    toggle_pin_chat(chat["id"], is_pinned)
                    st.experimental_rerun()
            with cols[2]:
                if st.button("🗑", key=f"del_{chat['id']}"):
                    soft_delete_chat(chat["id"])
                    if st.session_state.get("active_chat_id") == chat["id"]:
                        st.session_state["active_chat_id"] = None
                    st.experimental_rerun()
            with cols[3]:
                if st.button("Open", key=f"open_{chat['id']}"):
                    st.session_state["active_chat_id"] = chat["id"]
                    st.experimental_rerun()

    # New chat buttons
    st.sidebar.markdown("#### New chat")
    col_pdf, col_img = st.sidebar.columns(2)
    with col_pdf:
        if st.button("📘 PDF", key="new_pdf_chat"):
            st.session_state["active_chat_id"] = None
            st.session_state["new_chat_mode"] = "pdf"
    with col_img:
        if st.button("🖼 Image", key="new_img_chat"):
            st.session_state["active_chat_id"] = None
            st.session_state["new_chat_mode"] = "image"

    st.sidebar.markdown("---")

    for label, chats_group in groups.items():
        if not chats_group:
            continue
        st.sidebar.markdown(f"**{label}**")
        for c in chats_group:
            render_chat_row(c)

    if st.sidebar.button("🧹 Clear all chats", key="clear_all_chats"):
        # Soft-delete all user chats
        sb_patch(
            f"/rest/v1/{CHATS_TABLE}?user_id=eq.{user['id']}",
            json_body={"deleted_at": datetime.now(timezone.utc).isoformat()},
        )
        st.session_state["active_chat_id"] = None
        st.experimental_rerun()

    return st.session_state.get("active_chat_id")


def render_auth_sidebar(user: Optional[Dict[str, Any]]) -> None:
    if user:
        st.sidebar.markdown(f"👤 **{user['username']}**")
        with st.sidebar.expander("Change password"):
            old_pw = st.text_input("Current", type="password", key="pw_old")
            new_pw = st.text_input("New", type="password", key="pw_new")
            if st.button("Update password", key="btn_change_pw"):
                err = change_password(old_pw, new_pw)
                if err:
                    st.sidebar.error(err)
                else:
                    st.sidebar.success("Password updated.")
        if st.sidebar.button("Logout"):
            sign_out()
            st.experimental_rerun()
    else:
        st.sidebar.info("Not logged in")


def render_login_ui() -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔐 Welcome to SlideSense")
        st.caption("Secure workspace for your PDFs and images.")
        st.markdown(
            """
- Persistent chat history  
- PDF RAG with embeddings  
- Vision Q&A with BLIP  
- Powered by Gemini  
"""
        )
    with col2:
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_in(username, password)
                    if err:
                        st.error(err)
                    else:
                        st.experimental_rerun()
        with tab2:
            username = st.text_input("Username", key="su_username")
            password = st.text_input(
                "Password (min 6 chars)", type="password", key="su_password"
            )
            if st.button("Create account"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_up(username, password)
                    if err:
                        st.error(err)
                    else:
                        st.success("Account created! You can now log in.")
        with tab3:
            st.caption("Continue without an account.")
            if st.button("Continue as guest"):
                st.session_state["guest"] = True
                st.experimental_rerun()


def render_conversation(chat_id: str, mode: str) -> None:
    messages = fetch_chat_messages(chat_id)

    st.markdown("### 💬 Conversation")
    for m in messages:
        speaker = "user" if m["role"] == "user" else "assistant"
        with st.chat_message(speaker):
            if m["role"] == "assistant":
                render_answer_with_copy(m["content"])
            else:
                st.markdown(m["content"])


# =========================================================
# MAIN APP FLOW
# =========================================================

def main() -> None:
    inject_global_css()

    user = get_current_user()
    is_guest = st.session_state.get("guest", False)

    render_top_bar(user)

    if not user and not is_guest:
        render_login_ui()
        return

    if is_guest:
        # Create or fetch a shared guest user in DB so chats can be stored
        if "guest_db_user" not in st.session_state:
            try:
                rows = sb_get(
                    f"/rest/v1/{USER_TABLE}",
                    params={"select": "id,username", "username": "eq.guest"},
                )
            except RuntimeError:
                rows = []
            if rows:
                st.session_state["guest_db_user"] = rows[0]
            else:
                created = sb_post(
                    f"/rest/v1/{USER_TABLE}",
                    json_body={
                        "username": "guest",
                        "password_hash": hash_password("guest"),
                    },
                )
                st.session_state["guest_db_user"] = created[0]
        user = st.session_state["guest_db_user"]

    # Layout
    left_col, center_col = st.columns([0.28, 0.72])

    with left_col:
        render_auth_sidebar(None if is_guest else user)
        active_chat_id = render_sidebar_chat_system(user)

    with center_col:
        # Decide mode: existing chat or new chat mode
        active_mode = None
        if active_chat_id:
            # Fetch chat to know its mode
            rows = sb_get(
                f"/rest/v1/{CHATS_TABLE}",
                params={
                    "select": "id,mode,title",
                    "id": f"eq.{active_chat_id}",
                },
            )
            if rows:
                active_mode = rows[0]["mode"]
        else:
            active_mode = st.session_state.get("new_chat_mode")

        if not active_mode:
            st.markdown("### Start a new conversation")
            st.caption("Choose PDF Analyzer or Image Q&A from the left sidebar.")
            return

        # MODE: PDF ANALYZER CHAT
        if active_mode == "pdf":
            st.markdown("### 📘 PDF Analyzer")
            st.caption("Upload a PDF and chat with SlideSense about its contents.")
            st.divider()

            chat_id = active_chat_id

            # For new chat with no id yet, we ingest PDF once user asks first question.
            # But we still show uploader now.
            pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

            # Show existing conversation (if chat already exists)
            if chat_id:
                render_conversation(chat_id, "pdf")

            # Floating input
            with st.container():
                st.markdown('<div class="slidesense-input-container">', unsafe_allow_html=True)
                with st.container():
                    user_q = st.chat_input("Ask a question about your PDF")
                st.markdown("</div>", unsafe_allow_html=True)

            if user_q:
                # Create chat if needed
                if not chat_id:
                    chat = create_chat(user["id"], "pdf", user_q)
                    chat_id = chat["id"]
                    st.session_state["active_chat_id"] = chat_id

                # Ensure PDF index for this chat
                # If first time, use uploaded file; otherwise rebuild from stored text
                index = ensure_pdf_index_for_chat(chat_id)
                if index is None:
                    if not pdf_file:
                        st.error("Please upload a PDF for this chat before asking questions.")
                        return
                    index = ingest_pdf_for_chat(chat_id, pdf_file)
                    if index is None:
                        return

                # Append user message
                append_message(chat_id, "user", user_q)

                # Answer
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer, sources = answer_pdf_question(chat_id, user_q)
                        except Exception as e:
                            st.error(f"Error answering question: {e}")
                            return
                        # Typing effect for new answer
                        render_typing_animation(answer)
                        st.markdown("")  # spacer
                        render_answer_with_copy(answer)

                # Append assistant message
                append_message(chat_id, "assistant", answer)

                # Update metadata
                total_tokens = estimate_tokens(user_q) + estimate_tokens(answer)
                update_chat_metadata_after_message(chat_id, answer, total_tokens)

                # Show sources
                st.markdown("### 🔍 Sources used")
                for idx, src in enumerate(sources, start=1):
                    with st.expander(f"Source {idx}"):
                        st.write(src["snippet"])

                st.experimental_rerun()

        # MODE: IMAGE Q&A CHAT
        elif active_mode == "image":
            st.markdown("### 🖼 Image Q&A")
            st.caption("Upload an image and ask SlideSense questions about it.")
            st.divider()

            chat_id = active_chat_id

            # Show existing messages
            if chat_id:
                render_conversation(chat_id, "image")

            img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="img_uploader")

            with st.container():
                st.markdown('<div class="slidesense-input-container">', unsafe_allow_html=True)
                with st.container():
                    user_q = st.chat_input("Ask a question about the image")
                st.markdown("</div>", unsafe_allow_html=True)

            if user_q:
                if not img_file:
                    st.error("Please upload an image before asking a question.")
                    return

                if not chat_id:
                    chat = create_chat(user["id"], "image", user_q)
                    chat_id = chat["id"]
                    st.session_state["active_chat_id"] = chat_id

                img = Image.open(img_file).convert("RGB")
                st.image(img, use_column_width=True)

                append_message(chat_id, "user", user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing image..."):
                        try:
                            answer = answer_image_question(img, user_q)
                        except Exception as e:
                            st.error(f"Error answering question: {e}")
                            return
                        render_typing_animation(answer)
                        st.markdown("")
                        render_answer_with_copy(answer)

                append_message(chat_id, "assistant", answer)
                total_tokens = estimate_tokens(user_q) + estimate_tokens(answer)
                update_chat_metadata_after_message(chat_id, answer, total_tokens)
                st.experimental_rerun()

        else:
            st.error(f"Unknown chat mode: {active_mode}")


if __name__ == "__main__":
    main()

