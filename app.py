import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib, re, base64
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components

from PyPDF2 import PdfReader
from PIL import Image
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
USER_PROFILES_TABLE = "user_profiles"

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()


# ==================== CACHED MODELS ====================
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@st.cache_resource
def load_embeddings():
    # Google embeddings — no torch download needed
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


@st.cache_resource
def load_spacy_model():
    """Load spaCy en_core_web_sm. Returns None gracefully if unavailable."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


# ==================== NLP PIPELINE ====================
# Techniques applied automatically on every question — no button, no toggle.
#
#  1. Tokenisation          – split text into tokens          (spaCy)
#  2. POS Tagging           – label each token NOUN/VERB/ADJ… (spaCy)
#  3. Named Entity Recog.   – detect PERSON, ORG, DATE…       (spaCy)
#  4. Lemmatisation         – map words to base form           (spaCy)
#  5. Stop-word Removal     – drop filler words                (spaCy / regex fallback)
#  6. Query Normalisation   – LLM fixes grammar & intent       (Gemini)
#  7. Semantic Search       – cosine similarity via FAISS      (Google embeddings)
#  8. Entity Highlighting   – bold key terms in answer         (regex)
# ======================================================

_STOPWORDS = {
    "what","is","are","how","the","a","an","of","in","on","at","your","my",
    "our","their","its","do","does","did","tell","me","give","please","who",
    "where","when","why","this","that","was","were","has","have","had","be",
    "been","being","and","or","but","if","then","so","just","also","about","with",
}


def nlp_extract_entities_and_keywords(text: str) -> List[str]:
    """Techniques 1–5: tokenise → POS tag → NER → lemmatise → remove stopwords."""
    nlp = load_spacy_model()
    if nlp:
        doc = nlp(text)
        ner_terms = [ent.text.lower() for ent in doc.ents]
        lemma_terms = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN", "ADJ")
            and not token.is_stop
            and not token.is_punct
            and len(token.text) > 1
        ]
        seen, result = set(), []
        for t in ner_terms + lemma_terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result
    else:
        # Fallback: regex tokenise + stopword filter
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [w for w in words if w not in _STOPWORDS]


def nlp_normalize_query(raw_query: str) -> str:
    """Technique 6: LLM grammar/intent normalisation. Skips if query looks fine."""
    words = raw_query.strip().split()
    if len(words) >= 4 and raw_query.strip().endswith("?"):
        return raw_query
    llm = load_llm()
    prompt = (
        "You are a query normalisation assistant.\n"
        "Rewrite the user's question into a clean, grammatically correct English question "
        "that preserves the original intent exactly.\n"
        "Rules:\n"
        "- Keep the same topic and key nouns\n"
        "- Fix grammar (e.g. 'how is your name' → 'What is the name?')\n"
        "- Do NOT add new information\n"
        "- Return ONLY the corrected question — no explanation, no quotes\n\n"
        f"Raw question: {raw_query}\n"
        "Corrected question:"
    )
    try:
        result = load_llm().invoke(prompt).content.strip().strip('"').strip("'")
        return result if result and len(result) > 3 else raw_query
    except Exception:
        return raw_query


def nlp_highlight_entities(answer: str, entities: List[str]) -> str:
    """Technique 8: bold key entities in the answer text."""
    if not entities:
        return answer
    result = answer
    for entity in sorted(entities, key=len, reverse=True):
        pattern = re.compile(r"(?<!\*)\b" + re.escape(entity) + r"\b(?!\*)", re.IGNORECASE)
        result = pattern.sub(lambda m: f"**{m.group(0)}**", result)
    return result


def build_nlp_prompt(original_q: str, normalized_q: str, entities: List[str]) -> ChatPromptTemplate:
    """Entity-aware answer prompt that focuses the LLM on what the user asked about."""
    entity_hint = ", ".join(entities) if entities else "general topic"
    template = (
        "You are a helpful document assistant.\n\n"
        f"The user asked: \"{original_q}\"\n"
        f"Interpreted as: \"{normalized_q}\"\n"
        f"Key entities / topics: [{entity_hint}]\n\n"
        "Answer ONLY from the context below. "
        "If not found, say: 'Information not found in the document.'\n\n"
        "Context:\n{context}\n\nAnswer:"
    )
    return ChatPromptTemplate.from_template(template)


# ==================== IMAGE Q&A (Gemini Vision — no torch) ====================
def answer_image_question(image: Image.Image, question: str) -> str:
    """
    Use Gemini's native vision capability instead of BLIP.
    Eliminates torch + transformers (~3 GB) from dependencies entirely.
    """
    import google.generativeai as genai

    # Normalise the question first (NLP Technique 6)
    normalized_q = nlp_normalize_query(question)

    # Convert PIL image to bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([
        {"mime_type": "image/png", "data": img_bytes},
        normalized_q,
    ])
    return response.text.strip()


# ==================== AUTH HELPERS ====================
def _auth_headers() -> Dict[str, str]:
    return {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}

def _rest_request(method: str, path: str, **kwargs) -> requests.Response:
    return requests.request(
        method, f"{SUPABASE_URL}{path}", headers=_auth_headers(), timeout=10, **kwargs
    )

def set_session(user: Dict[str, Any]) -> None:
    st.session_state["session"] = {"user": user}

def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    return sess["user"] if sess else None

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."
    resp = _rest_request("GET", f"/rest/v1/{USER_PROFILES_TABLE}?select=username&username=eq.{username}")
    if resp.status_code >= 400:
        return resp.text
    if resp.json():
        return "Username already exists."
    resp = _rest_request("POST", f"/rest/v1/{USER_PROFILES_TABLE}",
                         json={"username": username, "password_hash": _hash_password(password)})
    return resp.text if resp.status_code >= 400 else None

def sign_in(username: str, password: str) -> Optional[str]:
    resp = _rest_request("GET",
        f"/rest/v1/{USER_PROFILES_TABLE}?select=id,username,password_hash&username=eq.{username}")
    if resp.status_code >= 400:
        return resp.text
    rows = resp.json()
    if not rows:
        return "Invalid username or password."
    row = rows[0]
    if row.get("password_hash") != _hash_password(password):
        return "Invalid username or password."
    set_session({"id": row.get("id"), "username": row["username"]})
    return None

def sign_out() -> None:
    st.session_state.pop("session", None)


# ==================== HELPERS ====================
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def type_text(text: str, speed: float = 0.03):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)

def render_answer_with_copy(answer: str, key_suffix: str) -> None:
    st.markdown(answer)
    safe_text = json.dumps(answer)
    components.html(
        f"""<button onclick="navigator.clipboard.writeText({safe_text});"
            style="margin-top:4px;padding:4px 10px;border-radius:4px;
                   border:1px solid #ccc;cursor:pointer;">Copy</button>""",
        height=40,
    )

def get_display_name(user: Dict[str, Any]) -> str:
    if user.get("username"):
        return user["username"]
    email = user.get("email", "")
    return email.split("@")[0] if "@" in email else email


# ==================== CHAT HISTORY ====================
def load_chat_history_from_db() -> None:
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user or not user.get("username"):
        return
    resp = _rest_request("GET",
        f"/rest/v1/chat_history?select=question,answer,mode,created_at"
        f"&username=eq.{user['username']}&mode=eq.pdf&order=created_at.asc")
    if resp.status_code >= 400:
        return
    st.session_state.chat_history = [(r["question"], r["answer"]) for r in resp.json()]
    st.session_state.history_loaded = True

def save_chat_to_db(question: str, answer: str) -> None:
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user or not user.get("username"):
        return
    _rest_request("POST", "/rest/v1/chat_history",
                  json={"username": user["username"], "mode": "pdf",
                        "question": question, "answer": answer})


# ==================== SESSION DEFAULTS ====================
defaults: Dict[str, Any] = {
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
    "last_sources": [],
    "last_nlp_meta": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==================== AUTH UI ====================
def login_ui():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"), height=300)
    with col2:
        type_text("🔐 Welcome to SlideSense")
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_in(username, password)
                    st.error(f"Login failed: {err}") if err else st.rerun()
        with tab2:
            username = st.text_input("Username", key="signup_username")
            password = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
            if st.button("Create Account"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_up(username, password)
                    st.error(f"Sign-up failed: {err}") if err else st.success("Account created! Log in now.")
        with tab3:
            st.markdown("Continue without creating an account.")
            if st.button("Continue as guest"):
                st.session_state["guest"] = True
                st.rerun()


# ==================== AUTH CHECK ====================
user = current_user()
if not user and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# ==================== SIDEBAR ====================
user = current_user()
st.sidebar.success(f"Logged in as {get_display_name(user)}" if user else "Logged in as Guest")

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k, v in defaults.items():
        st.session_state[k] = v
    sign_out()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

st.sidebar.markdown("### 💬 Chat History")
if st.session_state.chat_history:
    items = list(reversed(list(enumerate(st.session_state.chat_history, start=1))))
    labels = [f"{idx}. {q[:40]}..." for idx, (q, _) in items]
    selected_label = st.sidebar.selectbox("Select a message", options=labels,
                                          label_visibility="collapsed", key="history_select")
    if selected_label:
        sel_idx = int(selected_label.split(".")[0])
        q_sel, a_sel = st.session_state.chat_history[sel_idx - 1]
        with st.sidebar.expander("Selected chat", expanded=True):
            st.markdown("**You**")
            st.write(q_sel)
            st.markdown("**Assistant**")
            st.write(a_sel)
    if st.sidebar.button("🧹 Clear History"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.sidebar.caption("No history yet")


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    c1, c2 = st.columns([1, 3])
    with c1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"), height=180)
    with c2:
        st.markdown("## 📘 PDF Analyzer")
        st.caption("Upload a PDF and ask anything — typos and loose grammar are auto-corrected by NLP.")
    st.divider()

    pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []
            st.session_state.history_loaded = False
            st.session_state.last_sources = []
            st.session_state.last_nlp_meta = {}

        if not st.session_state.history_loaded and not st.session_state.get("guest"):
            load_chat_history_from_db()

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF…"):
                reader = PdfReader(pdf)
                page_texts = [
                    (i + 1, page.extract_text())
                    for i, page in enumerate(reader.pages)
                    if page.extract_text()
                ]
                if not page_texts:
                    st.error("No readable text found in PDF.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                all_chunks, metadatas = [], []
                for page_num, text in page_texts:
                    chunks = splitter.split_text(text)
                    all_chunks.extend(chunks)
                    metadatas.extend([{"page": page_num}] * len(chunks))

                # Technique 7: semantic embeddings via Google (no torch required)
                embeddings = load_embeddings()
                st.session_state.vector_db = FAISS.from_texts(all_chunks, embeddings, metadatas=metadatas)

        user_q = st.chat_input("Ask anything about this PDF — any phrasing works!")

        if user_q:
            # ── NLP PIPELINE ────────────────────────────────────────────────
            # Techniques 1–5
            entities = nlp_extract_entities_and_keywords(user_q)
            # Technique 6
            normalized_q = nlp_normalize_query(user_q)
            # Technique 7
            docs = st.session_state.vector_db.similarity_search(normalized_q, k=5)
            # Answer generation
            prompt = build_nlp_prompt(user_q, normalized_q, entities)
            chain = create_stuff_documents_chain(load_llm(), prompt)
            res = chain.invoke({"context": docs, "question": normalized_q})
            raw_answer = res if isinstance(res, str) else res.get("output_text", "")
            # Technique 8
            display_answer = nlp_highlight_entities(raw_answer, entities)
            # ────────────────────────────────────────────────────────────────

            st.session_state.chat_history.append((user_q, display_answer))
            save_chat_to_db(user_q, display_answer)
            st.session_state.last_sources = [
                {"page": d.metadata.get("page"),
                 "snippet": d.page_content[:400] + ("…" if len(d.page_content) > 400 else "")}
                for d in docs
            ]
            st.session_state.last_nlp_meta = {
                "original": user_q,
                "normalized": normalized_q,
                "entities": entities,
            }

        # Conversation
        st.markdown("## 💬 Conversation")
        for i, (uq, ua) in enumerate(reversed(st.session_state.chat_history)):
            with st.chat_message("user"):
                st.markdown(uq)
            with st.chat_message("assistant"):
                render_answer_with_copy(ua, key_suffix=f"pdf_{i}")

        # NLP Insight panel
        nlp_meta = st.session_state.get("last_nlp_meta", {})
        if nlp_meta:
            with st.expander("🧠 NLP Insight — how your question was understood", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original question**")
                    st.info(nlp_meta.get("original", "—"))
                    st.markdown("**Normalised question** *(Technique 6)*")
                    st.success(nlp_meta.get("normalized", "—"))
                with col_b:
                    st.markdown("**Extracted entities & keywords** *(Techniques 1–5)*")
                    ents = nlp_meta.get("entities", [])
                    st.markdown(" ".join(f"`{e}`" for e in ents) if ents else "*None detected*")
                st.caption(
                    "Techniques: Tokenisation · POS Tagging · NER · Lemmatisation · "
                    "Stop-word Removal · Query Normalisation · Semantic Search · Entity Highlighting"
                )

        # Sources
        if st.session_state.get("last_sources"):
            st.markdown("### 🔍 Sources used")
            for idx, src in enumerate(st.session_state.last_sources, start=1):
                with st.expander(f"Source {idx} • Page {src.get('page', '?')}"):
                    st.write(src.get("snippet", ""))


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    c1, c2 = st.columns([1, 3])
    with c1:
        st_lottie(load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"), height=180)
    with c2:
        st.markdown("## 🖼 Image Q&A")
        st.caption("Upload an image and ask questions about it.")
    st.divider()

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)
        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analysing image…"):
                ans = answer_image_question(img, question)
            st.success("Answer:")
            render_answer_with_copy(ans, key_suffix="img_answer")
            normalized_q = nlp_normalize_query(question)
            if normalized_q.strip().lower() != question.strip().lower():
                st.caption(f"🧠 Question interpreted as: *{normalized_q}*")
