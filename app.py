import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib, re
from typing import Any, Dict, Optional, List
import streamlit.components.v1 as components

from PyPDF2 import PdfReader
from PIL import Image
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering

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


# ==================== NLP UTILITIES ====================

@st.cache_resource
def load_spacy():
    """
    Load spaCy model for NLP processing.
    Falls back to None gracefully if model or spaCy is unavailable
    (e.g. on Streamlit Cloud where subprocess installs are blocked).
    """
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed — try pip install via sys (safer than subprocess)
            try:
                import sys
                import importlib
                from pip._internal.cli.main import main as pip_main
                pip_main(["install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl", "--quiet"])
                importlib.invalidate_caches()
                return spacy.load("en_core_web_sm")
            except Exception:
                # Give up silently — regex fallback will be used
                return None
    except ImportError:
        return None


def extract_key_entities(text: str) -> List[str]:
    """
    Extract key nouns and named entities from a user query using spaCy.
    Falls back to simple regex-based noun extraction if spaCy unavailable.
    """
    nlp = load_spacy()
    if nlp:
        doc = nlp(text)
        # Named entities (PERSON, ORG, GPE, PRODUCT, etc.)
        entities = [ent.text.lower() for ent in doc.ents]
        # Nouns and proper nouns not already in entities
        nouns = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN")
            and token.lemma_.lower() not in entities
            and not token.is_stop
        ]
        return list(dict.fromkeys(entities + nouns))  # deduplicated, order preserved
    else:
        # Fallback: simple word extraction ignoring common stopwords
        stopwords = {
            "what", "is", "are", "how", "the", "a", "an", "of", "in",
            "your", "my", "our", "their", "its", "do", "does", "tell",
            "me", "give", "name", "please", "who", "where", "when", "why"
        }
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return [w for w in words if w not in stopwords]


def normalize_query(raw_query: str) -> str:
    """
    Use the LLM to normalize/rephrase a malformed or informal user question
    into a clean, grammatically correct search query while preserving intent.
    
    Examples:
      "how is your name"          → "What is the name?"
      "name the name"             → "What is the name mentioned?"
      "who write this"            → "Who wrote this document?"
    """
    llm = load_llm()
    prompt = f"""You are a query normalization assistant. 
Your job: rewrite the user's question into a clean, grammatically correct English question that preserves the original intent.

Rules:
- Keep the same topic and key nouns
- Fix grammar mistakes (e.g. "how is your name" → "What is the name?")
- Do NOT add new information
- Return ONLY the corrected question, nothing else

User's raw question: "{raw_query}"
Corrected question:"""
    try:
        result = llm.invoke(prompt).content.strip().strip('"')
        return result if result else raw_query
    except Exception:
        return raw_query


def build_entity_aware_answer_prompt(user_question: str, normalized_question: str, entities: List[str]) -> ChatPromptTemplate:
    """
    Build a prompt that instructs the LLM to:
    1. Answer based on document context
    2. Use the same terminology/phrasing style the user used
    3. Highlight the key entities the user asked about
    """
    entities_str = ", ".join(entities) if entities else "N/A"

    template = f"""You are a helpful document assistant. Answer the user's question based ONLY on the provided context.

Key entities the user is asking about: [{entities_str}]
User's original question (may have grammar issues): "{user_question}"
Normalized/clarified question: "{normalized_question}"

Rules:
- Answer ONLY from the document context below
- If the user asked about specific entities (e.g. a name, place, date), make sure your answer prominently features those entities
- Mirror the user's terminology — if they said "name the name", respond in a way that directly names what they asked
- Keep the answer concise and direct
- If the answer is not found in the context, say: "Information not found in the document"

Context:
{{context}}

Answer:"""

    return ChatPromptTemplate.from_template(template)


def highlight_entities_in_answer(answer: str, entities: List[str]) -> str:
    """
    Wrap key entities in markdown bold so they stand out in the answer.
    """
    if not entities:
        return answer

    result = answer
    for entity in entities:
        # Case-insensitive replacement, avoid double-bolding
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        # Only bold if not already bolded
        result = pattern.sub(lambda m: f"**{m.group(0)}**" if "**" not in answer[max(0, answer.find(m.group(0))-2):answer.find(m.group(0))+len(m.group(0))+2] else m.group(0), result)
    return result


# -------------------- AUTH HELPERS (Supabase HTTP) --------------------
def _auth_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }


def _rest_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=10, **kwargs)


def set_session(user: Dict[str, Any]) -> None:
    st.session_state["session"] = {"user": user}


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["user"]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."

    resp = _rest_request(
        "GET",
        f"/rest/v1/{USER_PROFILES_TABLE}"
        f"?select=username&username=eq.{username}",
    )
    if resp.status_code >= 400:
        return resp.text

    existing = resp.json()
    if existing:
        return "Username already exists."

    payload = {
        "username": username,
        "password_hash": _hash_password(password),
    }

    resp = _rest_request("POST", f"/rest/v1/{USER_PROFILES_TABLE}", json=payload)
    if resp.status_code >= 400:
        return resp.text
    return None


def sign_in(username: str, password: str) -> Optional[str]:
    resp = _rest_request(
        "GET",
        f"/rest/v1/{USER_PROFILES_TABLE}"
        f"?select=id,username,password_hash&username=eq.{username}",
    )
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


# -------------------- HELPERS --------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


def type_text(text, speed=0.03):
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
        f"""
        <button onclick="navigator.clipboard.writeText({safe_text});"
                style="margin-top:4px;padding:4px 10px;border-radius:4px;border:1px solid #ccc;cursor:pointer;">
            Copy
        </button>
        """,
        height=40,
    )


def get_display_name(user: Dict[str, Any]) -> str:
    if user.get("username"):
        return user["username"]
    meta = user.get("user_metadata") or {}
    if meta.get("username"):
        return meta["username"]
    email = user.get("email", "")
    return email.split("@")[0] if "@" in email else email


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


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
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
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
            height=300,
        )

    with col2:
        type_text("🔐 Welcome to SlideSense")

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
                        st.error(f"Login failed: {err}")
                    else:
                        st.rerun()

        with tab2:
            username = st.text_input("Username", key="signup_username")
            password = st.text_input(
                "Password (min 6 chars)", type="password", key="signup_password"
            )
            if st.button("Create Account"):
                if not username or not password:
                    st.warning("Enter username and password.")
                else:
                    err = sign_up(username, password)
                    if err:
                        st.error(f"Sign-up failed: {err}")
                    else:
                        st.success("Account created! You can now log in.")

        with tab3:
            st.markdown("Continue without creating an account.")
            if st.button("Continue as guest"):
                st.session_state["guest"] = True
                st.rerun()


# -------------------- CHAT HISTORY PERSISTENCE --------------------
def load_chat_history_from_db() -> None:
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    username = user.get("username")
    if not username:
        return

    resp = _rest_request(
        "GET",
        "/rest/v1/chat_history"
        "?select=question,answer,mode,created_at"
        f"&username=eq.{username}"
        "&mode=eq.pdf"
        "&order=created_at.asc",
    )
    if resp.status_code >= 400:
        return

    rows: List[Dict[str, Any]] = resp.json()
    st.session_state.chat_history = [(r["question"], r["answer"]) for r in rows]
    st.session_state.history_loaded = True


def save_chat_to_db(question: str, answer: str) -> None:
    if st.session_state.get("guest"):
        return
    user = current_user()
    if not user:
        return
    username = user.get("username")
    if not username:
        return

    payload = {"username": username, "mode": "pdf", "question": question, "answer": answer}
    _rest_request("POST", "/rest/v1/chat_history", json=payload)


# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}
Convert into one clear sentence. No extra details.
"""
    return llm.invoke(prompt).content


# -------------------- AUTH CHECK --------------------
user = current_user()
if (not user) and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
user = current_user()
if user:
    display_name = get_display_name(user)
    label = f"Logged in as {display_name}"
elif st.session_state.get("guest"):
    label = "Logged in as Guest"
else:
    label = "Not logged in"

st.sidebar.success(label)

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.session_state["guest"] = False
    sign_out()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- NLP SETTINGS IN SIDEBAR --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 NLP Settings")

nlp_enabled = st.sidebar.toggle(
    "Enable Query Normalization",
    value=True,
    help="Automatically fix grammar and extract key terms from your questions for better answers.",
)

show_nlp_debug = st.sidebar.toggle(
    "Show NLP Debug Info",
    value=False,
    help="Display normalized query and extracted entities below each answer.",
)

# -------------------- SIDEBAR HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

if st.session_state.chat_history:
    items = list(reversed(list(enumerate(st.session_state.chat_history, start=1))))
    labels = [f"{idx}. {q[:40]}..." for idx, (q, _) in items]

    selected_label = st.sidebar.selectbox(
        "Select a message",
        options=labels,
        label_visibility="collapsed",
        key="history_select",
    )

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
    pdf_col_anim, pdf_col_title = st.columns([1, 3])

    with pdf_col_anim:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
            height=180,
        )

    with pdf_col_title:
        st.markdown("## 📘 PDF Analyzer")
        st.caption("Upload a PDF and ask questions about its content.")

        # NLP feature badge
        if nlp_enabled:
            st.markdown(
                "🧠 **NLP Query Normalization ON** — your questions are auto-corrected & entity-aware",
                help="Even informal or grammatically loose questions like 'how is your name' or 'name the name' will be understood correctly."
            )

    st.divider()

    pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if (not st.session_state.get("history_loaded")) and (not st.session_state.get("guest")):
            load_chat_history_from_db()

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                page_texts = []

                for page_num, pdf_page in enumerate(reader.pages, start=1):
                    extracted = pdf_page.extract_text()
                    if extracted:
                        page_texts.append((page_num, extracted))

                if not page_texts:
                    st.error("No readable text found in PDF")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

                all_chunks: List[str] = []
                metadatas: List[Dict[str, Any]] = []
                for page_num, page_text in page_texts:
                    page_chunks = splitter.split_text(page_text)
                    all_chunks.extend(page_chunks)
                    metadatas.extend([{"page": page_num} for _ in page_chunks])

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(
                    all_chunks, embeddings, metadatas=metadatas
                )

        user_q = st.chat_input("Ask a question about this PDF (any phrasing works!)")

        if user_q:
            llm = load_llm()

            # -------- NLP PIPELINE --------
            nlp_meta = {}

            if nlp_enabled:
                with st.spinner("🧠 Analyzing your question..."):
                    # Step 1: Extract entities/key terms from the raw question
                    entities = extract_key_entities(user_q)

                    # Step 2: Normalize the query (fix grammar, preserve intent)
                    normalized_q = normalize_query(user_q)

                    nlp_meta = {
                        "entities": entities,
                        "normalized": normalized_q,
                    }

                # Search using normalized query for better retrieval
                search_query = normalized_q
                prompt = build_entity_aware_answer_prompt(user_q, normalized_q, entities)
            else:
                entities = []
                normalized_q = user_q
                search_query = user_q
                prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            # -------- RETRIEVAL & ANSWER --------
            docs = st.session_state.vector_db.similarity_search(search_query, k=5)
            chain = create_stuff_documents_chain(llm, prompt)

            invoke_input = {"context": docs, "question": search_query}
            res = chain.invoke(invoke_input)

            if isinstance(res, dict):
                answer = res.get("output_text", "")
            else:
                answer = res

            # Step 3: Highlight entities in the answer
            if nlp_enabled and entities:
                display_answer = highlight_entities_in_answer(answer, entities)
            else:
                display_answer = answer

            st.session_state.chat_history.append((user_q, display_answer))
            save_chat_to_db(user_q, display_answer)

            # Store sources and NLP metadata for display
            st.session_state["last_sources"] = [
                {
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
                }
                for d in docs
            ]
            st.session_state["last_nlp_meta"] = nlp_meta

        # -------- CHAT DISPLAY --------
        st.markdown("## 💬 Conversation")

        chat_container = st.container()
        with chat_container:
            for i, (uq, ua) in enumerate(reversed(st.session_state.chat_history)):
                with st.chat_message("user"):
                    st.markdown(uq)
                with st.chat_message("assistant"):
                    render_answer_with_copy(ua, key_suffix=f"pdf_{i}")

        # -------- NLP DEBUG INFO --------
        nlp_meta = st.session_state.get("last_nlp_meta", {})
        if show_nlp_debug and nlp_meta:
            with st.expander("🔬 NLP Debug Info", expanded=False):
                st.markdown(f"**Original question:** `{st.session_state.chat_history[-1][0] if st.session_state.chat_history else ''}`")
                st.markdown(f"**Normalized question:** `{nlp_meta.get('normalized', 'N/A')}`")
                entities = nlp_meta.get("entities", [])
                if entities:
                    st.markdown(f"**Extracted entities/key terms:** `{', '.join(entities)}`")
                else:
                    st.markdown("**Extracted entities/key terms:** None detected")

        # -------- SOURCES --------
        sources = st.session_state.get("last_sources") or []
        if sources:
            st.markdown("### 🔍 Sources used")
            for idx, src in enumerate(sources, start=1):
                page = src.get("page", "?")
                snippet = src.get("snippet", "")
                with st.expander(f"Source {idx} • Page {page}"):
                    st.write(snippet)


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    img_col_anim, img_col_title = st.columns([1, 3])

    with img_col_anim:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
            height=180,
        )

    with img_col_title:
        st.markdown("## 🖼 Image Q&A")
        st.caption("Upload an image and ask questions about it.")

    st.divider()

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analyzing image..."):
                # Apply NLP normalization to image questions too
                if nlp_enabled:
                    normalized_q = normalize_query(question)
                    ans = answer_image_question(img, normalized_q)
                    if show_nlp_debug:
                        st.caption(f"🔬 Normalized: `{normalized_q}`")
                else:
                    ans = answer_image_question(img, question)

            st.success("Answer:")
            render_answer_with_copy(ans, key_suffix="img_answer")
