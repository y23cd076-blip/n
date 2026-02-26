import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib, re
from typing import Any, Dict, Optional, List, Tuple
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

# RDF imports
import spacy
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

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
    return sess["user"] if sess else None


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def sign_up(username: str, password: str) -> Optional[str]:
    if len(password) < 6:
        return "Password must be at least 6 characters."
    resp = _rest_request(
        "GET",
        f"/rest/v1/{USER_PROFILES_TABLE}?select=username&username=eq.{username}",
    )
    if resp.status_code >= 400:
        return resp.text
    if resp.json():
        return "Username already exists."
    payload = {"username": username, "password_hash": _hash_password(password)}
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


@st.cache_resource
def load_spacy():
    """Load spaCy model for RDF triple extraction."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


# -------------------- RDF HELPERS --------------------
EX = Namespace("http://slidesense.app/entity/")
PRED = Namespace("http://slidesense.app/predicate/")


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug for use in URIs."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text[:60].strip("_") or "entity"


def extract_rdf_triples(text: str, max_triples: int = 200) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, predicate, object) triples from text using spaCy
    dependency parsing.

    Strategy:
      - For each verb token, find its nominal subject (nsubj/nsubjpass) and
        object (dobj, attr, prep+pobj).
      - Noun chunks are used to expand single tokens into full phrases.
    """
    nlp = load_spacy()
    triples: List[Tuple[str, str, str]] = []

    # Build a chunk lookup: token index -> chunk text
    chunk_map: Dict[int, str] = {}

    # Process in segments to avoid spaCy's max_length limits
    segment_size = 100_000
    segments = [text[i : i + segment_size] for i in range(0, len(text), segment_size)]

    for segment in segments:
        doc = nlp(segment)

        # Rebuild chunk map for this segment
        chunk_map = {chunk.root.i: chunk.text for chunk in doc.noun_chunks}

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            # Find subject
            subjects = [
                child for child in token.children
                if child.dep_ in ("nsubj", "nsubjpass")
            ]
            # Find objects
            objects = [
                child for child in token.children
                if child.dep_ in ("dobj", "attr", "acomp")
            ]
            # Also pick up prepositional objects (verb → prep → pobj)
            for prep_child in token.children:
                if prep_child.dep_ == "prep":
                    for pobj in prep_child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(pobj)

            predicate = token.lemma_.lower()

            for subj in subjects:
                subj_text = chunk_map.get(subj.i, subj.text)
                for obj in objects:
                    obj_text = chunk_map.get(obj.i, obj.text)
                    triples.append((subj_text.strip(), predicate, obj_text.strip()))

        if len(triples) >= max_triples:
            break

    # Deduplicate while preserving order
    seen = set()
    unique: List[Tuple[str, str, str]] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:max_triples]


def build_rdf_graph(triples: List[Tuple[str, str, str]]) -> Graph:
    """
    Convert extracted (subject, predicate, object) triples into an rdflib Graph.
    Subjects and objects become URI resources; predicates are namespaced URIs.
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("pred", PRED)

    for subj_text, pred_text, obj_text in triples:
        subj_uri = EX[_slugify(subj_text)]
        pred_uri = PRED[_slugify(pred_text)]
        obj_uri  = EX[_slugify(obj_text)]

        g.add((subj_uri, RDF.type, RDFS.Resource))
        g.add((obj_uri,  RDF.type, RDFS.Resource))

        # Add human-readable labels
        g.add((subj_uri, RDFS.label, Literal(subj_text, datatype=XSD.string)))
        g.add((obj_uri,  RDFS.label, Literal(obj_text,  datatype=XSD.string)))

        # Add the relationship triple
        g.add((subj_uri, pred_uri, obj_uri))
        # Label the predicate too
        g.add((pred_uri, RDFS.label, Literal(pred_text, datatype=XSD.string)))

    return g


def query_rdf_graph(g: Graph, sparql: str) -> List[Dict[str, str]]:
    """Run a SPARQL SELECT query on the graph and return rows as dicts."""
    try:
        results = g.query(sparql)
        rows = []
        for row in results:
            rows.append({str(var): str(val) for var, val in zip(results.vars, row)})
        return rows
    except Exception as exc:
        return [{"error": str(exc)}]


# -------------------- SESSION DEFAULTS --------------------
defaults = {
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
    "rdf_triples": [],
    "rdf_graph": None,
    "rdf_pdf_id": None,
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
    label = f"Logged in as {get_display_name(user)}"
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

mode = st.sidebar.radio(
    "Mode",
    ["📘 PDF Analyzer", "🖼 Image Q&A", "🕸 Knowledge Graph (RDF)"],
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

        user_q = st.chat_input("Ask a question about this PDF")

        if user_q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(user_q, k=5)
            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")
            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": user_q})
            answer = res.get("output_text", "") if isinstance(res, dict) else res
            st.session_state.chat_history.append((user_q, answer))
            save_chat_to_db(user_q, answer)
            st.session_state["last_sources"] = [
                {
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
                }
                for d in docs
            ]

        st.markdown("## 💬 Conversation")
        chat_container = st.container()
        with chat_container:
            for uq, ua in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(uq)
                with st.chat_message("assistant"):
                    render_answer_with_copy(ua, key_suffix=f"pdf_{uq}")

        sources = st.session_state.get("last_sources") or []
        if sources:
            st.markdown("### 🔍 Sources used")
            for idx, src in enumerate(sources, start=1):
                with st.expander(f"Source {idx} • Page {src.get('page', '?')}"):
                    st.write(src.get("snippet", ""))


# ==================== IMAGE Q&A ====================
elif mode == "🖼 Image Q&A":
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
                ans = answer_image_question(img, question)
            st.success("Answer:")
            render_answer_with_copy(ans, key_suffix="img_answer")


# ==================== KNOWLEDGE GRAPH (RDF) ====================
elif mode == "🕸 Knowledge Graph (RDF)":
    rdf_col_anim, rdf_col_title = st.columns([1, 3])
    with rdf_col_anim:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
            height=180,
        )
    with rdf_col_title:
        st.markdown("## 🕸 Knowledge Graph (RDF)")
        st.caption(
            "Upload a PDF to automatically extract structured knowledge as "
            "RDF triples (Subject → Predicate → Object) using NLP dependency parsing. "
            "Query the graph with SPARQL or export it."
        )

    st.divider()

    # ---- Explainer ----
    with st.expander("ℹ️ What is RDF?", expanded=False):
        st.markdown(
            """
**Resource Description Framework (RDF)** is a W3C standard for representing knowledge as
a graph of triples:

| Component | Meaning | Example |
|-----------|---------|---------|
| **Subject** | The entity being described | `the_model` |
| **Predicate** | The relationship / property | `achieve` |
| **Object** | The value or related entity | `high_accuracy` |

SlideSense uses **spaCy dependency parsing** to automatically extract these triples from
your PDF text, then stores them in an **rdflib** graph.  You can:
- Browse all extracted triples
- Run **SPARQL** queries to explore relationships
- Export the graph in **Turtle** or **JSON-LD** format
"""
        )

    # ---- PDF Upload ----
    rdf_pdf = st.file_uploader("Upload PDF for RDF extraction", type="pdf", key="rdf_uploader")

    if rdf_pdf:
        rdf_pdf_id = f"{rdf_pdf.name}_{rdf_pdf.size}"

        if st.session_state.rdf_pdf_id != rdf_pdf_id:
            st.session_state.rdf_pdf_id = rdf_pdf_id
            st.session_state.rdf_triples = []
            st.session_state.rdf_graph = None

        if not st.session_state.rdf_triples:
            with st.spinner("Extracting RDF triples with spaCy NLP… (this may take a moment)"):
                reader = PdfReader(rdf_pdf)
                full_text = ""
                for pdf_page in reader.pages:
                    extracted = pdf_page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"

                if not full_text.strip():
                    st.error("No readable text found in PDF.")
                    st.stop()

                triples = extract_rdf_triples(full_text, max_triples=300)
                if not triples:
                    st.warning("No triples could be extracted from this document.")
                    st.stop()

                st.session_state.rdf_triples = triples
                st.session_state.rdf_graph = build_rdf_graph(triples)

            st.success(
                f"✅ Extracted **{len(st.session_state.rdf_triples)}** RDF triples "
                f"from *{rdf_pdf.name}*."
            )

        triples = st.session_state.rdf_triples
        g: Graph = st.session_state.rdf_graph

        # ---- Stats ----
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Triples", len(triples))
        col_b.metric("Unique Subjects", len({t[0] for t in triples}))
        col_c.metric("Unique Predicates", len({t[1] for t in triples}))

        st.divider()

        # ---- Tabs ----
        tab_browse, tab_search, tab_sparql, tab_export = st.tabs(
            ["📋 Browse Triples", "🔎 Search", "⚡ SPARQL Query", "📤 Export"]
        )

        # --- Browse ---
        with tab_browse:
            st.markdown("### All Extracted Triples")
            page_size = 50
            total_pages = max(1, (len(triples) - 1) // page_size + 1)
            page_num = st.number_input(
                f"Page (1–{total_pages})", min_value=1, max_value=total_pages, value=1, step=1
            )
            start = (page_num - 1) * page_size
            end = start + page_size
            page_triples = triples[start:end]

            table_data = [
                {"#": start + i + 1, "Subject": s, "Predicate": p, "Object": o}
                for i, (s, p, o) in enumerate(page_triples)
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

        # --- Search ---
        with tab_search:
            st.markdown("### Search Triples")
            search_term = st.text_input(
                "Filter by keyword (subject, predicate, or object)",
                placeholder="e.g. model, achieve, accuracy",
            )
            if search_term:
                term_lower = search_term.lower()
                filtered = [
                    t for t in triples
                    if term_lower in t[0].lower()
                    or term_lower in t[1].lower()
                    or term_lower in t[2].lower()
                ]
                st.write(f"Found **{len(filtered)}** matching triples:")
                if filtered:
                    st.dataframe(
                        [{"Subject": s, "Predicate": p, "Object": o} for s, p, o in filtered],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No triples match your search.")

        # --- SPARQL ---
        with tab_sparql:
            st.markdown("### SPARQL Query")
            st.caption(
                "Query the RDF graph using standard SPARQL SELECT syntax. "
                "Namespaces available: `ex:` (entities), `pred:` (predicates), `rdfs:label`."
            )

            default_sparql = """\
PREFIX ex:   <http://slidesense.app/entity/>
PREFIX pred: <http://slidesense.app/predicate/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?subject_label ?predicate_label ?object_label
WHERE {
  ?s ?p ?o .
  ?s rdfs:label ?subject_label .
  ?o rdfs:label ?object_label .
  ?p rdfs:label ?predicate_label .
}
LIMIT 25"""

            sparql_query = st.text_area("SPARQL Query", value=default_sparql, height=200)

            if st.button("▶ Run Query"):
                with st.spinner("Running SPARQL query…"):
                    rows = query_rdf_graph(g, sparql_query)

                if rows and "error" in rows[0]:
                    st.error(f"SPARQL error: {rows[0]['error']}")
                elif rows:
                    st.success(f"Returned **{len(rows)}** rows.")
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("Query returned no results.")

        # --- Export ---
        with tab_export:
            st.markdown("### Export Knowledge Graph")
            col_ttl, col_jsonld = st.columns(2)

            with col_ttl:
                st.markdown("**Turtle (.ttl)**")
                turtle_data = g.serialize(format="turtle")
                st.download_button(
                    label="⬇️ Download Turtle",
                    data=turtle_data,
                    file_name="knowledge_graph.ttl",
                    mime="text/turtle",
                )

            with col_jsonld:
                st.markdown("**JSON-LD (.jsonld)**")
                jsonld_data = g.serialize(format="json-ld", indent=2)
                st.download_button(
                    label="⬇️ Download JSON-LD",
                    data=jsonld_data,
                    file_name="knowledge_graph.jsonld",
                    mime="application/ld+json",
                )

            st.markdown("**Raw Triples (.csv)**")
            import csv, io
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["subject", "predicate", "object"])
            writer.writerows(triples)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_buf.getvalue(),
                file_name="rdf_triples.csv",
                mime="text/csv",
            )

    else:
        st.info("👆 Upload a PDF above to begin RDF knowledge extraction.")
