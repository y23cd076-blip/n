# ==============================
# SLIDESENSE - FULL APP
# Single File Production Version
# ==============================

import streamlit as st
import requests, os, hashlib, uuid
from datetime import datetime, timedelta

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:
    ChatGoogleGenerativeAI = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL", "") or st.secrets.get("SUPABASE_URL", "")
SUPABASE_URL = SUPABASE_URL.rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_ANON_KEY", "")

# Validate critical configuration early to avoid hard-to-debug runtime errors
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("`GOOGLE_API_KEY` is not set. Please add it to Streamlit secrets or environment variables.")
    st.stop()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Supabase configuration is missing. Please set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in secrets or env vars.")
    st.stop()

CHAT_TABLE = "chat_sessions"
MSG_TABLE = "chat_messages"
USER_TABLE = "user_profiles"

# ---------------- UTILS ----------------
def headers():
    return {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}

def rest(method, path, **kwargs):
    return requests.request(method, f"{SUPABASE_URL}{path}", headers=headers(), timeout=20, **kwargs)

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# ---------------- MODELS ----------------
@st.cache_resource
def load_llm():
    if ChatGoogleGenerativeAI is None:
        st.error(
            "The `langchain-google-genai` package is not available. "
            "Make sure it is listed in `requirements.txt` and successfully installed."
        )
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = load_llm()

# ---------------- SESSION ----------------
defaults = {
    "user": None,
    "guest": False,
    "current_chat": None,
    "vector_db": None,
    "mode": "chat"
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ---------------- AUTH ----------------
def signup(u,p):
    r = rest("GET", f"/rest/v1/{USER_TABLE}?username=eq.{u}")
    if r.json(): return "User exists"
    rest("POST", f"/rest/v1/{USER_TABLE}", json={"username":u,"password_hash":hash_pw(p)})
    return None

def login(u,p):
    r = rest("GET", f"/rest/v1/{USER_TABLE}?username=eq.{u}")
    data = r.json()
    if not data: return "Invalid login"
    if data[0]["password_hash"]!=hash_pw(p): return "Invalid login"
    st.session_state.user = data[0]["username"]
    return None

# ---------------- DB ----------------
def create_chat(username, mode):
    cid = str(uuid.uuid4())
    rest("POST", f"/rest/v1/{CHAT_TABLE}", json={
        "id":cid,"username":username,"mode":mode,
        "title":"New Chat","created_at":datetime.utcnow().isoformat()
    })
    return cid

def save_msg(cid, role, content):
    rest("POST", f"/rest/v1/{MSG_TABLE}", json={
        "chat_id":cid,"role":role,"content":content,
        "created_at":datetime.utcnow().isoformat()
    })
    if role=="user":
        r = rest("GET", f"/rest/v1/{MSG_TABLE}?chat_id=eq.{cid}&role=eq.user")
        if len(r.json())==1:
            title = content[:45]+"..." if len(content)>45 else content
            rest("PATCH", f"/rest/v1/{CHAT_TABLE}?id=eq.{cid}", json={"title":title})

def load_chats(user,mode):
    r = rest("GET", f"/rest/v1/{CHAT_TABLE}?username=eq.{user}&mode=eq.{mode}&order=created_at.desc")
    return r.json()

def load_msgs(cid):
    r = rest("GET", f"/rest/v1/{MSG_TABLE}?chat_id=eq.{cid}&order=created_at.asc")
    return r.json()

def delete_chat(cid):
    rest("DELETE", f"/rest/v1/{MSG_TABLE}?chat_id=eq.{cid}")
    rest("DELETE", f"/rest/v1/{CHAT_TABLE}?id=eq.{cid}")

# ---------------- GROUPING ----------------
def group(chats):
    today = datetime.utcnow().date()
    g={"Today":[], "Yesterday":[], "Last 7 Days":[], "Older":[]}
    for c in chats:
        d = datetime.fromisoformat(c["created_at"]).date()
        if d==today: g["Today"].append(c)
        elif d==today-timedelta(days=1): g["Yesterday"].append(c)
        elif (today-d).days<=7: g["Last 7 Days"].append(c)
        else: g["Older"].append(c)
    return g

# ---------------- CSS ----------------
st.markdown("""
<style>
.chat-item{padding:10px;border-radius:10px;margin:6px 0;
background:rgba(255,255,255,0.05);transition:.25s;cursor:pointer;}
.chat-item:hover{background:rgba(0,255,255,.15);transform:translateX(6px);}
.chat-active{background:linear-gradient(90deg,#00f5ff33,#00ff9c22);
border-left:3px solid #00f5ff;}
.group-title{font-size:11px;opacity:.6;margin:10px 0 4px 4px;}
</style>
""", unsafe_allow_html=True)

# ---------------- AUTH UI ----------------
if not st.session_state.user and not st.session_state.guest:
    st.title("📘 SlideSense")
    tab1,tab2 = st.tabs(["Login","Signup"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password",type="password")
        if st.button("Login"):
            err = login(u,p)
            if err: st.error(err)
            else: st.rerun()
    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password",type="password")
        if st.button("Signup"):
            err = signup(u,p)
            if err: st.error(err)
            else: st.success("Account created")

    if st.button("Continue as Guest"):
        st.session_state.guest=True
        st.session_state.user="guest"
        st.rerun()
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💬 Chats")
    if st.button("➕ New Chat"):
        st.session_state.current_chat = create_chat(st.session_state.user, st.session_state.mode)
        st.rerun()

    chats = load_chats(st.session_state.user, st.session_state.mode)
    grouped = group(chats)

    for gname,items in grouped.items():
        if not items: continue
        st.markdown(f"<div class='group-title'>{gname}</div>", unsafe_allow_html=True)
        for c in items:
            active = "chat-active" if c["id"]==st.session_state.current_chat else ""
            if st.markdown(f"<div class='chat-item {active}'>{c['title']}</div>", unsafe_allow_html=True):
                pass
            if st.button("Open", key="o"+c["id"]):
                st.session_state.current_chat=c["id"]; st.rerun()
            if st.button("🗑", key="d"+c["id"]):
                delete_chat(c["id"])
                if st.session_state.current_chat==c["id"]:
                    st.session_state.current_chat=None
                st.rerun()

# ---------------- MAIN ----------------
st.title("🤖 SlideSense AI")

if not st.session_state.current_chat:
    st.info("Start a new chat from sidebar")
    st.stop()

msgs = load_msgs(st.session_state.current_chat)
for m in msgs:
    if m["role"]=="user":
        st.markdown(f"**You:** {m['content']}")
    else:
        st.markdown(f"**AI:** {m['content']}")

q = st.text_input("Ask something...")
if st.button("Send") and q:
    save_msg(st.session_state.current_chat,"user",q)
    with st.spinner("Thinking..."):
        ans = llm.invoke(q).content
    save_msg(st.session_state.current_chat,"assistant",ans)
    st.rerun()
