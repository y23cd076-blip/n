import streamlit as st
import requests
import uuid
from datetime import datetime

# =============================
# CONFIG
# =============================
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_KEY = "YOUR_ANON_PUBLIC_KEY"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

USER_TABLE = "user_profiles"
CHAT_TABLE = "chat_sessions"
MSG_TABLE = "chat_messages"

# =============================
# LOW LEVEL REST CLIENT
# =============================
def sb_rest(method, table, json=None, params=None, silent=False):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.request(method, url, headers=HEADERS, json=json, params=params)

    if r.status_code >= 400:
        print("SUPABASE ERROR:", r.status_code, r.text)
        if silent:
            return None
        raise RuntimeError(f"Supabase error {r.status_code}: {r.text}")

    if r.text:
        try:
            return r.json()
        except:
            return r.text
    return None

# =============================
# SELF HEALING DB LAYER
# =============================
def ensure_user(uid, uname):
    # Check if exists
    res = sb_rest("GET", USER_TABLE, params={"id": f"eq.{uid}"}, silent=True)
    if res:
        return True

    # Create if missing
    try:
        sb_rest("POST", USER_TABLE, json={
            "id": uid,
            "username": uname,
            "password_hash": "guest",
            "created_at": datetime.utcnow().isoformat()
        })
        return True
    except:
        print("Auto-fix: creating user failed, retrying without hash...")
        # retry fallback
        sb_rest("POST", USER_TABLE, json={
            "id": uid,
            "username": uname,
            "created_at": datetime.utcnow().isoformat()
        })
        return True

def create_new_chat(uid, mode):
    cid = str(uuid.uuid4())

    # ensure user exists before FK insert
    ensure_user(uid, st.session_state.username)

    try:
        sb_rest("POST", CHAT_TABLE, json={
            "id": cid,
            "user_id": uid,
            "mode": mode,
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat()
        })
        return cid
    except:
        print("Chat insert failed → retry without FK binding")
        # emergency fallback (prevents crash)
        sb_rest("POST", CHAT_TABLE, json={
            "id": cid,
            "mode": mode,
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat()
        })
        return cid

def save_message(chat_id, role, content):
    try:
        sb_rest("POST", MSG_TABLE, json={
            "id": str(uuid.uuid4()),
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        }, silent=True)
    except:
        print("Message insert failed (ignored)")

# =============================
# AUTH SYSTEM
# =============================
def login_as_guest():
    uid = str(uuid.uuid4())
    uname = f"guest_{uid[:6]}"

    ensure_user(uid, uname)

    st.session_state.user_id = uid
    st.session_state.username = uname
    st.session_state.auth = True
    st.session_state.mode = "CHAT"
    st.session_state.chat_id = create_new_chat(uid, "CHAT")

# =============================
# UI
# =============================
st.set_page_config(page_title="AI Chat", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 Login")
    if st.button("Login as Guest"):
        login_as_guest()
        st.rerun()

else:
    st.sidebar.title("👤 User")
    st.sidebar.write(st.session_state.username)

    if st.sidebar.button("➕ New Chat"):
        st.session_state.chat_id = create_new_chat(
            st.session_state.user_id,
            st.session_state.mode
        )
        st.session_state.messages = []

    st.title("💬 Chat App")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat display
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    # Input
    prompt = st.chat_input("Type message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.chat_id, "user", prompt)

        reply = f"Echo: {prompt}"  # replace with AI model
        st.session_state.messages.append({"role": "assistant", "content": reply})
        save_message(st.session_state.chat_id, "assistant", reply)

        st.rerun()
