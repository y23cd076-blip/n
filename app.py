"""
Local Streamlit Task Manager with SQLite.
Run with: streamlit run app.py
"""
import streamlit as st
import sqlite3
import os

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Task Manager",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Relative path: creates local_store.db in the directory you run streamlit from
DB_PATH = "local_store.db"

# -------------------- DATABASE --------------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def add_task(task_name: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO tasks (task_name, status) VALUES (?, 'Pending')",
        (task_name.strip(),),
    )
    conn.commit()
    conn.close()


def get_all_tasks():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, task_name, status, date_added FROM tasks ORDER BY date_added DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_status(task_id: int, status: str) -> None:
    conn = get_connection()
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()
    conn.close()


def delete_task(task_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


# -------------------- INIT --------------------
init_db()
tasks = get_all_tasks()
total = len(tasks)
completed = sum(1 for t in tasks if t["status"] == "Completed")
pending = total - completed

# -------------------- UI: METRICS DASHBOARD --------------------
st.title("✓ Task Manager")
st.caption("Local SQLite • Add, complete, or remove tasks below.")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Tasks", total)
with m2:
    st.metric("Completed", completed)
with m3:
    st.metric("Pending", pending)

st.divider()

# -------------------- CREATE --------------------
st.subheader("Add a task")
col_input, col_btn = st.columns([4, 1])
with col_input:
    new_task = st.text_input(
        "Task name",
        placeholder="e.g. Review report, Call client…",
        label_visibility="collapsed",
        key="new_task_input",
    )
with col_btn:
    add_clicked = st.button("Add task", type="primary", use_container_width=True)

if add_clicked and new_task and new_task.strip():
    add_task(new_task)
    st.rerun()
elif add_clicked and not (new_task and new_task.strip()):
    st.warning("Enter a task name.")

st.divider()

# -------------------- READ: DATAFRAME --------------------
st.subheader("All tasks")
if tasks:
    df_data = [
        {
            "ID": t["id"],
            "Task": t["task_name"],
            "Status": t["status"],
            "Date added": t["date_added"],
        }
        for t in tasks
    ]
    st.dataframe(
        df_data,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No tasks yet. Add one above.")

st.divider()

# -------------------- UPDATE / DELETE --------------------
st.subheader("Update or delete")
if tasks:
    options = [f"{t['id']} — {t['task_name']} ({t['status']})" for t in tasks]
    choice = st.selectbox(
        "Choose a task",
        options=options,
        label_visibility="collapsed",
        key="task_choice",
    )
    if choice:
        task_id = int(choice.split(" — ")[0])
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        with c1:
            if st.button("✓ Completed", key="complete_btn"):
                update_status(task_id, "Completed")
                st.rerun()
        with c2:
            if st.button("↩ Pending", key="pending_btn"):
                update_status(task_id, "Pending")
                st.rerun()
        with c3:
            if st.button("Delete", key="delete_btn"):
                delete_task(task_id)
                st.rerun()
else:
    st.caption("Add tasks to enable update and delete.")
