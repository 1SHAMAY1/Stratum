"""
Stratum — Chat with Your Documents
=====================================

Streamlit entry point for the Stratum RAPTOR RAG application.

Run with:
    streamlit run app.py
"""

import io
import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from src.pipeline import StratumPipeline

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stratum · Document Intelligence",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:   #0d0f14;
    --bg-card:      #1a1e2a;
    --bg-raised:    #20263a;
    --accent:       #6c63ff;
    --accent-light: #9d97ff;
    --accent-glow:  rgba(108,99,255,0.15);
    --text:         #e8eaf0;
    --muted:        #8a8fa8;
    --border:       #252a3a;
    --success:      #4caf82;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main area */
.main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 800px;
    margin: 0 auto;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff 20%, #c084fc 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}
.hero p { color: var(--muted); font-size: 0.95rem; }

/* Top bar */
.top-bar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin-bottom: 0.5rem;
}

/* Three-dot button override */
button[data-testid="baseButton-secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-size: 1.2rem !important;
    padding: 0.3rem 0.7rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    min-width: unset !important;
    width: auto !important;
}
button[data-testid="baseButton-secondary"]:hover {
    background: var(--bg-raised) !important;
    border-color: var(--accent) !important;
}

/* Popover panel */
[data-testid="stPopover"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    min-width: 380px !important;
}

/* Primary ingest button */
div[data-testid="stPopover"] .stButton > button,
.ingest-btn .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
div[data-testid="stPopover"] .stButton > button:hover,
.ingest-btn .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
textarea, input[type="text"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: var(--bg-raised) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-light) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Chat */
[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.85rem !important;
    animation: fadeUp 0.2s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}
[data-testid="stChatInput"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent-light) !important; font-weight:700 !important; }

hr { border-color: var(--border) !important; }

/* Status pill */
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}
.pill-ready   { background:#1a3a2a; color:#4caf82; border:1px solid #2d5a3d; }
.pill-waiting { background:#2a2520; color:#f0a04b; border:1px solid #5a4020; }

/* File tags */
.ftag {
    display: inline-block;
    background: var(--bg-raised); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.12rem 0.45rem;
    font-size: 0.71rem; color: var(--accent-light); margin: 0.1rem;
}

/* Model badge */
.model-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.25rem 0.65rem;
    font-size: 0.72rem; color: var(--muted); font-weight: 500;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".pdf", ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml", ".csv"}


def read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def load_uploaded_files(files) -> tuple[str, list[str]]:
    parts, names = [], []
    for f in files:
        ext = Path(f.name).suffix.lower()
        try:
            text = read_pdf(f.read()) if ext == ".pdf" else f.read().decode("utf-8", errors="ignore")
            if text.strip():
                parts.append(f"=== {f.name} ===\n{text}")
                names.append(f.name)
        except Exception as e:
            st.toast(f"Skipped {f.name}: {e}", icon="⚠️")
    return "\n\n".join(parts), names


def load_folder(path_str: str) -> tuple[str, list[str], list[str]]:
    folder = Path(path_str.strip())
    if not folder.exists() or not folder.is_dir():
        return "", [], []
    parts, loaded, skipped = [], [], []
    for fp in sorted(folder.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in SUPPORTED_EXT:
            skipped.append(fp.name)
            continue
        try:
            text = read_pdf(fp.read_bytes()) if fp.suffix.lower() == ".pdf" else fp.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                parts.append(f"=== {fp.relative_to(folder)} ===\n{text}")
                loaded.append(str(fp.relative_to(folder)))
        except Exception as e:
            skipped.append(f"{fp.name} ({e})")
    return "\n\n".join(parts), loaded, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = StratumPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []

# ──────────────────────────────────────────────────────────────────────────────
# Hero Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🪨 Stratum</h1>
    <p>RAPTOR-powered document intelligence &nbsp;·&nbsp; Chat with your knowledge</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Top action bar — model badge + ⋮ menu
# ──────────────────────────────────────────────────────────────────────────────
from config.settings import settings

bar_left, bar_mid, bar_right = st.columns([6, 1, 1])

with bar_left:
    # Status pill
    if st.session_state.stats:
        st.markdown(
            f'<span class="pill pill-ready">● Ready &nbsp;·&nbsp; '
            f'{st.session_state.stats["total_nodes"]} nodes &nbsp;·&nbsp; '
            f'{len(st.session_state.loaded_files)} file(s)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="pill pill-waiting">○ No documents loaded</span>', unsafe_allow_html=True)

with bar_mid:
    st.markdown(
        f'<div class="model-badge">🤖 {settings.LLM_MODEL}</div>',
        unsafe_allow_html=True,
    )

with bar_right:
    # ── ⋮ Three-dot popover menu ──────────────────────────────────────────
    with st.popover("⋮", use_container_width=False):
        st.markdown("#### 📂 Load Documents")
        tab_up, tab_folder, tab_paste = st.tabs(["Upload Files", "Folder Path", "Paste Text"])

        raw_text = ""
        file_names = []

        # Tab 1 — Upload
        with tab_up:
            st.caption("Drag & drop PDFs, .txt, .md, .py, .json and more")
            uploaded = st.file_uploader(
                "files",
                type=list(ext.lstrip(".") for ext in SUPPORTED_EXT),
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="uploader",
            )
            if uploaded:
                raw_text, file_names = load_uploaded_files(uploaded)

        # Tab 2 — Folder path
        with tab_folder:
            st.caption("Enter a full folder path — all supported files load recursively")
            folder_input = st.text_input(
                "path",
                placeholder=r"e.g.  E:\Resume\AIProjects",
                label_visibility="collapsed",
                key="folder_input",
            )
            if folder_input:
                p = Path(folder_input.strip())
                if not p.exists():
                    st.error("Path not found.")
                elif not p.is_dir():
                    st.error("That's a file, not a folder.")
                else:
                    raw_text, file_names, skipped = load_folder(folder_input)
                    if file_names:
                        st.success(f"Found {len(file_names)} file(s)")
                    if skipped:
                        st.caption(f"Skipped {len(skipped)} unsupported file(s)")

        # Tab 3 — Paste
        with tab_paste:
            st.caption("Paste raw text, code, or documentation")
            pasted = st.text_area(
                "text",
                placeholder="Paste content here...",
                height=180,
                label_visibility="collapsed",
                key="pasted",
            )
            if pasted.strip():
                raw_text = pasted
                file_names = ["(pasted text)"]

        # Preview
        if file_names:
            tags = "".join(f'<span class="ftag">{f}</span>' for f in file_names[:10])
            if len(file_names) > 10:
                tags += f'<span class="ftag">+{len(file_names)-10} more</span>'
            st.markdown(tags, unsafe_allow_html=True)
            st.markdown("")

        # Ingest button
        if st.button("⚡ Build Knowledge Tree", key="ingest_btn", type="primary"):
            if raw_text.strip():
                with st.spinner("Building RAPTOR layers…"):
                    try:
                        stats = st.session_state.pipeline.ingest_raw_documents(raw_text)
                        st.session_state.stats = stats
                        st.session_state.loaded_files = file_names
                        st.session_state.messages = []
                        st.success(f"Done! {stats['total_nodes']} nodes built.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                st.warning("No content selected yet.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Chat area
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.pipeline.is_ingested:
        st.info(
            "**Click the ⋮ button** (top right) to load your documents.\n\n"
            "- 📎 **Upload Files** — drag & drop PDFs, code files, text files\n"
            "- 📁 **Folder Path** — point to any folder on your drive\n"
            "- 📝 **Paste Text** — paste content directly\n\n"
            "Then click **Build Knowledge Tree** and start chatting!"
        )
    else:
        st.info("Knowledge tree ready — ask anything below!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🪨"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents…"):
    if not st.session_state.pipeline.is_ingested:
        st.warning("Click ⋮ to load documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🪨"):
            with st.spinner("Searching knowledge layers…"):
                try:
                    answer = st.session_state.pipeline.execute_query(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"Query failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
