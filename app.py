import streamlit as st
import time
import os
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "SARVAM_API_KEY" in st.secrets:
    os.environ["SARVAM_API_KEY"] = st.secrets["SARVAM_API_KEY"]

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables: Dark Orange, Black & White Theme ── */
:root {
    --bg: #000000;
    --surface: #0a0a0a;
    --surface-2: #141414;
    --border: #1f1f1f;
    --accent: #ff7000;
    --accent-glow: rgba(255, 112, 0, 0.15);
    --accent-dim: #cc5a00;
    --text: #ffffff;
    --text-muted: #a0a0a0;
    --success: #ffffff;
    --warning: #ff7000;
    --danger: #ff3333;
}

/* ── Global Reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Minimalist mesh grid background background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(255, 112, 0, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 112, 0, 0.015) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 70%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px var(--accent-glow);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--text));
}

.card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
    font-weight: 400;
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(255, 112, 0, 0.15); color: var(--accent); border: 1px solid rgba(255, 112, 0, 0.3); }
.badge-cyan   { background: rgba(255, 255, 255, 0.1); color: var(--text);    border: 1px solid rgba(255, 255, 255, 0.2); }
.badge-green  { background: rgba(255, 112, 0, 0.1); color: var(--accent); border: 1px solid rgba(255, 112, 0, 0.2); }

/* ── Input Components ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* Primary Form Control Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background-color: var(--accent-dim) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(255, 112, 0, 0.3) !important;
}

/* Secondary Actions styling overrides */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* ── Progress / Pipeline Status UI ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
    font-weight: 500;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--text); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat Canvas ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent); }
.bot-label   { color: var(--text); }

.user-bubble { background: rgba(255, 112, 0, 0.1); border: 1px solid rgba(255, 112, 0, 0.2); align-self: flex-end; }
.bot-bubble  { background: var(--surface-2);  border: 1px solid var(--border);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript Text Area ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.85rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* Fix basic Markdown Container text flows */
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; font-weight: 500; }

/* Tabs Accents Adjustment */
button[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
}
button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "uploaded_file_path": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar Input Controls ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Analyzer Engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input Source</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    uploaded_file = st.file_uploader("Or upload a file", type=["mp4", "mp3", "wav", "m4a", "flac", "ogg", "avi", "mkv", "mov"])
    if uploaded_file is not None:
        save_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(save_dir, exist_ok=True)
        temp_path = os.path.join(save_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_path
        st.caption(f"Uploaded: `{uploaded_file.name}`")
    else:
        if st.session_state.get("uploaded_file_path") and not os.path.exists(st.session_state["uploaded_file_path"]):
            st.session_state.uploaded_file_path = None

    language = st.selectbox("Language Dialect", ["english", "hinglish"], index=0)
    # Inside the sidebar section of app.py
    st.markdown('<span class="badge badge-purple">Compute Engine</span>', unsafe_allow_html=True)
    # compute_mode = st.radio("Processing Environment", ["Cloud API (Fast)", "Local CPU (Private)"])
    compute_mode_raw = st.radio("Processing Environment", ["Cloud API (Fast)", "Local CPU (Private)"])
    compute_mode = "api" if "cloud" in compute_mode_raw.lower() else "local"
    show_local_info = compute_mode == "local"

    

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Workspace Area ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

if show_local_info:
    st.markdown(f"""
    <div class="card" style="border-color:var(--accent)">
        <div class="card-title">💻 Local CPU Mode</div>
        <div class="card-content">
            You can run this app locally on your own machine using the Local CPU (Private) option.
            No cloud compute engine required — everything runs on your device.
            <br><br>
            📦 To download and set up the app locally, visit the GitHub repository:<br>
            <a href="https://github.com/ineverydomain/ai_video_analyzer" target="_blank" style="color:var(--accent);font-weight:600;text-decoration:none">
                https://github.com/ineverydomain/ai_video_analyzer
            </a>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Run Pipeline Processing ─────────────────────────────────────────────────────
if run_btn:
    effective_source = st.session_state.get("uploaded_file_path") or source.strip()
    if not effective_source:
        st.error("Please enter a YouTube URL, file path, or upload a file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            chunks = process_input(effective_source)
            update_step("audio", "done")

            update_step("transcript", "active")
            # CHANGED: Pass compute mode down to choose between Groq API or Local Whisper
            transcript = transcribe_all(chunks, language, mode=compute_mode.lower())
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript, mode=compute_mode)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript, mode=compute_mode)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript, mode=compute_mode)
            decisions     = extract_key_decisions(transcript, mode=compute_mode)
            questions     = extract_questions(transcript, mode=compute_mode)
            update_step("extract", "done")

            update_step("rag", "active")
            # CHANGED: Pass compute mode down to choose between HF Inference API or Local Embeddings
            rag_chain = build_rag_chain(transcript, mode=compute_mode.lower())
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Display Analysis Results Modules ────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner card component
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row layout splitting summary insights + transcript collapse area
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript Layout Mapping", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Metric feature extraction segments grid
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Strategic Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions Tracked</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.write("---")

    # ── Context Driven RAG Chat Interface ───────────────────────────────────────
    st.markdown('<div style="font-family:\'Inter\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Knowledge Base</div>', unsafe_allow_html=True)

    # Chat history viewport frame render
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask target questions regarding the computed material context natively.</div>
        </div>""", unsafe_allow_html=True)

    # Query Input panel
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Scanning database embeddings…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Portfolio Empty Splash State Banner
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse Material
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Provide a valid YouTube link location parameters or local system files inside the left configurations dashboard selector to begin data parsing loops.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription Layers</span>
            <span class="badge badge-cyan">Summarisation Mapping</span>
            <span class="badge badge-green">RAG Vector Engine</span>
        </div>
    </div>""", unsafe_allow_html=True)