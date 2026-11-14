"""
web_app.py
Streamlit chat interface for the AAST Smart Academic Advisor.

Run:  streamlit run web_app.py
"""
import html
import re

import streamlit as st

from rag import AcademicRAG

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="AAST Smart Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMS_BY_YEAR = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}

# ----------------------------------
# STYLE (works in light and dark mode: only neutral transparent colours are
# used for cards/borders, and text colour is inherited from the theme)
# ----------------------------------
st.markdown("""
<style>
.block-container { max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem; }

/* sidebar: dark panel, light text. Widgets keep their own colours. */
[data-testid="stSidebar"] { background: #14243a; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown *,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #ffffff !important; }

.main-title { font-size: 36px; font-weight: 800; margin-bottom: 4px; }
.subtitle { opacity: 0.7; font-size: 16px; margin-bottom: 22px; }

.welcome-card {
    background: linear-gradient(120deg, #172b46, #28496b);
    padding: 26px; border-radius: 18px; margin: 12px 0 22px 0;
    box-shadow: 0 8px 25px rgba(20, 36, 58, 0.12);
}
.welcome-card h2 { color: #ffffff !important; margin: 0; font-size: 25px; }
.welcome-card p { color: #e2e8f0 !important; margin: 8px 0 0 0; }

.info-card {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 18px; border-radius: 14px; min-height: 110px;
}
.info-card h3 { color: #c69b45 !important; font-size: 21px; margin: 0; }
.info-card p { opacity: 0.75; font-size: 13px; margin-top: 8px; }

.section-title { font-size: 21px; font-weight: 700; margin: 24px 0 12px 0; }

.stButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stChatMessage"] { border-radius: 14px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD RAG (once)
# ----------------------------------
@st.cache_resource(show_spinner="Loading the academic knowledge base...")
def load_rag():
    return AcademicRAG()


try:
    rag = load_rag()
except Exception as e:  # missing chroma_db, bad .env, ...
    st.error("The assistant could not start. Make sure you ran "
             "`python ingest.py` and `python index.py`, and that `.env` is set up.")
    st.caption(f"Details: {e}")
    st.stop()


# ----------------------------------
# HELPERS
# ----------------------------------
def personalize(question: str, year, term) -> str:
    """Turn 'my term' / 'this year' into an explicit 'Term 3' / 'Year 2'.

    The question is NOT padded with extra text, so retrieval only sees what the
    student actually asked (extra 'Year 2 / Term 3' words would pull in
    unrelated summaries and waste the 4 retrieval slots).
    """
    if term:
        question = re.sub(r"\b(?:my|this|current|selected)(?:\s+selected)?\s+term\b",
                          f"Term {term}", question, flags=re.I)
    if year:
        question = re.sub(r"\b(?:my|this|current|selected)(?:\s+selected)?\s+year\b",
                          f"Year {year}", question, flags=re.I)
    return question


def suggestions(year, term):
    """Ready-made questions that follow the student's profile."""
    if term:
        return [
            (f"📚 Courses in Term {term}", f"Which courses are offered in Term {term}?"),
            (f"🎯 Prerequisites in Term {term}",
             f"Show all courses in Term {term} with their prerequisites."),
            (f"📊 Credit hours in Term {term}",
             f"What is the total number of credit hours in Term {term}?"),
        ]
    if year:
        return [
            (f"📚 Courses in Year {year}", f"What courses are included in Year {year}?"),
            (f"🎯 Year {year} prerequisites",
             f"List all Year {year} courses with their codes, credit hours, and prerequisites."),
        ]
    return [
        ("📚 Courses in Term 1", "Which courses are offered in Term 1?"),
        ("🎯 Prerequisites example", "What are the prerequisites for Database Systems?"),
    ]


def source_label(meta: dict) -> str:
    kind = meta.get("doc_type")
    if kind == "course":
        return f"{meta.get('course_code', '')} – {meta.get('course_name', '')}".strip(" –")
    if kind == "term_summary":
        return f"Year {meta.get('year', '?')}, Term {meta.get('term', '?')} summary"
    if kind == "year_summary":
        return f"Year {meta.get('year', '?')} summary"
    return str(meta.get("doc_id") or meta.get("source") or "Academic document")


def show_sources(sources):
    if sources:
        with st.expander("📄 View academic sources"):
            for s in sources:
                st.write(f"• {s}")


# ----------------------------------
# SESSION STATE
# ----------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)


# ----------------------------------
# SIDEBAR - STUDENT PROFILE (all optional)
# ----------------------------------
with st.sidebar:
    st.markdown("## 🎓 AAST")
    st.markdown("### Smart Academic Advisor")
    st.caption("Your personal university assistant")
    st.divider()

    st.markdown("### 👤 Student Profile")
    student_name = st.text_input("Your Name (optional)", placeholder="Enter your name")

    year = st.selectbox(
        "Academic Year",
        [None, 1, 2, 3, 4],
        format_func=lambda y: "Select your year" if y is None else f"Year {y}",
    )

    term_options = [None] + (TERMS_BY_YEAR[year] if year else [])
    term = st.selectbox(
        "Academic Term",
        term_options,
        format_func=lambda t: "Not selected" if t is None else f"Term {t}",
        disabled=year is None,
        key=f"term_{year}",  # reset the term when the year changes
    )

    st.divider()
    st.markdown("### 📚 What can I help with?")
    st.markdown("- Course information\n- Credit hours\n- Prerequisites\n"
                "- Course progression\n- Academic terms")
    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption("AAST Academic Advisor | 2026")


# ----------------------------------
# HEADER (+ welcome and info cards only before the first question)
# ----------------------------------
st.markdown('<div class="main-title">AAST Smart Academic Advisor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent assistant for academic planning '
            'and university information.</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    welcome_name = html.escape(student_name.strip()) or "Student"  # escaped: user input
    st.markdown(f"""
    <div class="welcome-card">
        <h2>Welcome, {welcome_name}! 👋</h2>
        <p>Ready to explore your academic journey?
           Ask me anything about your university courses.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📖 Courses", "Explore course details and codes."),
        ("🎯 Prerequisites", "Understand what you need before taking a course."),
        ("📊 Credit Hours", "Find credit hours and academic term information."),
    ]
    for col, (title, text) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(f'<div class="info-card"><h3>{title}</h3><p>{text}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Start with a question</div>',
                unsafe_allow_html=True)
    items = suggestions(year, term)
    for col, (label, q) in zip(st.columns(len(items)), items):
        with col:
            if st.button(label, use_container_width=True, key=f"sugg_{label}"):
                st.session_state.pending_question = q
                st.rerun()


# ----------------------------------
# CHAT HISTORY
# ----------------------------------
st.markdown('<div class="section-title">💬 Academic Assistant</div>',
            unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("error"):
            st.warning("The AI service is temporarily unavailable. Please try again.")
        show_sources(m.get("sources"))


# ----------------------------------
# NEW QUESTION
# ----------------------------------
typed = st.chat_input("Ask your academic question here...")
question = st.session_state.pending_question or typed
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching academic documents..."):
            try:
                result = rag.ask(personalize(question, year, term))
                answer, ok = result["answer"], result["ok"]
                sources = [] if not ok else list(dict.fromkeys(
                    source_label(d.metadata) for d in result["docs"]))
            except Exception as e:
                print(f"[web_app] error: {type(e).__name__}: {e}")
                answer, ok, sources = "Sorry, something went wrong while answering.", False, []

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "error": not ok,
    })
    st.rerun()  # redraw the page with the full history