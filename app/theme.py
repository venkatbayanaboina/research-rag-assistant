"""Central design system for the Research RAG Suite UI.

inject_theme() emits the full stylesheet once at app start. Everything visual
(chrome, sidebar, chat, chips, buttons) is styled from here so the look stays
coherent and can be re-tuned in one place.

Design language: creamy light glassmorphism — frosted translucent panels over
soft aurora gradients, thick rounded corners, airy spacing, minimal text.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Design tokens ---------- */
:root {
  --bg: #F5F1E8;
  --ink: #232A3B;
  --muted: #6B7288;
  --accent: #6C5CE7;
  --accent-2: #3E8DE3;
  --glass: rgba(255, 255, 255, 0.58);
  --glass-strong: rgba(255, 255, 255, 0.78);
  --glass-border: rgba(255, 255, 255, 0.85);
  --shadow: 0 10px 40px rgba(96, 91, 152, 0.14), 0 2px 8px rgba(96, 91, 152, 0.06);
  --radius: 22px;
}

/* ---------- Base chrome ---------- */
html, body, .stApp {
  background:
    radial-gradient(900px 480px at 12% -8%, rgba(178, 160, 255, 0.35), transparent 60%),
    radial-gradient(820px 420px at 95% 12%, rgba(255, 199, 168, 0.34), transparent 55%),
    radial-gradient(760px 520px at 55% 115%, rgba(148, 196, 255, 0.30), transparent 60%),
    var(--bg);
  background-attachment: fixed;
  color: var(--ink);
}
#MainMenu, footer, div[data-testid="stDecoration"],
header[data-testid="stHeader"] [data-testid="stStatusWidget"],
div[data-testid="stAppDeployButton"], div[data-testid="stToolbarActions"] {
  display: none;
}
header[data-testid="stHeader"] {
  background: transparent;
}
section[data-testid="stMain"] > div.block-container {
  padding-top: 2.6rem;
  max-width: 60rem;
}
::selection { background: rgba(108, 92, 231, 0.22); }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: rgba(108, 92, 231, 0.22); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(108, 92, 231, 0.38); }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- Glass primitives ---------- */
.glass, div[data-testid="stMetric"], div[data-testid="stExpander"],
div[data-testid="stChatMessage"], div[data-testid="stFileUploaderDropzone"],
div[data-testid="stAlert"] {
  background: var(--glass);
  -webkit-backdrop-filter: blur(22px) saturate(1.5);
  backdrop-filter: blur(22px) saturate(1.5);
  border: 1px solid var(--glass-border);
  box-shadow: var(--shadow);
}

/* ---------- Hero header ---------- */
.hero { margin: 0 0 0.3rem 0; }
.hero-top { display: flex; align-items: center; gap: 16px; }
.hero-mark {
  width: 52px; height: 52px; border-radius: 18px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; color: #fff;
  background: linear-gradient(135deg, #6C5CE7 0%, #3E8DE3 100%);
  box-shadow: 0 12px 30px rgba(108, 92, 231, 0.35);
}
.hero-title {
  font-family: 'Outfit', sans-serif; font-size: 2.1rem; font-weight: 700;
  letter-spacing: -0.03em; line-height: 1.1; margin: 0;
  background: linear-gradient(100deg, #2A3150 15%, #6C5CE7 60%, #3E8DE3 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: 0.98rem; margin: 0.5rem 0 0 0; font-weight: 400; }
.hero-badges { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 16px; }
.hbadge {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
  color: #4A5170; text-transform: uppercase;
  padding: 7px 14px; border-radius: 999px;
  background: var(--glass);
  -webkit-backdrop-filter: blur(18px) saturate(1.4);
  backdrop-filter: blur(18px) saturate(1.4);
  border: 1px solid var(--glass-border);
  box-shadow: 0 4px 14px rgba(96, 91, 152, 0.10);
}
.hbadge .dot { width: 7px; height: 7px; border-radius: 50%;
  background: linear-gradient(135deg, #6C5CE7, #3E8DE3); }

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
  border-radius: var(--radius);
  padding: 18px 20px 14px 20px;
}
div[data-testid="stMetric"], div[data-testid="stMetric"] * {
  font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] > div > label {
  color: var(--muted) !important; font-size: 0.7rem !important;
  text-transform: uppercase; letter-spacing: 0.10em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
  font-weight: 700 !important; letter-spacing: -0.02em; color: var(--ink) !important;
}

/* ---------- Sidebar (frosted panel) ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.66), rgba(255,255,255,0.42));
  -webkit-backdrop-filter: blur(26px) saturate(1.5);
  backdrop-filter: blur(26px) saturate(1.5);
  border-right: 1px solid rgba(255, 255, 255, 0.9);
  box-shadow: 6px 0 40px rgba(96, 91, 152, 0.10);
}
/* Breathing room above the uploader — the first control sat flush to the top */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
  padding-top: 2.6rem;
}
section[data-testid="stSidebar"] .side-label:first-of-type {
  margin-top: 0.4rem;
}
.side-label {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--muted);
  margin: 1.25rem 0 0.6rem 0;
}
.side-label::after { content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(108,92,231,0.30), transparent); }

section[data-testid="stSidebar"] .stButton > button {
  width: 100%; border-radius: 14px; font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.9);
  background: var(--glass);
  -webkit-backdrop-filter: blur(14px);
  backdrop-filter: blur(14px);
  color: var(--ink);
  box-shadow: 0 4px 14px rgba(96, 91, 152, 0.10);
  transition: all 0.18s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  transform: translateY(-1px);
  border-color: rgba(108, 92, 231, 0.45);
  box-shadow: 0 8px 22px rgba(108, 92, 231, 0.20);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6C5CE7 0%, #3E8DE3 100%);
  border: none; color: #fff;
  box-shadow: 0 10px 26px rgba(108, 92, 231, 0.38);
}
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 18px; border: 1.5px dashed rgba(108, 92, 231, 0.35);
  background: rgba(255, 255, 255, 0.45);
}
div[data-testid="stFileUploaderDropzone"] button { border-radius: 11px !important; }

/* Document library cards */
.doc-card {
  border-radius: 16px;
  background: var(--glass);
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  box-shadow: 0 4px 16px rgba(96, 91, 152, 0.09);
  padding: 12px 14px; margin-bottom: 4px;
}
.doc-name {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.76rem; color: var(--ink); font-weight: 600;
  word-break: break-all; line-height: 1.4;
}
.doc-meta { color: var(--muted); font-size: 0.7rem; margin-top: 4px; }

/* Progress */
div[data-testid="stProgressBar"] > div {
  background: linear-gradient(90deg, #6C5CE7, #3E8DE3);
}
.stage-msg { color: var(--muted); font-size: 0.8rem; margin: 6px 0 4px 0; }

/* ---------- Buttons (main area) ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 15px; font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.9);
  background: var(--glass);
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
  color: var(--ink);
  box-shadow: 0 4px 14px rgba(96, 91, 152, 0.10);
  transition: all 0.18s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  border-color: rgba(108, 92, 231, 0.5);
  box-shadow: 0 10px 26px rgba(108, 92, 231, 0.22);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6C5CE7 0%, #3E8DE3 100%);
  border: none; color: #fff;
  box-shadow: 0 10px 26px rgba(108, 92, 231, 0.35);
}

/* ---------- Chat ---------- */
div[data-testid="stChatMessage"] {
  border-radius: 24px;
  padding: 10px 12px;
  margin-bottom: 14px;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(135deg, rgba(108, 92, 231, 0.16), rgba(62, 141, 227, 0.12));
  border-color: rgba(108, 92, 231, 0.30);
}
div[data-testid="stChatInput"] textarea, div[data-testid="stChatInput"] {
  background: rgba(255, 255, 255, 0.7);
}
div[data-testid="stChatInput"] > div {
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.95);
  box-shadow: var(--shadow);
  -webkit-backdrop-filter: blur(18px);
  backdrop-filter: blur(18px);
}

/* Verification chips */
.vchip {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
  padding: 5px 13px; border-radius: 999px; margin-top: 10px;
  border: 1px solid;
}
.vchip-verified { color: #167C5B; border-color: rgba(22, 124, 91, 0.30);
  background: rgba(52, 211, 153, 0.16); }
.vchip-qualified { color: #9A6B00; border-color: rgba(214, 158, 46, 0.35);
  background: rgba(251, 191, 36, 0.18); }
.vchip-failed { color: #B23B3B; border-color: rgba(178, 59, 59, 0.30);
  background: rgba(248, 113, 113, 0.16); }

/* Source cards inside expanders */
.src-head { display: flex; align-items: center; gap: 8px; margin: 10px 0 6px 0; }
.src-file {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.72rem; color: #3B4258; font-weight: 600;
  padding: 4px 11px; border-radius: 9px;
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(255, 255, 255, 0.95);
  box-shadow: 0 2px 8px rgba(96, 91, 152, 0.08);
}
.src-page {
  font-size: 0.7rem; font-weight: 700; color: #4A3FBF;
  padding: 4px 11px; border-radius: 9px;
  background: rgba(108, 92, 231, 0.12);
  border: 1px solid rgba(108, 92, 231, 0.28);
}
.src-text {
  font-size: 0.83rem; color: #454E66; line-height: 1.6;
  border-left: 3px solid rgba(108, 92, 231, 0.35);
  padding-left: 14px; margin: 0 0 14px 2px; white-space: pre-wrap;
}

/* Expander cards */
div[data-testid="stExpander"] { border-radius: 18px; }
div[data-testid="stExpander"] details { border: none !important; background: transparent !important; }
div[data-testid="stExpander"] summary { font-weight: 600; font-size: 0.9rem; color: var(--ink); }

/* Figure frame */
.fig-frame {
  border-radius: 18px; overflow: hidden;
  border: 1px solid var(--glass-border);
  box-shadow: var(--shadow);
  background: rgba(255,255,255,0.6);
}
.fig-cap {
  font-size: 0.72rem; color: var(--muted); padding: 9px 14px;
  border-top: 1px solid rgba(255,255,255,0.9); background: rgba(255,255,255,0.65);
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 16px; border: 1px solid rgba(255,255,255,0.9); }

/* Empty state */
hr { border-color: rgba(108, 92, 231, 0.15); }
.empty-hero { text-align: center; padding: 2rem 1rem 1rem 1rem; }
.empty-glyph {
  width: 68px; height: 68px; margin: 0 auto 16px auto; border-radius: 24px;
  display: flex; align-items: center; justify-content: center; font-size: 30px; color: #fff;
  background: linear-gradient(135deg, rgba(108,92,231,0.9), rgba(62,141,227,0.85));
  box-shadow: 0 14px 34px rgba(108, 92, 231, 0.35);
}
.empty-title {
  font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700;
  letter-spacing: -0.02em; margin: 0 0 6px 0; color: var(--ink);
}
.empty-sub { color: var(--muted); font-size: 0.92rem; margin: 0 auto 1.5rem auto; max-width: 30rem; }
"""

def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero_html(title: str, subtitle: str, badges: list) -> str:
    badges_html = ""
    for b in badges:
        badges_html += f'<span class="hbadge"><span class="dot"></span>{b}</span>'
    return f"""
<div class="hero">
  <div class="hero-top">
    <div class="hero-mark">◈</div>
    <div>
      <h1 class="hero-title">{title}</h1>
    </div>
  </div>
  <p class="hero-sub">{subtitle}</p>
  <div class="hero-badges">{badges_html}</div>
</div>
"""


def verification_chip(status: str, coverage) -> str:
    """Renders the citation-verification outcome as a colored pill."""
    if status == "verified":
        pct = f"{coverage*100:.0f}% cited" if isinstance(coverage, (int, float)) else ""
        label = f"✓ Verified{(' · ' + pct) if pct else ''}"
        cls = "vchip-verified"
    elif status == "qualified":
        label = "⚠ Partially verified"
        cls = "vchip-qualified"
    elif status == "failed":
        label = "✗ Citation check failed"
        cls = "vchip-failed"
    else:
        return ""
    return f'<span class="vchip {cls}">{label}</span>'


def source_head(file: str, page, extra: str = "") -> str:
    return (f'<div class="src-head"><span class="src-file">{file}</span>'
            f'<span class="src-page">Page {page}</span>{extra}</div>')


def source_text(text: str) -> str:
    import html as _html
    return f'<div class="src-text">{_html.escape(text)}</div>'


def figure_frame(path: str, caption: str) -> str:
    import base64, os, html as _html
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpeg"
        cap = _html.escape(caption or "")
        return (f'<div class="fig-frame"><img src="data:image/{ext};base64,{b64}" '
                f'style="width:100%;display:block;"/>'
                f'<div class="fig-cap">{cap}</div></div>')
    except Exception:
        return ""
