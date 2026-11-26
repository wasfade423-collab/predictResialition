"""
ChurnGuard — landing (inchangée structurellement) + formulaire premium.
removeChild fix: aucun HTML custom dans st.form ; pas de display:none sur chrome Streamlit.
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
HERO_WIDE = APP_DIR / "assets" / "churnguard-hero-wide.jpg"
HERO_SQ = APP_DIR / "assets" / "churnguard-hero.jpg"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.churn_core import (  # noqa: E402
    DEPARTEMENTS,
    load_history,
    load_model,
    load_monitoring,
    monitoring_summary,
    predict_one,
    record_outcome,
    save_prediction,
    train_and_export,
)

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"


def ensure_theme() -> str:
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    return st.session_state.theme


def _remix_icon_data_uri(name: str) -> str:
    """SVG officiels Remix Icon (sun-line / moon-line) en data-URI."""
    # Source: https://remixicon.com/icon/sun-line · https://remixicon.com/icon/moon-line
    paths = {
        "sun-line": (
            "M12 18C8.68629 18 6 15.3137 6 12C6 8.68629 8.68629 6 12 6C15.3137 6 18 8.68629 "
            "18 12C18 15.3137 15.3137 18 12 18ZM12 16C14.2091 16 16 14.2091 16 12C16 9.79086 "
            "14.2091 8 12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16ZM11 1H13V4H11V1ZM11 "
            "20H13V23H11V20ZM3.51472 4.92893L4.92893 3.51472L7.05025 5.63604L5.63604 7.05025L3.51472 "
            "4.92893ZM16.9497 18.364L18.364 16.9497L20.4853 19.0711L19.0711 20.4853L16.9497 "
            "18.364ZM19.0711 3.51472L20.4853 4.92893L18.364 7.05025L16.9497 5.63604L19.0711 "
            "3.51472ZM5.63604 16.9497L7.05025 18.364L4.92893 20.4853L3.51472 19.0711L5.63604 "
            "16.9497ZM23 11V13H20V11H23ZM4 11V13H1V11H4Z"
        ),
        "moon-line": (
            "M10 7C10 10.866 13.134 14 17 14C18.9584 14 20.729 13.1957 21.9995 11.8995C22 11.933 "
            "22 11.9665 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 "
            "6.47715 2 12 2C12.0335 2 12.067 2 12.1005 2.00049C10.8043 3.27098 10 5.04157 10 "
            "7ZM4 12C4 16.4183 7.58172 20 12 20C15.0583 20 17.7158 18.2839 19.062 15.7621C18.3945 "
            "15.9187 17.7035 16 17 16C12.0294 16 8 11.9706 8 7C8 6.29648 8.08133 5.60547 8.2379 "
            "4.938C5.71611 6.28423 4 8.9417 4 12Z"
        ),
    }
    path = paths[name]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#12141a">'
        f'<path d="{path}"/></svg>'
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def render_theme_toggle() -> None:
    """
    Icône thème sur la PAGE (pas dans la sidebar).
    - Sidebar fermée → juste à droite de la flèche >>
    - Sidebar ouverte → juste après la largeur de la sidebar (+ padding)
    """
    is_dark = st.session_state.get("theme", "dark") == "dark"
    icon_uri = _remix_icon_data_uri("moon-line" if is_dark else "sun-line")
    st.markdown(
        f"""
<style>
/* Flèche ouvrir / fermer toujours visible */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {{
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 1000005 !important;
}}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button {{
  background: #ffffff !important;
  color: #12141a !important;
  border: 0 !important;
  border-radius: 0.55rem !important;
  box-shadow: 0 2px 14px rgba(0,0,0,.35) !important;
  width: 2.35rem !important;
  height: 2.35rem !important;
  min-height: 2.35rem !important;
}}

/* Host du bouton thème (type tertiary = unique, hors Retour secondary) */
div[data-testid="stElementContainer"]:has(button[kind="tertiary"]),
div[data-testid="stElementContainer"]:has(button[data-testid="stBaseButton-tertiary"]) {{
  position: fixed !important;
  top: 0.72rem !important;
  /* Sidebar FERMÉE : à côté de >> (flèche ~0.75–1rem, bouton 2.35rem → ~3.5rem) */
  left: 3.55rem !important;
  z-index: 1000007 !important;
  width: 2.45rem !important;
  height: 2.45rem !important;
  margin: 0 !important;
  padding: 0 !important;
}}
/* Sidebar OUVERTE : juste après la sidebar (+ padding) */
.stApp:has([data-testid="stSidebar"][aria-expanded="true"])
  div[data-testid="stElementContainer"]:has(button[kind="tertiary"]),
.stApp:has([data-testid="stSidebar"][aria-expanded="true"])
  div[data-testid="stElementContainer"]:has(button[data-testid="stBaseButton-tertiary"]) {{
  left: calc(21rem + 0.85rem) !important;
}}
@media (max-width: 768px) {{
  .stApp:has([data-testid="stSidebar"][aria-expanded="true"])
    div[data-testid="stElementContainer"]:has(button[kind="tertiary"]),
  .stApp:has([data-testid="stSidebar"][aria-expanded="true"])
    div[data-testid="stElementContainer"]:has(button[data-testid="stBaseButton-tertiary"]) {{
    left: 3.55rem !important;
  }}
}}

button[kind="tertiary"],
button[data-testid="stBaseButton-tertiary"] {{
  width: 2.45rem !important;
  min-width: 2.45rem !important;
  max-width: 2.45rem !important;
  height: 2.45rem !important;
  min-height: 2.45rem !important;
  padding: 0 !important;
  border-radius: 0.55rem !important;
  border: 0 !important;
  background: #ffffff url("{icon_uri}") center / 18px 18px no-repeat !important;
  box-shadow: 0 2px 14px rgba(0,0,0,.35) !important;
  color: transparent !important;
  font-size: 0 !important;
  line-height: 0 !important;
}}
button[kind="tertiary"] *,
button[data-testid="stBaseButton-tertiary"] * {{
  opacity: 0 !important;
}}
button[kind="tertiary"]:hover,
button[data-testid="stBaseButton-tertiary"]:hover {{
  filter: brightness(0.96);
  transform: none !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )
    if st.button(
        "·",
        key="theme_toggle",
        type="tertiary",
        help="Basculer clair / sombre",
    ):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()


def inject_app_css(theme: str = "dark") -> None:
    """CSS app — police lisible (Plus Jakarta Sans) + thèmes clair/sombre."""
    dark = theme != "light"
    if dark:
        theme_css = """
:root {
  --ink: #eef1f6;
  --soft: #9aa3b5;
  --line: rgba(255,255,255,.10);
  --coral: #ff6a3d;
  --coral-deep: #e4572e;
  --title: #ffffff;
  --label: #d7dde8;
  --input-bg: rgba(8,10,16,.75);
  --input-border: rgba(255,255,255,.14);
  --input-text: #f3f5f9;
  --step-bg: rgba(255,255,255,.03);
  --step-border: rgba(255,255,255,.08);
  --step-text: #e8ecf4;
  --panel-bg: linear-gradient(155deg, rgba(255,255,255,.08) 0%, rgba(255,255,255,.03) 100%);
  --fields-bg: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.025));
  --fields-border: rgba(255,255,255,.12);
  --sec-btn-bg: rgba(255,255,255,.06);
  --sec-btn-border: rgba(255,255,255,.14);
  --sec-btn-text: #eef1f6;
  --metric-bg: rgba(255,255,255,.04);
  --metric-border: rgba(255,255,255,.1);
  --empty-border: rgba(255,255,255,.16);
  --empty-bg: rgba(255,255,255,.03);
  --verdict-ok-bg: linear-gradient(145deg, rgba(30,58,95,.55), rgba(15,23,42,.7));
  --verdict-ok-border: rgba(96,165,250,.35);
  --verdict-risk-bg: linear-gradient(145deg, rgba(90,35,20,.55), rgba(40,18,12,.75));
  --verdict-body: #d7dde8;
}
.stApp {
  background:
    radial-gradient(ellipse 55% 45% at 12% 0%, rgba(255,106,61,.18), transparent 55%),
    radial-gradient(ellipse 40% 35% at 90% 15%, rgba(80,140,255,.10), transparent 50%),
    linear-gradient(165deg, #0b0d14 0%, #12161f 45%, #0e1118 100%);
  background-attachment: fixed;
}
.stApp::before {
  background-image:
    linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #10131a 0%, #080a0f 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] * { color: #f4f4f5 !important; }
"""
    else:
        theme_css = """
:root {
  --ink: #141820;
  --soft: #5a6578;
  --line: rgba(20,24,32,.10);
  --coral: #e4572e;
  --coral-deep: #c2410c;
  --title: #12141a;
  --label: #1c2430;
  --input-bg: #ffffff;
  --input-border: #d5dbe6;
  --input-text: #12141a;
  --step-bg: #ffffff;
  --step-border: #e2e7ef;
  --step-text: #2a3344;
  --panel-bg: linear-gradient(155deg, #ffffff 0%, #f7f8fb 100%);
  --fields-bg: linear-gradient(180deg, #ffffff, #f8f9fc);
  --fields-border: #e2e7ef;
  --sec-btn-bg: #eef1f6;
  --sec-btn-border: #d5dbe6;
  --sec-btn-text: #1c2430;
  --metric-bg: #ffffff;
  --metric-border: #e2e7ef;
  --empty-border: #d0d7e2;
  --empty-bg: #f4f6fa;
  --verdict-ok-bg: linear-gradient(145deg, #eef6ff, #e4eefc);
  --verdict-ok-border: #b6d4fe;
  --verdict-risk-bg: linear-gradient(145deg, #fff5ef, #ffe8dc);
  --verdict-body: #334155;
}
.stApp {
  background:
    radial-gradient(ellipse 55% 45% at 10% 0%, rgba(228,87,46,.12), transparent 55%),
    radial-gradient(ellipse 40% 35% at 95% 8%, rgba(80,140,255,.08), transparent 50%),
    linear-gradient(180deg, #f7f8fb 0%, #eef1f6 100%);
  background-attachment: fixed;
}
.stApp::before {
  background-image:
    linear-gradient(rgba(20,24,32,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,24,32,.04) 1px, transparent 1px);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #161a24 0%, #0f1219 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] * { color: #f4f4f5 !important; }
"""

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@500;700&display=swap');

{theme_css}

html, body, .stApp {{
  font-family: "DM Sans", system-ui, sans-serif !important;
  color: var(--ink);
}}
.stApp::before {{
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse 70% 60% at 50% 30%, #000 20%, transparent 75%);
}}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

[data-testid="stMainBlockContainer"] {{
  max-width: 980px !important;
  padding-top: 1.2rem !important;
  padding-bottom: 3.5rem !important;
  position: relative; z-index: 1;
}}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {{
  color: var(--label) !important;
  font-weight: 650 !important;
  font-size: 1.08rem !important;
  letter-spacing: -.01em;
}}
[data-testid="stMarkdownContainer"] p {{
  color: var(--soft);
  line-height: 1.65;
  font-size: 1.08rem;
}}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(22px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes beam {{
  0% {{ background-position: 0% 50%; }}
  100% {{ background-position: 100% 50%; }}
}}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
  color: rgba(244,244,245,.55) !important;
  font-size: .95rem !important;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .75rem .85rem;
  border-radius: 14px;
  font-weight: 650;
  font-size: 1.05rem !important;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: rgba(255,255,255,.06);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: rgba(255,106,61,.22);
}}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 14px;
  padding: .85rem 1rem;
}}
.sb-brand {{
  display: flex; gap: .75rem; align-items: center;
  padding-bottom: 1.1rem; margin-bottom: .7rem;
  border-bottom: 1px solid rgba(255,255,255,.1);
}}
.sb-mark {{
  width: 42px; height: 42px; border-radius: 12px;
  background: var(--coral);
  display: grid; place-items: center;
  box-shadow: 0 0 24px rgba(255,106,61,.45);
}}
.sb-name {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-weight: 800; font-size: 1.2rem;
}}
.sb-sub {{
  font-size: .78rem; opacity: .5;
  letter-spacing: .12em; text-transform: uppercase;
}}

.page-head {{
  margin: 0 0 1.5rem;
  padding: 1.3rem 1.45rem 1.35rem;
  border-radius: 22px;
  background: var(--panel-bg);
  border: 1px solid var(--line);
  backdrop-filter: blur(18px);
  animation: rise .5s ease both;
}}
.page-head .k {{
  color: var(--coral) !important;
  font-size: .82rem !important;
  font-weight: 800 !important;
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: .4rem;
}}
.page-head h2 {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif !important;
  font-size: clamp(1.9rem, 3.4vw, 2.5rem) !important;
  font-weight: 800 !important;
  margin: 0 0 .45rem !important;
  color: var(--title) !important;
  letter-spacing: -.03em;
  line-height: 1.15;
}}
.page-head p {{
  margin: 0 !important;
  max-width: 40rem;
  color: var(--soft) !important;
  font-size: 1.12rem !important;
  line-height: 1.6 !important;
}}

.wiz-top {{
  margin-bottom: 1.35rem;
  animation: rise .55s ease both;
}}
.wiz-kicker {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--coral);
  margin-bottom: .55rem;
}}
.wiz-title {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: clamp(2rem, 3.8vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -.028em;
  color: var(--title);
  margin: 0 0 .55rem;
  line-height: 1.12;
}}
.wiz-sub {{
  color: var(--soft);
  font-size: 1.15rem;
  line-height: 1.55;
  max-width: 36rem;
  margin: 0;
}}

.wiz-rail {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .85rem;
  margin: 1.5rem 0 1.35rem;
  animation: rise .55s .08s ease both;
}}
.wiz-step {{
  position: relative;
  padding: 1rem 1.05rem 1.05rem;
  border-radius: 18px;
  background: var(--step-bg);
  border: 1px solid var(--step-border);
  overflow: hidden;
}}
.wiz-step::after {{
  content: "";
  position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  background: var(--step-border);
}}
.wiz-step.on {{
  background: linear-gradient(160deg, rgba(255,106,61,.18), rgba(255,106,61,.04));
  border-color: rgba(255,106,61,.45);
}}
.wiz-step.on::after {{
  background: linear-gradient(90deg, var(--coral), #ffb087, var(--coral));
  background-size: 200% 100%;
  animation: beam 2.4s linear infinite;
}}
.wiz-step.done {{
  border-color: rgba(255,106,61,.28);
  background: rgba(255,106,61,.07);
}}
.wiz-step .n {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .12em;
  color: var(--soft);
  margin-bottom: .35rem;
}}
.wiz-step.on .n {{ color: var(--coral); }}
.wiz-step .t {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-weight: 750;
  font-size: 1.12rem;
  color: var(--step-text);
}}
.wiz-step.on .t {{ color: var(--title); }}

.wiz-panel {{
  position: relative;
  padding: 1.35rem 1.5rem 1.3rem;
  border-radius: 24px 24px 0 0;
  background: var(--panel-bg);
  border: 1px solid var(--fields-border);
  border-bottom: none;
  box-shadow: 0 0 0 1px rgba(255,106,61,.08) inset;
  backdrop-filter: blur(22px);
  animation: rise .6s .12s ease both;
  margin-bottom: 0;
}}
.wiz-panel::before {{
  content: "";
  position: absolute; top: 0; left: 12%; right: 12%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,106,61,.7), transparent);
}}
.wiz-panel-head {{
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1rem;
}}
.wiz-panel-head .phase {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .78rem; letter-spacing: .16em; color: var(--coral);
  font-weight: 700; text-transform: uppercase;
}}
.wiz-panel-head h3 {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: clamp(1.5rem, 2.6vw, 1.9rem);
  font-weight: 800; color: var(--title);
  margin: .2rem 0 0; letter-spacing: -.025em;
}}
.wiz-panel-head .hint {{
  color: var(--soft); font-size: 1rem; max-width: 16rem; text-align: right;
  line-height: 1.45;
}}
div[data-testid="stHorizontalBlock"]:has([data-testid="stTextInput"]),
div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]),
div[data-testid="stHorizontalBlock"]:has([data-testid="stSlider"]) {{
  background: var(--fields-bg);
  border: 1px solid var(--fields-border);
  border-top: 1px solid var(--line);
  border-radius: 0 0 24px 24px;
  padding: 1.35rem 1.35rem 1.5rem !important;
  margin-top: 0 !important;
  margin-bottom: .85rem !important;
  box-shadow: 0 22px 50px rgba(0,0,0,.08);
  backdrop-filter: blur(22px);
  animation: rise .6s .16s ease both;
}}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
  background: var(--input-bg) !important;
  border: 1px solid var(--input-border) !important;
  color: var(--input-text) !important;
  border-radius: 14px !important;
  min-height: 3.2rem !important;
  font-size: 1.12rem !important;
  font-weight: 500 !important;
  padding: .6rem 1rem !important;
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {{
  border-color: var(--coral) !important;
  box-shadow: 0 0 0 3px rgba(255,106,61,.22) !important;
}}
div[data-baseweb="select"] > div {{
  background: var(--input-bg) !important;
  border-color: var(--input-border) !important;
  border-radius: 14px !important;
  min-height: 3.2rem !important;
  color: var(--input-text) !important;
  font-size: 1.12rem !important;
}}
div[data-testid="stNumberInput"] button {{
  background: var(--sec-btn-bg) !important;
  color: var(--input-text) !important;
  border-color: var(--input-border) !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
  background: var(--coral) !important;
  box-shadow: 0 0 16px rgba(255,106,61,.45);
}}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {{
  border-radius: 16px !important;
  min-height: 3.4rem !important;
  border: 0 !important;
  background: linear-gradient(135deg, #ff6a3d, #e4572e) !important;
  color: #fff !important;
  font-weight: 750 !important;
  font-size: 1.12rem !important;
  letter-spacing: -.01em;
  box-shadow: 0 12px 36px rgba(255,106,61,.35);
}}
div.stButton > button *,
div[data-testid="stFormSubmitButton"] button * {{ color: #fff !important; }}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {{
  filter: brightness(1.06);
  transform: translateY(-1px);
}}
div.stButton > button[kind="secondary"] {{
  background: var(--sec-btn-bg) !important;
  border: 1px solid var(--sec-btn-border) !important;
  box-shadow: none !important;
  color: var(--sec-btn-text) !important;
}}
div.stButton > button[kind="secondary"] * {{ color: var(--sec-btn-text) !important; }}

.verdict {{
  margin: 1.35rem 0 1rem;
  padding: 1.6rem 1.7rem;
  border-radius: 24px;
  border: 1px solid var(--verdict-ok-border);
  background: var(--verdict-ok-bg);
  animation: rise .5s ease both;
}}
.verdict.risk {{
  border-color: rgba(255,106,61,.45);
  background: var(--verdict-risk-bg);
}}
.verdict .vk {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .78rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--soft);
}}
.verdict .vt {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: clamp(1.55rem, 2.5vw, 1.95rem);
  font-weight: 800;
  margin: .4rem 0 .35rem; color: var(--title);
  letter-spacing: -.025em;
}}
.verdict .vm {{ color: var(--soft); font-size: 1.05rem; }}
.verdict .vc {{
  margin-top: .8rem; color: var(--verdict-body);
  font-size: 1.12rem; line-height: 1.7;
}}

.empty {{
  text-align: center;
  padding: 2.4rem 1.2rem;
  border: 1px dashed var(--empty-border);
  border-radius: 22px;
  background: var(--empty-bg);
  color: var(--soft);
  font-size: 1.08rem;
}}
.empty strong {{
  display: block; color: var(--title);
  margin-bottom: .35rem; font-size: 1.2rem;
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
}}

[data-testid="stMetric"] {{
  background: var(--metric-bg);
  border: 1px solid var(--metric-border);
  border-radius: 18px;
  padding: 1.05rem 1.1rem;
}}
[data-testid="stMetricValue"] {{
  color: var(--title) !important;
  font-weight: 800 !important;
  font-size: 1.5rem !important;
}}
[data-testid="stMetricLabel"] {{
  color: var(--soft) !important;
  font-size: 1rem !important;
}}
[data-testid="stProgressBarTrack"] > div {{ background: var(--coral) !important; }}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--metric-border); border-radius: 16px; overflow: hidden;
}}
[data-testid="stForm"] {{
  background: transparent;
  border: none;
  padding: 0 !important;
}}
div[data-testid="stCaptionContainer"] p {{ color: var(--soft) !important; font-size: 1rem !important; }}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model()


def try_load():
    try:
        return get_model()
    except FileNotFoundError:
        return None, None


def pct(meta: dict, key: str, digits: int = 1) -> str:
    value = meta.get(key)
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f} %"
    except (TypeError, ValueError):
        return "—"


def page_head(kicker: str, title: str, desc: str) -> None:
    st.html(
        f"""
<div class="page-head">
  <div class="k">{html.escape(kicker)}</div>
  <h2>{html.escape(title)}</h2>
  <p>{html.escape(desc)}</p>
</div>
"""
    )


if "view" not in st.session_state:
    st.session_state.view = "landing"

# ============================== LANDING ====================================
# Pas de bouton thème ici — uniquement sur le formulaire (sidebar app)
if st.session_state.view == "landing":
    hero = HERO_WIDE if HERO_WIDE.exists() else HERO_SQ
    bg = data_uri(hero) if hero.exists() else ""

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');
html, body, .stApp {{ font-family: "DM Sans", "Plus Jakarta Sans", system-ui, sans-serif; }}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{ background: transparent !important; }}
[data-testid="stMainBlockContainer"] {{ max-width: 100% !important; padding: 0 !important; }}
#MainMenu, footer {{ visibility: hidden; }}
@keyframes rise {{
  from {{ opacity: 0; transform: translateY(28px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes drift {{
  0%, 100% {{ transform: translate3d(0,0,0) scale(1); }}
  50% {{ transform: translate3d(12px,-18px,0) scale(1.04); }}
}}
@keyframes shimmer {{
  0% {{ background-position: 0% 50%; }}
  100% {{ background-position: 100% 50%; }}
}}
@keyframes pulseGlow {{
  0%, 100% {{ opacity: .55; }}
  50% {{ opacity: .9; }}
}}
.cg-stage {{
  position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
}}
.cg-stage-img {{
  position: absolute; inset: -3%;
  background-image: url('{bg}');
  background-size: cover; background-position: center 38%;
  animation: drift 18s ease-in-out infinite;
}}
.cg-stage-veil {{
  position: absolute; inset: 0;
  background:
    linear-gradient(105deg,
      rgba(250,249,247,.98) 0%,
      rgba(250,249,247,.94) 42%,
      rgba(250,249,247,.55) 68%,
      rgba(18,20,26,.38) 100%),
    radial-gradient(ellipse 45% 55% at 12% 78%, rgba(228,87,46,.22), transparent 65%);
}}
.cg-stage-orb {{
  position: absolute; width: min(520px, 48vw); height: min(520px, 48vw);
  right: 6%; top: -6%; border-radius: 50%;
  background: radial-gradient(circle, rgba(228,87,46,.32), transparent 70%);
  filter: blur(36px); animation: pulseGlow 6s ease-in-out infinite;
}}
.cg-top {{
  position: relative; z-index: 2; max-width: 1180px; margin: 0 auto;
  padding: 2rem clamp(1.2rem, 4vw, 3rem) 0;
}}
.cg-nav {{
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  animation: rise .7s cubic-bezier(.22,1,.36,1) both;
}}
.cg-mark {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif; font-weight: 800;
  font-size: 1.35rem; letter-spacing: -.03em; color: #12141a;
}}
.cg-mark span {{ color: #e4572e; }}
.cg-meta {{
  font-size: 1rem; font-weight: 750; letter-spacing: .12em;
  text-transform: uppercase; color: #12141a;
  background: rgba(255,255,255,.88);
  border: 1px solid rgba(18,20,26,.08);
  padding: .55rem 1rem;
  border-radius: 999px;
  box-shadow: 0 8px 24px rgba(0,0,0,.08);
  white-space: nowrap;
}}
.cg-hero {{ max-width: 40rem; padding: 7vh 0 1.2rem; }}
.cg-brand {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: clamp(3.8rem, 9.5vw, 6.4rem); font-weight: 800;
  line-height: .95; letter-spacing: -.04em; color: #12141a;
  margin: 0 0 1.4rem;
  animation: rise .8s .05s cubic-bezier(.22,1,.36,1) both;
}}
.cg-brand em {{
  font-style: normal;
  background: linear-gradient(120deg, #e4572e, #ff7a4d 45%, #c2410c);
  background-size: 200% 200%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 4s linear infinite alternate;
}}
.cg-line {{
  font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: clamp(1.7rem, 3.4vw, 2.35rem); font-weight: 700;
  line-height: 1.28; letter-spacing: -.025em; color: #12141a;
  margin: 0 0 1.1rem; max-width: 18ch;
  animation: rise .8s .15s cubic-bezier(.22,1,.36,1) both;
}}
.cg-copy {{
  font-size: clamp(1.35rem, 2.2vw, 1.55rem); line-height: 1.7; color: #3d4554;
  margin: 0; max-width: 34rem; font-weight: 500;
  animation: rise .8s .25s cubic-bezier(.22,1,.36,1) both;
}}
.cg-foot {{
  position: relative; z-index: 2; max-width: 1180px; margin: 0 auto;
  padding: 3rem clamp(1.2rem, 4vw, 3rem) 2.8rem;
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.75rem;
  border-top: 1px solid rgba(18,20,26,.1);
  animation: rise .8s .45s cubic-bezier(.22,1,.36,1) both;
}}
.cg-foot strong {{
  display: block; font-family: "Plus Jakarta Sans", "DM Sans", system-ui, sans-serif;
  font-size: 1.35rem; color: #12141a; margin-bottom: .4rem; font-weight: 800;
}}
.cg-foot span {{
  display: block; font-size: 1.18rem; line-height: 1.55; color: #3d4554; font-weight: 500;
}}
@media (max-width: 860px) {{
  .cg-stage-veil {{
    background: linear-gradient(180deg, rgba(250,249,247,.98) 0%, rgba(250,249,247,.9) 55%, rgba(18,20,26,.45) 100%);
  }}
  .cg-foot {{ grid-template-columns: 1fr; }}
  .cg-hero {{ padding-top: 5vh; }}
  .cg-meta {{ font-size: .9rem; padding: .45rem .85rem; }}
}}
div[data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {{
  max-width: 1180px !important;
  margin: 0.45rem auto 0 !important;
  padding: 0 clamp(1.2rem, 4vw, 3rem) !important;
  position: relative; z-index: 3;
  animation: rise .8s .32s cubic-bezier(.22,1,.36,1) both;
}}
div.stButton > button[kind="primary"] {{
  min-height: 4rem !important; border-radius: 999px !important; border: 0 !important;
  background: #e4572e !important; color: #fff !important;
  font-weight: 750 !important; font-size: 1.28rem !important;
  box-shadow: 0 18px 40px rgba(228,87,46,.35);
  transition: transform .2s ease, box-shadow .2s ease, background .2s ease !important;
}}
div.stButton > button[kind="primary"] * {{ color: #fff !important; }}
div.stButton > button[kind="primary"]:hover {{
  background: #c2410c !important; transform: translateY(-2px);
  box-shadow: 0 22px 48px rgba(228,87,46,.42);
}}
</style>
<div class="cg-stage">
  <div class="cg-stage-img"></div>
  <div class="cg-stage-veil"></div>
  <div class="cg-stage-orb"></div>
</div>
<div class="cg-top">
  <div class="cg-nav">
    <div class="cg-mark">Churn<span>Guard</span></div>
    <div class="cg-meta">Rétention clients</div>
  </div>
  <div class="cg-hero">
    <h1 class="cg-brand">Churn<em>Guard</em></h1>
    <p class="cg-line">Anticipez la résiliation avant qu’elle n’arrive.</p>
    <p class="cg-copy">
      Scorez le risque, expliquez chaque décision, suivez l’historique
      et améliorez le modèle avec le feedback terrain.
    </p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    _l, mid, _r = st.columns([0.02, 0.36, 0.62])
    with mid:
        if st.button(
            "Commencer l’analyse →",
            type="primary",
            use_container_width=True,
            key="landing_cta",
        ):
            st.session_state.view = "app"
            st.session_state.page = "Prédiction"
            st.rerun()

    st.markdown(
        """
<div class="cg-foot">
  <div><strong>Score &amp; justification</strong><span>Probabilité claire, décision actionnable.</span></div>
  <div><strong>Historique vivant</strong><span>Chaque profil évalué, avec confiance et résultat.</span></div>
  <div><strong>Amélioration continue</strong><span>Monitoring et réentraînement guidés.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# ============================== APP ========================================
ensure_theme()
inject_app_css(st.session_state.theme)
# Thème sur la page (à côté de la flèche sidebar), pas dans la sidebar
render_theme_toggle()

with st.sidebar:
    st.html(
        """
<div class="sb-brand">
  <div class="sb-mark">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4">
      <path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3z"/>
    </svg>
  </div>
  <div>
    <div class="sb-name">ChurnGuard</div>
    <div class="sb-sub">Workspace</div>
  </div>
</div>
"""
    )
    if st.button("← Accueil", use_container_width=True, key="back_home", type="primary"):
        st.session_state.view = "landing"
        st.rerun()

    st.caption("Navigation")
    page = st.radio(
        "nav",
        ["Prédiction", "Historique", "Monitoring", "Réentraînement"],
        key="page",
        label_visibility="collapsed",
    )
    st.divider()
    pipe_meta = try_load()
    if pipe_meta[0] is None:
        st.warning("Aucun modèle.")
    else:
        _, meta = pipe_meta
        st.caption("Modèle")
        st.markdown(f"**{html.escape(str(meta.get('best_model', '—')))}**")
        st.metric("Acc. haute confiance", pct(meta, "test_accuracy"))
        full = meta.get("test_accuracy_full") or meta.get("test_accuracy_opt")
        if full is not None:
            st.metric("Acc. test complet", f"{float(full) * 100:.1f} %")

# ----- Prédiction (wizard section par section) -----
if page == "Prédiction":
    pipeline, meta = try_load()
    if pipeline is None:
        st.error("Modèle indisponible — lancez un entraînement.")
        st.stop()

    if "pred_step" not in st.session_state:
        st.session_state.pred_step = 0

    # Defaults une seule fois (évite conflit key/value Streamlit)
    _defaults = {
        "f_id_client": "NT100000",
        "f_zone": "urbaine",
        "f_departement": "Littoral" if "Littoral" in DEPARTEMENTS else DEPARTEMENTS[0],
        "f_type_offre": "prépayé",
        "f_age": 32,
        "f_anciennete": 24,
        "f_appels": 120.0,
        "f_data": 2500.0,
        "f_sms": 40,
        "f_depense": 4500,
        "f_recharges": 6,
        "f_momo": "non",
        "f_reclamations": 1,
        "f_note": 3,
        "f_variation": -5.0,
    }
    for _k, _v in _defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    STEPS = [
        ("01", "Identité", "Qui est le client ?"),
        ("02", "Usage", "Comment consomme-t-il ?"),
        ("03", "Expérience", "Quelle qualité de service ?"),
    ]
    step = int(st.session_state.pred_step)
    step = max(0, min(step, 2))
    st.session_state.pred_step = step

    rail_html = []
    for i, (num, label, _) in enumerate(STEPS):
        cls = "on" if i == step else ("done" if i < step else "")
        rail_html.append(
            f'<div class="wiz-step {cls}"><div class="n">PHASE {num}</div>'
            f'<div class="t">{label}</div></div>'
        )
    st.html(
        f"""
<div class="wiz-top">
  <div class="wiz-kicker">Mission · Scoring client</div>
  <h1 class="wiz-title">Scanner un profil</h1>
  <p class="wiz-sub">Parcours guidé en 3 phases. Chaque étape affine le signal — puis le modèle délivre score, décision et justification.</p>
</div>
<div class="wiz-rail">{''.join(rail_html)}</div>
"""
    )

    num, label, hint = STEPS[step]
    st.html(
        f"""
<div class="wiz-panel">
  <div class="wiz-panel-head">
    <div>
      <div class="phase">Phase {num} / 03</div>
      <h3>{label}</h3>
    </div>
    <div class="hint">{hint}</div>
  </div>
</div>
"""
    )

    # Champs natifs uniquement (key sans value) — hors de tout HTML
    if step == 0:
        a, b = st.columns(2, gap="large")
        with a:
            st.text_input("ID client", key="f_id_client")
            st.selectbox("Zone", ["urbaine", "rurale"], key="f_zone")
            st.selectbox("Département", DEPARTEMENTS, key="f_departement")
        with b:
            st.selectbox("Type d’offre", ["prépayé", "postpayé"], key="f_type_offre")
            st.number_input("Âge", 15, 100, key="f_age")
            st.number_input("Ancienneté (mois)", 0, 240, key="f_anciennete")

    elif step == 1:
        a, b = st.columns(2, gap="large")
        with a:
            st.number_input("Appels (min / mois)", 0.0, key="f_appels", step=5.0)
            st.number_input("Data (Mo)", 0.0, key="f_data", step=50.0)
            st.number_input("SMS", 0, key="f_sms")
        with b:
            st.number_input("Dépense (FCFA)", 0, key="f_depense", step=100)
            st.number_input("Nb recharges", 0, key="f_recharges")
            st.selectbox("Mobile Money", ["non", "oui"], key="f_momo")

    else:
        a, b = st.columns(2, gap="large")
        with a:
            st.number_input("Réclamations (3 mois)", 0, 20, key="f_reclamations")
            st.slider("Note réseau", 1, 5, key="f_note")
        with b:
            st.number_input("Variation conso (%)", key="f_variation", step=0.5)
            st.caption("Données récentes et cohérentes → score plus fiable.")

    # Navigation (widgets purs)
    st.write("")
    nav_l, nav_r = st.columns([1, 1], gap="medium")
    with nav_l:
        if step > 0:
            if st.button("← Retour", use_container_width=True, type="secondary", key="wiz_back"):
                st.session_state.pred_step = step - 1
                st.rerun()
    with nav_r:
        if step < 2:
            if st.button("Continuer →", use_container_width=True, type="primary", key="wiz_next"):
                st.session_state.pred_step = step + 1
                st.rerun()
        else:
            if st.button(
                "Lancer le scan →",
                use_container_width=True,
                type="primary",
                key="wiz_predict",
            ):
                values = {
                    "id_client": str(st.session_state.get("f_id_client", "NT100000")).strip(),
                    "zone": st.session_state.get("f_zone", "urbaine"),
                    "departement": st.session_state.get("f_departement", DEPARTEMENTS[0]),
                    "age": int(st.session_state.get("f_age", 32)),
                    "type_offre": st.session_state.get("f_type_offre", "prépayé"),
                    "appels_min": float(st.session_state.get("f_appels", 120.0)),
                    "data_mo": float(st.session_state.get("f_data", 2500.0)),
                    "sms": int(st.session_state.get("f_sms", 40)),
                    "depense_fcfa": float(st.session_state.get("f_depense", 4500)),
                    "nb_recharges": int(st.session_state.get("f_recharges", 6)),
                    "mobile_money": 1 if st.session_state.get("f_momo", "non") == "oui" else 0,
                    "reclamations_3m": int(st.session_state.get("f_reclamations", 1)),
                    "note_reseau": int(st.session_state.get("f_note", 3)),
                    "variation_conso_pct": float(st.session_state.get("f_variation", -5.0)),
                    "anciennete_mois": float(st.session_state.get("f_anciennete", 24)),
                }
                result = predict_one(values, pipeline, meta)
                save_prediction(result)
                st.session_state["last_result"] = result
                st.rerun()

    # Verdict
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        proba = float(r["proba_resiliation"])
        risk = r["prediction"] == 1
        conf_level = float(meta.get("conf_level", 0.9))
        high = proba >= conf_level or proba <= 1 - conf_level
        conf_label = "Haute confiance" if high else "Confiance intermédiaire"
        title = "Risque de résiliation" if risk else "Client a priori stable"
        st.html(
            f"""
<div class="verdict {'risk' if risk else ''}">
  <div class="vk">Scan terminé · {html.escape(conf_label)}</div>
  <div class="vt">{html.escape(title)}</div>
  <div class="vm">Client {html.escape(str(r.get('id_client') or '—'))} · seuil {float(r['threshold']):.2f}</div>
  <div class="vc">{html.escape(str(r['justification']))}</div>
</div>
"""
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Proba. résiliation", f"{proba:.1%}")
        m2.metric("Proba. stabilité", f"{1 - proba:.1%}")
        m3.metric("Décision", r["label"])
        st.progress(proba, text="Score de résiliation")
        if not high:
            st.info("Zone intermédiaire — une vérification humaine est recommandée.")

# ----- Historique -----
elif page == "Historique":
    hist = load_history()
    page_head(
        "Activité",
        "Historique des prédictions",
        "Retrouvez chaque évaluation, sa probabilité, et enregistrez le résultat réel.",
    )
    if hist.empty:
        st.html(
            '<div class="empty"><strong>Aucune prédiction</strong>Les analyses apparaîtront ici.</div>'
        )
    else:
        hist = hist.copy()
        if "predicted_at" in hist:
            hist["predicted_at"] = pd.to_datetime(hist["predicted_at"], errors="coerce")
        a, b, c = st.columns(3)
        a.metric("Analyses", len(hist))
        b.metric(
            "À risque",
            int((hist["prediction"].astype(int) == 1).sum()) if "prediction" in hist else 0,
        )
        c.metric(
            "Feedbacks",
            int(hist["actual_resilie"].notna().sum()) if "actual_resilie" in hist else 0,
        )

        f1, f2 = st.columns([0.35, 0.65])
        with f1:
            status = st.selectbox("Filtrer", ["Toutes", "Résilie", "Reste"])
        with f2:
            query = st.text_input("Rechercher un client", placeholder="NT…")

        shown = hist
        if status == "Résilie" and "prediction" in shown:
            shown = shown[shown["prediction"].astype(int) == 1]
        elif status == "Reste" and "prediction" in shown:
            shown = shown[shown["prediction"].astype(int) == 0]
        if query.strip() and "id_client" in shown:
            shown = shown[
                shown["id_client"].astype(str).str.contains(query.strip(), case=False, na=False)
            ]
        if "predicted_at" in shown:
            shown = shown.sort_values("predicted_at", ascending=False)

        cols = [
            x
            for x in [
                "predicted_at",
                "id_client",
                "label",
                "proba_resiliation",
                "actual_resilie",
                "justification",
            ]
            if x in shown.columns
        ]
        view = shown[cols].copy()
        if "predicted_at" in view:
            view["predicted_at"] = view["predicted_at"].dt.strftime("%d/%m/%Y %H:%M")
        if "proba_resiliation" in view:
            view["proba_resiliation"] = view["proba_resiliation"].map(
                lambda v: f"{float(v):.1%}" if pd.notna(v) else "—"
            )
        if "actual_resilie" in view:
            view["actual_resilie"] = view["actual_resilie"].map(
                lambda v: "Résilié" if v == 1 else ("Stable" if v == 0 else "En attente")
            )
        view = view.rename(
            columns={
                "predicted_at": "Date",
                "id_client": "Client",
                "label": "Décision",
                "proba_resiliation": "Probabilité",
                "actual_resilie": "Réel",
                "justification": "Justification",
            }
        )
        st.dataframe(view, hide_index=True, use_container_width=True)

        chart = hist.dropna(subset=["predicted_at"]).sort_values("predicted_at")
        if not chart.empty:
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("##### Probabilités")
                st.line_chart(
                    chart.set_index("predicted_at")[["proba_resiliation"]],
                    use_container_width=True,
                )
            with g2:
                st.markdown("##### Décisions")
                st.bar_chart(chart["label"].value_counts(), use_container_width=True)

        pending = hist[hist["actual_resilie"].isna()] if "actual_resilie" in hist else pd.DataFrame()
        st.markdown("### Résultat réel")
        if pending.empty:
            st.success("Tous les cas ont un feedback.")
        else:
            with st.form("feedback_form", border=False):
                idx = st.selectbox(
                    "Prédiction",
                    list(pending.index),
                    format_func=lambda i: (
                        f"{pending.at[i, 'id_client']} · {pending.at[i, 'predicted_at']} · "
                        f"{pending.at[i, 'label']}"
                    ),
                )
                outcome = st.radio("A résilié ?", ["Non", "Oui"], horizontal=True)
                if st.form_submit_button("Enregistrer", type="primary", use_container_width=True):
                    record_outcome(int(idx), 1 if outcome == "Oui" else 0)
                    st.success("Enregistré.")
                    st.rerun()

# ----- Monitoring -----
elif page == "Monitoring":
    page_head(
        "Surveillance",
        "Santé du modèle",
        "Volume, exactitude terrain et dérive éventuelle des scores.",
    )
    hist = load_history()
    mon = load_monitoring()
    summary = monitoring_summary(hist, mon)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prédictions", summary["n_predictions"])
    m2.metric("Avec feedback", summary["n_with_feedback"])
    acc = summary["accuracy_on_feedback"]
    m3.metric("Exactitude terrain", f"{acc:.1%}" if acc is not None else "—")
    drift = summary["proba_drift"]
    m4.metric("Dérive", f"{drift:+.3f}" if drift is not None else "—")

    if drift is not None and abs(drift) > 0.08:
        st.warning("Dérive notable — un réentraînement peut aider.")
    elif summary["n_predictions"] > 0:
        st.success("Pas de dérive marquée.")
    else:
        st.info("Le monitoring se remplira après les premières prédictions.")

    if not hist.empty:
        left, right = st.columns(2)
        with left:
            h = hist.copy()
            h["predicted_at"] = pd.to_datetime(h["predicted_at"], errors="coerce")
            h = h.dropna(subset=["predicted_at"]).sort_values("predicted_at")
            if not h.empty:
                st.line_chart(
                    h.set_index("predicted_at")[["proba_resiliation"]],
                    use_container_width=True,
                )
        with right:
            scores = pd.to_numeric(hist["proba_resiliation"], errors="coerce").dropna()
            if not scores.empty:
                buckets = (scores * 10).astype(int).clip(0, 9) / 10
                st.bar_chart(buckets.value_counts().sort_index(), use_container_width=True)

    if not mon.empty:
        st.dataframe(
            mon.sort_values("timestamp", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    _, meta = try_load()
    if meta and meta.get("rounds"):
        rounds = pd.DataFrame(meta["rounds"])
        plot = [c for c in ["accuracy_full", "accuracy_confident", "accuracy_opt"] if c in rounds.columns]
        if "round" in rounds and plot:
            st.line_chart(rounds.set_index("round")[plot], use_container_width=True)
        st.dataframe(rounds, hide_index=True, use_container_width=True)

# ----- Réentraînement -----
elif page == "Réentraînement":
    page_head(
        "Amélioration",
        "Réentraîner le modèle",
        "Comparez les algorithmes avec les données et les feedbacks confirmés.",
    )
    st.info(
        "L’accuracy haute confiance ne couvre qu’une partie des clients ; "
        "l’accuracy sur tout le test reste affichée à côté."
    )
    with st.container(border=True):
        target = st.slider("Objectif accuracy haute confiance", 0.85, 0.99, 0.97, 0.01)
        if st.button("Lancer le réentraînement", type="primary", use_container_width=True):
            with st.spinner("Entraînement…"):
                get_model.clear()
                result = train_and_export(target_score=float(target))
            tm = result["meta"]
            st.success(f"Modèle {tm['best_model']} sauvegardé.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Acc. haute confiance", pct(tm, "test_accuracy"))
            c2.metric("Acc. test complet", pct(tm, "test_accuracy_full"))
            c3.metric("Précision", pct(tm, "test_precision"))
            c4.metric("AUC", pct(tm, "test_auc"))
            if result.get("history"):
                with st.expander("Détail"):
                    st.dataframe(
                        pd.DataFrame(result["history"]),
                        hide_index=True,
                        use_container_width=True,
                    )

    hist = load_history()
    fb = (
        hist.dropna(subset=["actual_resilie"])
        if not hist.empty and "actual_resilie" in hist.columns
        else pd.DataFrame()
    )
    st.markdown("### Feedbacks disponibles")
    st.metric("Lignes labellisées", len(fb))
    if not fb.empty:
        show = [
            c
            for c in ["predicted_at", "id_client", "prediction", "proba_resiliation", "actual_resilie"]
            if c in fb.columns
        ]
        st.dataframe(fb[show], hide_index=True, use_container_width=True)
