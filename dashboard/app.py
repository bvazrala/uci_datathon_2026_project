"""
UCI Datathon 2026 — Emoji Prediction Dashboard
Single-page Streamlit app with WebGL moonlit-ripple background.

Run from repo root:
    streamlit run dashboard/app.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
ASSETS    = Path(__file__).parent / "assets"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR  = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_tweet         # noqa: E402
from predict import EMOJI_MAP, load_pipeline, predict_top_k  # noqa: E402

CLASSES = list(EMOJI_MAP.keys())
ACCENT  = "#c8956c"   # warm amber — matches the moon in the shader

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Emoji Prediction · UCI Datathon 2026",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── WebGL background injection ────────────────────────────────────────────────
def _inject_webgl():
    """
    Reads the moonlit-ripple shader HTML, extracts the main script block,
    then injects it into the *parent* Streamlit document via a 0-height
    iframe component.  A canvas element is created in the parent DOM first
    so the shader's document.getElementById('canvas') call succeeds.
    """
    html_path = ASSETS / "moonlit-ripple.html"
    if not html_path.exists():
        return

    raw = html_path.read_text()
    # Extract the first bare <script> block (no src / defer attributes)
    start = raw.find("<script>") + len("<script>")
    end   = raw.find("</script>", start)
    main_js = raw[start:end]

    injector = f"""<script>
(function () {{
  var d = window.parent.document;
  if (d.getElementById('moonlit-bg')) return;   // already injected

  // Full-screen fixed container behind everything
  var wrap = d.createElement('div');
  wrap.id = 'moonlit-bg';
  wrap.style.cssText = [
    'position:fixed', 'top:0', 'left:0',
    'width:100vw',    'height:100vh',
    'z-index:0',      'overflow:hidden',
    'background:#0a0a0a', 'pointer-events:none'
  ].join(';');

  var cv = d.createElement('canvas');
  cv.id = 'canvas';
  cv.style.cssText = 'display:block;width:100%;height:100%;';
  wrap.appendChild(cv);
  d.body.insertBefore(wrap, d.body.firstChild);

  // Inject the shader script so it runs in the parent document context
  var s = d.createElement('script');
  s.textContent = {json.dumps(main_js)};
  d.head.appendChild(s);
}})();
</script>"""
    components.html(injector, height=0, scrolling=False)


_inject_webgl()


# ── Dark glass theme ──────────────────────────────────────────────────────────
st.markdown(f"""<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}

/* ── Transparent app shell ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* ── Glass card for the content column ── */
[data-testid="stAppViewContainer"] > .main .block-container {{
    background: rgba(8, 6, 4, 0.62);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(200, 149, 108, 0.18);
    padding: 2.5rem 3rem 3rem;
    margin-top: 1.5rem;
    position: relative;
    z-index: 1;
}}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {{ color: #e8e0d5 !important; }}
p, span, div, label, li  {{ color: #cdc4b8 !important; }}
.stMarkdown p            {{ color: #cdc4b8 !important; }}

/* ── Section divider ── */
hr {{ border-color: rgba(200, 149, 108, 0.25) !important; margin: 1.8rem 0 !important; }}

/* ── Text input / area ── */
.stTextArea textarea {{
    background: rgba(255,255,255,0.06) !important;
    color: #e8e0d5 !important;
    border: 1px solid rgba(200,149,108,0.30) !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    resize: vertical;
}}
.stTextArea textarea:focus {{
    border-color: rgba(200,149,108,0.75) !important;
    box-shadow: 0 0 0 3px rgba(200,149,108,0.12) !important;
}}

/* ── Button ── */
.stButton > button {{
    background: rgba(200,149,108,0.16) !important;
    color: #e8e0d5 !important;
    border: 1px solid rgba(200,149,108,0.45) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.18s ease !important;
}}
.stButton > button:hover {{
    background: rgba(200,149,108,0.30) !important;
    border-color: {ACCENT} !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(200,149,108,0.20) !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.1rem !important;
}}
[data-testid="metric-container"] label,
[data-testid="metric-container"] div {{
    color: #cdc4b8 !important;
}}

/* ── Select box ── */
.stSelectbox > div > div {{
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(200,149,108,0.30) !important;
    color: #e8e0d5 !important;
    border-radius: 8px !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
</style>""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading dataset…")
def _load_df():
    tp = DATA_DIR / "tweets.txt"
    ep = DATA_DIR / "emoji.txt"
    if not tp.exists():
        return None
    tweets = [l.strip() for l in tp.read_text().splitlines() if l.strip()]
    emojis = [l.strip() for l in ep.read_text().splitlines() if l.strip()]
    n = min(len(tweets), len(emojis))
    df = pd.DataFrame({"text": tweets[:n], "label": emojis[:n]})
    df["cleaned"] = df["text"].apply(clean_tweet)
    df = df.drop_duplicates(subset=["cleaned"])
    df = df[df["cleaned"].str.len() > 0]
    return df


@st.cache_data
def _load_results():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df["accuracy"].isna().all():
        return None
    return df


model      = _load_model()
df         = _load_df()
results_df = _load_results()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:0.5rem 0 0.2rem;">
  <h1 style="font-size:2.6rem;font-weight:700;letter-spacing:0.04em;
             background:linear-gradient(100deg,{ACCENT},#e8e0d5 50%,{ACCENT});
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin-bottom:0.3rem;">
    🌙 Emoji Prediction
  </h1>
  <p style="color:rgba(205,196,184,0.65)!important;font-size:0.95rem;">
    UCI Datathon 2026 &nbsp;·&nbsp; TF-IDF + Linear SVM &nbsp;·&nbsp; 10-class emoji classification
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Predict")

col_input, col_result = st.columns([1.3, 1], gap="large")

with col_input:
    tweet_input = st.text_area(
        "Tweet text",
        placeholder="e.g. Just had the most amazing coffee this morning...",
        height=130,
        label_visibility="collapsed",
    )
    predict_btn = st.button("Predict Emoji →", use_container_width=True)

with col_result:
    if predict_btn and tweet_input.strip():
        if model is None:
            st.warning("Model not found. Run `python src/train.py` from the repo root first.")
        else:
            cleaned = clean_tweet(tweet_input)
            top3    = predict_top_k(cleaned, model, k=3)
            top_label, top_score = top3[0]
            top_emoji = EMOJI_MAP.get(top_label, "❓")

            # Score bar: normalise SVM decision scores to 0-1 for display
            scores_arr = np.array([s for _, s in top3])
            scores_norm = (scores_arr - scores_arr.min())
            if scores_norm.max() > 0:
                scores_norm = scores_norm / scores_norm.max()

            bars_html = "".join(
                f"""<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
                     <span style="font-size:1.2rem;width:28px;">{EMOJI_MAP.get(lbl,'❓')}</span>
                     <span style="color:#cdc4b8;font-size:0.82rem;width:90px;">{lbl}</span>
                     <div style="flex:1;background:rgba(255,255,255,0.07);
                                 border-radius:4px;height:10px;overflow:hidden;">
                       <div style="width:{norm*100:.0f}%;height:100%;
                                   background:linear-gradient(90deg,{ACCENT},#e8d5c5);
                                   border-radius:4px;"></div>
                     </div>
                   </div>"""
                for (lbl, _), norm in zip(top3, scores_norm)
            )

            st.markdown(f"""
            <div style="background:rgba(200,149,108,0.10);
                        border:1px solid rgba(200,149,108,0.35);
                        border-radius:14px;padding:1.2rem 1.4rem;">
              <div style="text-align:center;margin-bottom:0.8rem;">
                <span style="font-size:4rem;line-height:1;">{top_emoji}</span><br>
                <span style="font-size:1.15rem;font-weight:600;
                             color:#e8e0d5;">{top_label}</span>
              </div>
              <hr style="border-color:rgba(200,149,108,0.2);margin:0.6rem 0;">
              <div style="font-size:0.78rem;color:rgba(205,196,184,0.6);
                          margin-bottom:6px;">Top 3 predictions</div>
              {bars_html}
            </div>
            """, unsafe_allow_html=True)

    elif predict_btn:
        st.info("Enter some tweet text above.")
    else:
        st.markdown("""
        <div style="height:160px;display:flex;align-items:center;
                    justify-content:center;
                    border:1px dashed rgba(200,149,108,0.25);
                    border-radius:14px;color:rgba(205,196,184,0.35);
                    font-size:0.9rem;">
          prediction will appear here
        </div>""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Data Explorer")

if df is None:
    st.warning("Dataset not found. Ensure `data/tweets.txt` and `data/emoji.txt` are present.")
else:
    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tweets (deduped)", f"{len(df):,}")
    m2.metric("Emoji classes",    df["label"].nunique())
    m3.metric("Avg length",       f"{df['cleaned'].str.len().mean():.0f} chars")
    m4.metric("Majority class",   df["label"].value_counts().index[0])

    st.markdown("<br>", unsafe_allow_html=True)

    col_dist, col_words = st.columns(2, gap="large")

    # ── Class distribution ───────────────────────────────────────────────────
    with col_dist:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["display"] = counts["label"].map(lambda x: f"{EMOJI_MAP.get(x,'')} {x}")

        fig_dist = px.bar(
            counts, x="count", y="display", orientation="h",
            color="count",
            color_continuous_scale=[[0, "#1a0a05"], [0.45, "#8b4513"], [1, ACCENT]],
            template="plotly_dark",
            labels={"count": "Tweets", "display": ""},
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=30, b=10),
            font=dict(color="#cdc4b8"),
            yaxis=dict(autorange="reversed"),
            title=dict(text="Class Distribution", font=dict(size=14, color="#e8e0d5")),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Top words per class ──────────────────────────────────────────────────
    with col_words:
        selected = st.selectbox(
            "Inspect class:",
            options=sorted(df["label"].unique()),
            format_func=lambda x: f"{EMOJI_MAP.get(x,'❓')}  {x}",
        )
        _noise = {
            "i", "the", "a", "to", "and", "you", "my", "in", "is", "it",
            "for", "so", "me", "that", "was", "this", "of", "on", "at",
            "be", "are", "have", "with", "they", "we", "do", "not", "no",
            "he", "she", "what", "your", "an", "but", "up", "all", "had",
            "been", "from", "got", "get", "just", "like", "bet",
            "starbucks", "walmart", "mcdonalds", "dominos", "amp", "rt", "ll",
        }
        subset_words = " ".join(df[df["label"] == selected]["cleaned"]).split()
        freq = Counter(w for w in subset_words if len(w) > 2 and w not in _noise)
        top_words = freq.most_common(15)

        if top_words:
            wdf = pd.DataFrame(top_words, columns=["word", "freq"])
            fig_words = px.bar(
                wdf, x="freq", y="word", orientation="h",
                color="freq",
                color_continuous_scale=[[0, "#1a0a05"], [1, ACCENT]],
                template="plotly_dark",
                labels={"freq": "Frequency", "word": ""},
            )
            fig_words.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=30, b=10),
                font=dict(color="#cdc4b8"),
                yaxis=dict(autorange="reversed"),
                title=dict(
                    text=f"Top Words · {EMOJI_MAP.get(selected,'')} {selected}",
                    font=dict(size=14, color="#e8e0d5"),
                ),
            )
            st.plotly_chart(fig_words, use_container_width=True)

    # ── Tweet-length distribution ────────────────────────────────────────────
    with st.expander("Tweet length by class"):
        df_len = df.copy()
        df_len["tweet_length"] = df_len["cleaned"].str.len()
        df_len["display"] = df_len["label"].map(lambda x: f"{EMOJI_MAP.get(x,'')} {x}")

        fig_box = px.box(
            df_len, x="display", y="tweet_length",
            color="display",
            color_discrete_sequence=px.colors.sequential.Oranges[2:],
            template="plotly_dark",
            labels={"display": "", "tweet_length": "Length (chars)"},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color="#cdc4b8"),
            margin=dict(t=20, b=20),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Model Performance")

if results_df is not None:
    # ── Metrics table ─────────────────────────────────────────────────────────
    display = results_df.copy()
    for col in ["accuracy", "f1_macro", "f1_weighted"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    numeric = results_df.copy()
    for col in ["accuracy", "f1_macro", "f1_weighted"]:
        if col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    melted = numeric.melt(
        id_vars=["model"],
        value_vars=[c for c in ["accuracy", "f1_macro", "f1_weighted"] if c in numeric.columns],
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])

    if not melted.empty:
        fig_perf = px.bar(
            melted, x="model", y="value", color="metric", barmode="group",
            template="plotly_dark",
            color_discrete_sequence=[ACCENT, "#8b4513", "#e8d5c5"],
            labels={"value": "Score", "model": "", "metric": "Metric"},
        )
        fig_perf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdc4b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cdc4b8")),
            margin=dict(t=20, b=10),
            xaxis=dict(tickangle=-10),
        )
        fig_perf.update_traces(marker_line_width=0)
        st.plotly_chart(fig_perf, use_container_width=True)

else:
    st.info(
        "No results yet. Run `python src/train.py` from the repo root to train all models "
        "and populate `results/model_comparison.csv`."
    )
    # Show expected improvement preview
    st.markdown("**Expected improvement over the notebook baseline:**")
    preview = pd.DataFrame([
        {"model": "Naive Bayes (baseline)",                  "accuracy": "31.2%", "f1_macro": "0.211", "note": "no class weighting"},
        {"model": "Logistic Regression + balanced",          "accuracy": "~33%",  "f1_macro": "~0.28", "note": "class_weight='balanced'"},
        {"model": "Linear SVM + balanced + char n-grams",   "accuracy": "~36%",  "f1_macro": "~0.32", "note": "best expected"},
    ])
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption(
        "Key changes: `class_weight='balanced'` prevents majority-class dominance (sob); "
        "character n-grams (3-5) capture informal patterns like *lmao*, *omg*, *haha* "
        "that word tokenisation misses."
    )
