"""
UCI Datathon 2026 — Emoji Prediction Dashboard
Single-page Streamlit app with moonlit-ripple WebGL background.

Run from repo root:  streamlit run dashboard/app.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
ASSETS     = Path(__file__).parent / "assets"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR   = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_tweet
from predict import EMOJI_MAP, load_pipeline, predict_top_k

CLASSES = list(EMOJI_MAP.keys())
ACCENT  = "#c8956c"
FONT    = "'Styrene B', system-ui, -apple-system, sans-serif"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="tweeti",
    page_icon=str(ASSETS / "tweeti.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── WebGL background ─────────────────────────────────────────────────────────
def _inject_webgl():
    html_path = ASSETS / "moonlit-ripple.html"
    if not html_path.exists():
        return
    raw   = html_path.read_text()
    start = raw.find("<script>") + len("<script>")
    end   = raw.find("</script>", start)
    js    = raw[start:end]
    components.html(f"""<script>
(function(){{
  var d=window.parent.document;
  if(d.getElementById('moonlit-bg'))return;
  var w=d.createElement('div');
  w.id='moonlit-bg';
  w.style.cssText='position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;overflow:hidden;background:#0a0a0a;pointer-events:none;';
  var cv=d.createElement('canvas');cv.id='canvas';
  cv.style.cssText='display:block;width:100%;height:100%;';
  w.appendChild(cv);d.body.insertBefore(w,d.body.firstChild);
  var s=d.createElement('script');s.textContent={json.dumps(js)};
  d.head.appendChild(s);
}})();
</script>""", height=0, scrolling=False)

_inject_webgl()

# ── Favicon override (SVG) ───────────────────────────────────────────────────
def _inject_favicon():
    import base64
    svg_path = ASSETS / "tweeti.svg"
    if svg_path.exists():
        b64 = base64.b64encode(svg_path.read_bytes()).decode()
        st.markdown(
            f'<link rel="shortcut icon" href="data:image/svg+xml;base64,{b64}">',
            unsafe_allow_html=True,
        )

_inject_favicon()


# ── Theme ────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.cdnfonts.com/css/styrene-b') ;

html,body,*{{font-family:{FONT}!important;}}

#MainMenu,footer,[data-testid="stToolbar"]{{visibility:hidden;}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"]>.main,
[data-testid="stHeader"]{{background:transparent!important;}}

[data-testid="stAppViewContainer"]>.main .block-container{{
  background:rgba(8,6,4,0.65);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border-radius:18px;
  border:1px solid rgba(200,149,108,0.18);
  padding:1.6rem 3rem 3rem;
  margin-top:0.4rem;
  position:relative;z-index:1;
}}

h1,h2,h3,h4,h5,h6{{color:#e8e0d5!important;font-family:{FONT}!important;}}
p,span,div,label,li{{color:#cdc4b8!important;font-family:{FONT}!important;}}
.stMarkdown p{{color:#cdc4b8!important;}}

hr{{border-color:rgba(200,149,108,0.25)!important;margin:1.6rem 0!important;}}

.stTextArea textarea{{
  background:rgba(255,255,255,0.06)!important;color:#e8e0d5!important;
  border:1px solid rgba(200,149,108,0.30)!important;border-radius:10px!important;
  font-size:15px!important;font-family:{FONT}!important;
}}
.stTextArea textarea:focus{{
  border-color:rgba(200,149,108,0.75)!important;
  box-shadow:0 0 0 3px rgba(200,149,108,0.12)!important;
}}
.stTextInput input{{
  background:rgba(255,255,255,0.06)!important;color:#e8e0d5!important;
  border:1px solid rgba(200,149,108,0.30)!important;border-radius:10px!important;
  font-family:{FONT}!important;
}}

.stButton>button{{
  background:rgba(200,149,108,0.16)!important;color:#e8e0d5!important;
  border:1px solid rgba(200,149,108,0.45)!important;border-radius:10px!important;
  font-weight:600!important;letter-spacing:0.04em!important;
  padding:0.55rem 1.6rem!important;transition:all .18s ease!important;
  font-family:{FONT}!important;
}}
.stButton>button:hover{{
  background:rgba(200,149,108,0.30)!important;border-color:{ACCENT}!important;
}}

[data-testid="metric-container"]{{
  background:rgba(255,255,255,0.05)!important;
  border:1px solid rgba(200,149,108,0.20)!important;border-radius:12px!important;
  padding:0.9rem 1.1rem!important;
}}
[data-testid="metric-container"] label,[data-testid="metric-container"] div{{
  color:#cdc4b8!important;font-family:{FONT}!important;
}}

.stSelectbox>div>div{{
  background:rgba(255,255,255,0.07)!important;
  border-color:rgba(200,149,108,0.30)!important;
  color:#e8e0d5!important;border-radius:8px!important;
  font-family:{FONT}!important;
}}

[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th{{
  font-family:{FONT}!important;font-size:0.85rem!important;
}}

/* expander */
[data-testid="stExpander"]{{
  background:rgba(255,255,255,0.03)!important;
  border:1px solid rgba(200,149,108,0.15)!important;border-radius:10px!important;
}}
[data-testid="stExpander"] summary{{font-family:{FONT}!important;color:#e8e0d5!important;}}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _chart_layout(**kw):
    """Common Plotly layout for dark glass theme."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#cdc4b8", size=12),
        margin=dict(l=0, r=0, t=8, b=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h", x=0, y=-0.18,
            font=dict(family=FONT, size=11),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#cdc4b8",
                   tickfont=dict(family=FONT)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#cdc4b8",
                   tickfont=dict(family=FONT)),
    )
    base.update(kw)
    return base


def _label(text, sub=None):
    """Render a small chart label above a Plotly chart."""
    s = (f'<p style="margin:0 0 4px;font-size:0.82rem;font-weight:600;'
         f'letter-spacing:0.06em;text-transform:uppercase;'
         f'color:rgba(200,149,108,0.85);font-family:{FONT};">{text}</p>')
    if sub:
        s += (f'<p style="margin:0 0 4px;font-size:0.75rem;color:rgba(205,196,184,0.45);">'
              f'{sub}</p>')
    st.markdown(s, unsafe_allow_html=True)


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
    n  = min(len(tweets), len(emojis))
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
    d = pd.read_csv(path)
    return d if d["accuracy"].notna().any() else None


@st.cache_data
def _load_brain():
    fp = ROOT / "brain_features.npy"
    lp = ROOT / "balanced_emojis.npy"
    if not fp.exists():
        return None, None
    return (np.load(fp).astype(np.float32),
            np.load(lp, allow_pickle=True).astype(str))



model      = _load_model()
df         = _load_df()
results_df = _load_results()
brain_feat, brain_lbl = _load_brain()
LEFT_V = 10242


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
import base64 as _b64
_logo_b64 = _b64.b64encode((ASSETS / "tweeti.svg").read_bytes()).decode() if (ASSETS / "tweeti.svg").exists() else ""
st.markdown(f"""
<div style="text-align:center;padding:0.2rem 0 0.2rem;">
  <div style="display:flex;align-items:center;justify-content:center;gap:0.7rem;margin-bottom:0.15rem;">
    {"<img src='data:image/svg+xml;base64," + _logo_b64 + "' style='width:52px;height:52px;border-radius:50%;box-shadow:0 0 18px rgba(200,149,108,0.4);'>" if _logo_b64 else ""}
    <h1 style="font-size:2.6rem;font-weight:700;letter-spacing:0.02em;
               background:linear-gradient(100deg,{ACCENT},#e8e0d5 50%,{ACCENT});
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;margin:0;
               font-family:{FONT};">
      tweeti
    </h1>
  </div>
  <p style="color:rgba(205,196,184,0.60)!important;font-size:0.9rem;
            font-family:{FONT};">
    UCI Datathon 2026 &nbsp;·&nbsp; TF-IDF + Linear SVM
    &nbsp;·&nbsp; 10-class emoji classification
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Predict")
col_in, col_out = st.columns([1.3, 1], gap="large")

with col_in:
    tweet_input = st.text_area(
        "Tweet text",
        placeholder="e.g. Just had the most amazing coffee this morning...",
        height=130,
        label_visibility="collapsed",
    )
    predict_btn = st.button("Predict Emoji →", use_container_width=True)

with col_out:
    if predict_btn and tweet_input.strip():
        if model is None:
            st.warning("Model not found. Run `python src/train.py` first.")
        else:
            cleaned = clean_tweet(tweet_input)
            top3    = predict_top_k(cleaned, model, k=3)
            top_lbl, _ = top3[0]
            top_emoji  = EMOJI_MAP.get(top_lbl, "❓")

            scores_arr  = np.array([s for _, s in top3])
            scores_norm = scores_arr - scores_arr.min()
            if scores_norm.max() > 0:
                scores_norm /= scores_norm.max()

            bars = "".join(
                f"""<div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
                     <span style="font-size:1.15rem;width:26px;">{EMOJI_MAP.get(l,'❓')}</span>
                     <span style="color:#cdc4b8;font-size:0.80rem;width:88px;
                                  font-family:{FONT};">{l}</span>
                     <div style="flex:1;background:rgba(255,255,255,0.07);
                                 border-radius:4px;height:9px;overflow:hidden;">
                       <div style="width:{n*100:.0f}%;height:100%;
                                   background:linear-gradient(90deg,{ACCENT},#e8d5c5);
                                   border-radius:4px;"></div>
                     </div>
                   </div>"""
                for (l, _), n in zip(top3, scores_norm)
            )
            st.markdown(f"""
            <div style="background:rgba(200,149,108,0.10);
                        border:1px solid rgba(200,149,108,0.35);
                        border-radius:14px;padding:1.1rem 1.3rem;">
              <div style="text-align:center;margin-bottom:0.7rem;">
                <span style="font-size:3.8rem;line-height:1;">{top_emoji}</span><br>
                <span style="font-size:1.1rem;font-weight:600;color:#e8e0d5;
                             font-family:{FONT};">{top_lbl}</span>
              </div>
              <hr style="border-color:rgba(200,149,108,0.2);margin:0.5rem 0;">
              <div style="font-size:0.75rem;color:rgba(205,196,184,0.5);
                          margin-bottom:5px;font-family:{FONT};">Top 3</div>
              {bars}
            </div>""", unsafe_allow_html=True)
    elif predict_btn:
        st.info("Enter some tweet text above.")
    else:
        st.markdown("""
        <div style="height:155px;display:flex;align-items:center;
                    justify-content:center;
                    border:1px dashed rgba(200,149,108,0.22);
                    border-radius:14px;color:rgba(205,196,184,0.30);
                    font-size:0.88rem;">prediction will appear here</div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Data Explorer")

if df is None:
    st.warning("Dataset not found (`data/tweets.txt` / `data/emoji.txt`).")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tweets (deduped)", f"{len(df):,}")
    m2.metric("Emoji classes",    df["label"].nunique())
    m3.metric("Avg tweet length", f"{df['cleaned'].str.len().mean():.0f} chars")
    m4.metric("Majority class",   df["label"].value_counts().index[0])

    st.markdown("<br>", unsafe_allow_html=True)
    col_d, col_w = st.columns(2, gap="large")

    with col_d:
        counts = (df["label"].value_counts()
                  .reset_index()
                  .rename(columns={"label": "label", "count": "count"}))
        counts.columns = ["label", "count"]
        counts["display"] = counts["label"].map(
            lambda x: f"{EMOJI_MAP.get(x,'')} {x}")

        _label("Class distribution")
        fig = go.Figure(go.Bar(
            x=counts["count"], y=counts["display"],
            orientation="h", marker_line_width=0,
            marker=dict(
                color=counts["count"],
                colorscale=[[0,"#1a0a05"],[0.45,"#8b4513"],[1, ACCENT]],
            ),
        ))
        fig.update_layout(**_chart_layout(
            showlegend=False,
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            xaxis=dict(title="Tweets", gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=8),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_w:
        selected = st.selectbox(
            "Inspect class:",
            options=sorted(df["label"].unique()),
            format_func=lambda x: f"{EMOJI_MAP.get(x,'❓')}  {x}",
        )
        _noise = {
            "i","the","a","to","and","you","my","in","is","it","for","so",
            "me","that","was","this","of","on","at","be","are","have","with",
            "they","we","do","not","no","he","she","what","your","an","but",
            "up","all","had","been","from","got","get","just","like","bet",
            "starbucks","walmart","mcdonalds","dominos","amp","rt","ll",
        }
        words  = " ".join(df[df["label"] == selected]["cleaned"]).split()
        freq   = Counter(w for w in words if len(w) > 2 and w not in _noise)
        top_w  = freq.most_common(15)

        if top_w:
            wdf = pd.DataFrame(top_w, columns=["word", "freq"])
            _label(f"Top words · {EMOJI_MAP.get(selected,'')} {selected}")
            fig2 = go.Figure(go.Bar(
                x=wdf["freq"], y=wdf["word"],
                orientation="h", marker_line_width=0,
                marker=dict(
                    color=wdf["freq"],
                    colorscale=[[0,"#1a0a05"],[1, ACCENT]],
                ),
            ))
            fig2.update_layout(**_chart_layout(
                showlegend=False,
                yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)",
                           color="#cdc4b8", tickfont=dict(family=FONT)),
                xaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)",
                           color="#cdc4b8", tickfont=dict(family=FONT)),
                margin=dict(l=0, r=0, t=8, b=8),
            ))
            st.plotly_chart(fig2, use_container_width=True)

    # Tweet length box plot (no expander — avoids the icon-overlay issue)
    st.markdown("<br>", unsafe_allow_html=True)
    _label("Tweet length by class")
    df_len = df.copy()
    df_len["tweet_length"] = df_len["cleaned"].str.len()
    df_len["display"] = df_len["label"].map(lambda x: f"{EMOJI_MAP.get(x,'')} {x}")

    import plotly.express as px
    fig_box = px.box(
        df_len, x="display", y="tweet_length",
        color="display",
        color_discrete_sequence=px.colors.sequential.Oranges[2:],
        template="plotly_dark",
        labels={"display": "", "tweet_length": "Length (chars)"},
    )
    fig_box.update_layout(**_chart_layout(
        showlegend=False,
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)",
                   color="#cdc4b8", tickfont=dict(family=FONT)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                   color="#cdc4b8", tickfont=dict(family=FONT)),
        margin=dict(l=0, r=0, t=8, b=8),
    ))
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BRAIN SIGNALS · TRIBE v2
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Brain Signals · TRIBE v2")

if brain_feat is None:
    st.warning("Brain features not found at repo root (`brain_features.npy`).")
else:
    st.markdown(
        "**TRIBE v2** decodes emoji categories from cortical surface activation "
        "patterns (fMRI). Each of the 25 samples contains **20,484 features** — "
        "10,242 vertices per hemisphere on the fsaverage5 mesh. "
        "Both modalities predict the same 10 emoji classes: "
        "one from *language*, one from *brain activation*."
    )

    # ── Static brain activation image ────────────────────────────────────────
    static_img = ASSETS / "brain_activation.png"
    if static_img.exists():
        _label("Mean cortical activation per emoji class · left hemisphere lateral view")
        st.image(str(static_img), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sample activation explorer + class means ─────────────────────────────
    col_b1, col_b2 = st.columns(2, gap="large")

    with col_b1:
        sample_i  = st.selectbox(
            "Brain sample",
            options=list(range(len(brain_lbl))),
            format_func=lambda i:
                f"{i:02d} · {EMOJI_MAP.get(brain_lbl[i],'❓')} {brain_lbl[i]}",
            key="brain_sample",
        )
        left_act  = brain_feat[sample_i, :LEFT_V]
        right_act = brain_feat[sample_i, LEFT_V:]

        def _ds(arr, n=120):
            idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
            return arr[idx]

        _label(
            f"Activation profile · "
            f"{EMOJI_MAP.get(brain_lbl[sample_i],'❓')} {brain_lbl[sample_i]}",
            sub="Downsampled to 120 vertices per hemisphere",
        )
        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(
            x=list(range(120)), y=_ds(left_act), name="Left hemisphere",
            line=dict(color="#6baed6", width=1.5),
            fill="tozeroy", fillcolor="rgba(107,174,214,0.12)",
        ))
        fig_act.add_trace(go.Scatter(
            x=list(range(120)), y=_ds(right_act), name="Right hemisphere",
            line=dict(color="#fd8d3c", width=1.5),
            fill="tozeroy", fillcolor="rgba(253,141,60,0.10)",
        ))
        fig_act.add_hline(y=0,
            line=dict(color="rgba(255,255,255,0.12)", dash="dot"))
        fig_act.update_layout(**_chart_layout(
            xaxis=dict(title="Vertex (downsampled)", showgrid=False,
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            yaxis=dict(title="Activation", zeroline=False,
                       gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=55),
        ))
        st.plotly_chart(fig_act, use_container_width=True)

        s1, s2 = st.columns(2)
        s1.metric("Left mean",  f"{left_act.mean():.4f}")
        s2.metric("Right mean", f"{right_act.mean():.4f}")
        s1.metric("Left std",   f"{left_act.std():.4f}")
        s2.metric("Right std",  f"{right_act.std():.4f}")

    with col_b2:
        uniq_cls = sorted(set(brain_lbl.tolist()))
        rows = []
        for cls in uniq_cls:
            mask = brain_lbl == cls
            sub  = brain_feat[mask]
            rows.append({
                "label":    cls,
                "display":  f"{EMOJI_MAP.get(cls,'❓')} {cls}",
                "n":        int(mask.sum()),
                "mean_L":   float(sub[:, :LEFT_V].mean()),
                "mean_R":   float(sub[:, LEFT_V:].mean()),
                "lat_idx":  float(
                    (sub[:, LEFT_V:].mean() - sub[:, :LEFT_V].mean())
                    / (abs(sub.mean()) + 1e-9)
                ),
            })
        cs = pd.DataFrame(rows)

        _label("Mean cortical activation by class")
        fig_cls = go.Figure()
        fig_cls.add_trace(go.Bar(
            name="Left", x=cs["display"], y=cs["mean_L"],
            marker_color="#6baed6", marker_line_width=0,
        ))
        fig_cls.add_trace(go.Bar(
            name="Right", x=cs["display"], y=cs["mean_R"],
            marker_color="#fd8d3c", marker_line_width=0,
        ))
        fig_cls.update_layout(**_chart_layout(
            barmode="group",
            xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            yaxis=dict(title="Mean activation",
                       gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=55),
        ))
        st.plotly_chart(fig_cls, use_container_width=True)

        _label("Hemispheric lateralisation index",
               sub="Positive = right-dominant, negative = left-dominant")
        fig_lat = go.Figure(go.Bar(
            x=cs["display"], y=cs["lat_idx"],
            marker_line_width=0,
            marker=dict(
                color=cs["lat_idx"],
                colorscale=[[0,"#6baed6"],[0.5,"#555"],[1,"#fd8d3c"]],
            ),
        ))
        fig_lat.add_hline(y=0,
            line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
        fig_lat.update_layout(**_chart_layout(
            showlegend=False,
            xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=8),
        ))
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── 3D interactive viewer (Brain_Visualization.py) ───────────────────────
    viewer_path = Path(__file__).parent / "brain_surface.html"

    st.markdown("<br>", unsafe_allow_html=True)
    _label("Interactive 3D brain surface viewer")
    if viewer_path.exists():
        components.html(viewer_path.read_text(), height=820, scrolling=False)
    else:
        st.info(
            "Interactive viewer not yet generated. "
            "Run `python dashboard/Brain_Visualization.py` from the repo root "
            "to build `dashboard/brain_surface.html`."
        )

    # ── Text model vs TRIBE v2 ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Text model vs TRIBE v2")
    col_r1, col_r2 = st.columns([1.1, 1], gap="large")

    with col_r1:
        bd = pd.Series(brain_lbl).value_counts().rename("Brain %")
        bd = (bd / len(brain_lbl) * 100).round(1).reset_index()
        bd.columns = ["label", "Brain %"]

        if df is not None:
            td = (df["label"].value_counts() / len(df) * 100).round(1).reset_index()
            td.columns = ["label", "Tweet %"]
            dist = bd.merge(td, on="label", how="left")
        else:
            dist = bd

        dist["display"] = dist["label"].map(lambda x: f"{EMOJI_MAP.get(x,'❓')} {x}")

        _label("Class share · brain vs tweet dataset")
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Bar(
            name="Brain dataset (%)", x=dist["display"], y=dist["Brain %"],
            marker_color="#6baed6", marker_line_width=0,
        ))
        if "Tweet %" in dist.columns:
            fig_rel.add_trace(go.Bar(
                name="Tweet dataset (%)", x=dist["display"], y=dist["Tweet %"],
                marker_color=ACCENT, marker_line_width=0,
            ))
        fig_rel.update_layout(**_chart_layout(
            barmode="group",
            xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            yaxis=dict(title="Share (%)", gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=55),
        ))
        st.plotly_chart(fig_rel, use_container_width=True)

    with col_r2:
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid
            rgba(200,149,108,0.18);border-radius:12px;
            padding:1.2rem 1.4rem;font-size:0.88rem;line-height:1.7;
            font-family:{FONT};">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="border-bottom:1px solid rgba(200,149,108,0.25);">
  <th style="text-align:left;padding:4px 8px;color:#e8e0d5;"></th>
  <th style="text-align:left;padding:4px 8px;color:{ACCENT};">Text model</th>
  <th style="text-align:left;padding:4px 8px;color:#6baed6;">TRIBE v2</th>
</tr></thead>
<tbody style="color:#cdc4b8;">
<tr><td style="padding:4px 8px;color:#888;">Input</td>
    <td>Tweet text</td><td>fMRI cortical map</td></tr>
<tr><td style="padding:4px 8px;color:#888;">Features</td>
    <td>50K TF-IDF n-grams</td><td>20,484 cortical vertices</td></tr>
<tr><td style="padding:4px 8px;color:#888;">Signal</td>
    <td>Linguistic patterns</td><td>Neural activation</td></tr>
<tr><td style="padding:4px 8px;color:#888;">Samples</td>
    <td>140K tweets</td><td>25 brain scans</td></tr>
<tr><td style="padding:4px 8px;color:#888;">Classes</td>
    <td>10 emoji</td><td>10 emoji (5 represented)</td></tr>
</tbody></table>
<p style="margin-top:0.9rem;font-size:0.80rem;color:rgba(205,196,184,0.50);">
The text model learns <em>how people express</em> an emotion in language;
TRIBE v2 learns <em>how the brain activates</em> when perceiving it.
Shared classes with distinct neural signatures (sob, heart_eyes) are where
the two approaches are most complementary.
</p></div>""", unsafe_allow_html=True)

    # ── Live bridge ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _label("Text → emoji → brain sample lookup")
    bridge = st.text_input(
        "Text", placeholder="e.g. I love this so much!",
        key="bridge_input", label_visibility="collapsed",
    )
    if bridge.strip() and model is not None:
        top_b = predict_top_k(clean_tweet(bridge), model, k=1)
        p_lbl = top_b[0][0]
        p_em  = EMOJI_MAP.get(p_lbl, "❓")
        hits  = [i for i, l in enumerate(brain_lbl) if l == p_lbl]
        st.markdown(
            f"Text model predicts **{p_em} {p_lbl}** · "
            f"TRIBE v2 dataset has **{len(hits)} matching sample(s)**: "
            + (", ".join(f"sample {i:02d}" for i in hits) if hits
               else "none in the 25-sample set")
        )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Model Performance")

if results_df is not None:
    disp = results_df.copy()
    for c in ["accuracy", "f1_macro", "f1_weighted"]:
        if c in disp.columns:
            disp[c] = disp[c].map(
                lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    num = results_df.copy()
    for c in ["accuracy", "f1_macro", "f1_weighted"]:
        if c in num.columns:
            num[c] = pd.to_numeric(num[c], errors="coerce")

    melted = num.melt(
        id_vars=["model"],
        value_vars=[c for c in ["accuracy","f1_macro","f1_weighted"]
                    if c in num.columns],
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])

    if not melted.empty:
        _label("Accuracy · Macro F1 · Weighted F1 by model")
        fig_p = go.Figure()
        colors = [ACCENT, "#8b4513", "#e8d5c5"]
        for i, metric in enumerate(melted["metric"].unique()):
            sub = melted[melted["metric"] == metric]
            fig_p.add_trace(go.Bar(
                name=metric, x=sub["model"], y=sub["value"],
                marker_color=colors[i % len(colors)],
                marker_line_width=0,
            ))
        fig_p.update_layout(**_chart_layout(
            barmode="group",
            xaxis=dict(tickangle=-10, gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)",
                       color="#cdc4b8", tickfont=dict(family=FONT)),
            margin=dict(l=0, r=0, t=8, b=55),
        ))
        st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Run `python src/train.py` to populate results.")
    st.markdown("**Expected improvement over notebook baseline:**")
    st.dataframe(pd.DataFrame([
        {"model": "Naive Bayes (baseline)",              "accuracy": "31.2%",
         "f1_macro": "0.211", "note": "no class weighting"},
        {"model": "Logistic Regression + balanced",      "accuracy": "~33%",
         "f1_macro": "~0.28", "note": "class_weight='balanced'"},
        {"model": "Linear SVM + balanced + char n-grams","accuracy": "~36%",
         "f1_macro": "~0.32", "note": "best expected"},
    ]), use_container_width=True, hide_index=True)
