import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Deepfake Detection Research Dashboard",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600;700&family=Fjalla+One&display=swap');

* { font-family: 'Fjalla One', sans-serif; }
h1, h2, h3, h4 { font-family: 'Roboto Mono', monospace !important; }

html, body, .stApp { background-color: #F8F9FA; color: #1A1A2E; }

.block-container {
    padding-top: 2.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 100% !important;
}

p, li, td, th, label, span, div {
    font-family: 'Fjalla One', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #1A1A2E;
}

.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 24px 20px;
    text-align: center;
}
.metric-number {
    font-family: 'Roboto Mono', monospace !important;
    font-size: 2rem;
    font-weight: 700;
    color: #2C5F8A;
    line-height: 1.2;
}
.metric-label {
    font-family: 'Fjalla One', sans-serif;
    font-size: 0.85rem;
    color: #6B7280;
    margin-top: 8px;
}

.abstract-box {
    background: #E8F0F7;
    border-left: 4px solid #2C5F8A;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
    font-family: 'Fjalla One', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #1A1A2E;
    margin: 16px 0;
}

.warning-box {
    background: #FFF3CD;
    border-left: 4px solid #F0A500;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Fjalla One', sans-serif;
    font-size: 0.95rem;
    color: #1A1A2E;
    line-height: 1.6;
    margin: 16px 0;
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Fjalla One', sans-serif;
    font-size: 0.9rem;
}
.styled-table th {
    background-color: #2C5F8A;
    color: white;
    font-family: 'Roboto Mono', monospace;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
}
.styled-table tr:nth-child(even) td { background-color: #F8F9FA; }
.styled-table tr:nth-child(odd) td  { background-color: #FFFFFF; }
.styled-table td { padding: 10px 16px; color: #1A1A2E; }
.table-container {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IN_DIST_AUC  = 0.968
RANDOM_AUC   = 0.5
METHODS      = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
BAR_PALETTE  = ["#2C5F8A", "#5B9BD5", "#E8643C", "#F0A500"]
RED_BAR      = "#E74C3C"
CHART_CONFIG = {"modeBarButtonsToKeep": ["toImage"], "displaylogo": False}

TRACE_COLORS = {
    ("Baseline", "3ep"):  "#5B9BD5",
    ("Baseline", "10ep"): "#2C5F8A",
    ("Dual",     "3ep"):  "#F0A500",
    ("Dual",     "10ep"): "#E8643C",
}

# ── Hardcoded data ────────────────────────────────────────────────────────────
cross_df = pd.DataFrame([
    {"held_out": "Deepfakes",      "model": "Baseline", "epochs": "3ep",  "auc": 0.878},
    {"held_out": "Deepfakes",      "model": "Baseline", "epochs": "10ep", "auc": 0.929},
    {"held_out": "Deepfakes",      "model": "Dual",     "epochs": "3ep",  "auc": 0.872},
    {"held_out": "Deepfakes",      "model": "Dual",     "epochs": "10ep", "auc": 0.927},
    {"held_out": "Face2Face",      "model": "Baseline", "epochs": "3ep",  "auc": 0.781},
    {"held_out": "Face2Face",      "model": "Baseline", "epochs": "10ep", "auc": 0.745},
    {"held_out": "Face2Face",      "model": "Dual",     "epochs": "3ep",  "auc": 0.801},
    {"held_out": "Face2Face",      "model": "Dual",     "epochs": "10ep", "auc": 0.767},
    {"held_out": "FaceSwap",       "model": "Baseline", "epochs": "3ep",  "auc": 0.469},
    {"held_out": "FaceSwap",       "model": "Baseline", "epochs": "10ep", "auc": 0.514},
    {"held_out": "FaceSwap",       "model": "Dual",     "epochs": "3ep",  "auc": 0.453},
    {"held_out": "FaceSwap",       "model": "Dual",     "epochs": "10ep", "auc": 0.435},
    {"held_out": "NeuralTextures", "model": "Baseline", "epochs": "3ep",  "auc": 0.720},
    {"held_out": "NeuralTextures", "model": "Baseline", "epochs": "10ep", "auc": 0.747},
    {"held_out": "NeuralTextures", "model": "Dual",     "epochs": "3ep",  "auc": 0.746},
    {"held_out": "NeuralTextures", "model": "Dual",     "epochs": "10ep", "auc": 0.724},
])

conf_df = pd.DataFrame([
    {"model": "Baseline", "group": "FaceSwap (fake)", "prob": 0.062},
    {"model": "Baseline", "group": "Real",            "prob": 0.085},
    {"model": "Dual",     "group": "FaceSwap (fake)", "prob": 0.134},
    {"model": "Dual",     "group": "Real",            "prob": 0.211},
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def style_fig(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Roboto Mono, monospace", color="#1A1A2E"),
        title_font=dict(size=14, color="#1A1A2E", family="Roboto Mono, monospace"),
        xaxis=dict(
            gridcolor="#E5E5E5", gridwidth=1, linecolor="#CCCCCC",
            tickfont=dict(color="#1A1A2E", size=12, family="Roboto Mono, monospace"),
            title_font=dict(color="#1A1A2E", size=13, family="Roboto Mono, monospace"),
        ),
        yaxis=dict(
            gridcolor="#E5E5E5", gridwidth=1, linecolor="#CCCCCC",
            tickfont=dict(color="#1A1A2E", size=12, family="Roboto Mono, monospace"),
            title_font=dict(color="#1A1A2E", size=13, family="Roboto Mono, monospace"),
        ),
        legend=dict(font=dict(color="#1A1A2E", size=12, family="Roboto Mono, monospace")),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


def add_auc_refs(fig):
    fig.add_hline(
        y=RANDOM_AUC, line_dash="dash", line_color="#E8643C", line_width=1.5,
        annotation_text="Random chance (0.5)", annotation_position="bottom right",
        annotation_font=dict(color="#E8643C", size=11),
    )
    fig.add_hline(
        y=IN_DIST_AUC, line_dash="dash", line_color="#27ae60", line_width=1.5,
        annotation_text="In-distribution UB (0.968)", annotation_position="top right",
        annotation_font=dict(color="#27ae60", size=11),
    )
    return fig


def df_to_html(df):
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for _, row in df.iterrows()
    )
    return (
        '<div class="table-container">'
        '<table class="styled-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


def metric_card(number, label):
    return (
        '<div class="metric-card">'
        f'<div class="metric-number">{number}</div>'
        f'<div class="metric-label">{label}</div>'
        "</div>"
    )


def section_header(title):
    st.markdown(
        f'<h2 style="font-family:\'Roboto Mono\',monospace; color:#2C5F8A; '
        f'font-size:1.9rem; font-weight:700; border-bottom:2px solid #2C5F8A; '
        f'padding-bottom:10px; margin-bottom:20px;">{title}</h2>',
        unsafe_allow_html=True,
    )


def hr():
    st.markdown(
        "<hr style='border:1px solid #E5E5E5; margin:40px 0'>",
        unsafe_allow_html=True,
    )


# ── Overview ──────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-family:\'Roboto Mono\',monospace; font-size:2.6rem; font-weight:700; '
    'color:#1A1A2E; margin-bottom:4px;">'
    "Cross-Manipulation Generalization in Deepfake Detection</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#6B7280; font-family:\'Roboto Mono\',monospace; '
    'font-size:0.85rem; margin-bottom:24px;">INDENG 242B, Spring 2026</p>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(metric_card("0.968", "In-Distribution AUC"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("0.734", "Best Cross-Manipulation AUC"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("95.4%", "FaceSwap Miss Rate"), unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
Deepfake detectors trained on known manipulation methods routinely fail when tested on unseen ones.
We investigate whether frequency-domain features improve cross-manipulation generalization using the
FaceForensics++ C23 dataset and a leave-one-manipulation-out protocol across four methods: Deepfakes,
Face2Face, FaceSwap, and NeuralTextures. We compare an EfficientNet-B0 baseline against a dual
spatial-frequency model with an FFT log-magnitude branch, and further run experiments with stronger
augmentation plus focal loss, a frequency-only ablation, and an FFT-vs-DCT comparison. The
in-distribution AUC reaches 0.968, but cross-manipulation AUC averages only 0.713-0.734. FaceSwap
is a systematic failure across all configurations, confirmed through confidence inversion analysis.
Frequency features alone are insufficient; DCT shows discriminative signal on some methods but
collapses on FaceSwap. Cross-manipulation generalization in deepfake detection remains an open problem.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style=\"font-family:'Roboto Mono',monospace; font-size:0.85rem; color:#666666; margin-top:16px;\">"
    "Munazza Nadir &nbsp;&middot;&nbsp; Vriddhi Mittal &nbsp;&middot;&nbsp; "
    "Aarohi Zade &nbsp;&middot;&nbsp; Eesha Danish &nbsp;&middot;&nbsp; "
    "<a href='https://github.com/MunazzaNadir/deepfake-detection' style='color:#2C5F8A;'>GitHub &#8594;</a>"
    "</p>",
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    section_header("What is Cross-Manipulation Generalization?")
    st.markdown(
        "Cross-manipulation generalization measures whether a deepfake detector trained on "
        "certain fake methods can also detect manipulation types it has never encountered. "
        "Most detectors perform well when tested on the same method they trained on, but "
        "performance drops sharply when the manipulation type changes -- revealing whether "
        "a model learned a general notion of fakeness or just memorized specific artifacts."
    )
with right:
    section_header("The Leave-One-Out Protocol")
    st.markdown(
        "Each experiment trains on three of the four manipulation methods and tests on the "
        "fourth. This is repeated for all four choices of held-out method, giving a full "
        "picture of how each technique behaves as an unseen test case. It mirrors a realistic "
        "deployment scenario where a detector encounters a new manipulation type it was not "
        "trained on."
    )

hr()

# ── Cross-Manipulation Results ────────────────────────────────────────────────
section_header("Cross-Manipulation Results")

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_models = st.multiselect(
        "Model", options=["Baseline", "Dual"], default=["Baseline", "Dual"]
    )
with col_f2:
    selected_epochs = st.multiselect(
        "Training epochs", options=["3ep", "10ep"], default=["10ep"]
    )

filtered = cross_df[
    cross_df["model"].isin(selected_models) &
    cross_df["epochs"].isin(selected_epochs)
].copy()

if filtered.empty:
    st.warning("No results match the current filters. Select at least one model and one epoch setting.")
else:
    fig_cross = go.Figure()
    for (model, epoch), grp in filtered.groupby(["model", "epochs"]):
        grp = grp.set_index("held_out").reindex(METHODS).reset_index()
        normal = TRACE_COLORS[(model, epoch)]
        colors = [RED_BAR if v < RANDOM_AUC else normal for v in grp["auc"]]
        fig_cross.add_trace(go.Bar(
            name=f"{model} {epoch}",
            x=grp["held_out"],
            y=grp["auc"],
            marker_color=colors,
            text=[f"{v:.3f}" for v in grp["auc"]],
            textposition="outside",
        ))

    add_auc_refs(fig_cross)
    fig_cross.update_layout(
        barmode="group",
        yaxis=dict(title="AUC", range=[0, 1.12]),
        xaxis_title="Held-Out Method",
        legend_title="Model / Epochs",
        height=460,
    )
    style_fig(fig_cross)
    st.plotly_chart(fig_cross, use_container_width=True, config=CHART_CONFIG)
    st.caption("Bars shaded red fall below random chance (AUC < 0.5).")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    display = (
        filtered[["model", "epochs", "held_out", "auc"]]
        .rename(columns={
            "model": "Model", "epochs": "Epochs",
            "held_out": "Held-Out Method", "auc": "AUC",
        })
        .assign(AUC=lambda d: d["AUC"].round(3))
        .sort_values(["Held-Out Method", "Model", "Epochs"])
        .reset_index(drop=True)
    )
    st.markdown(df_to_html(display), unsafe_allow_html=True)

hr()

# ── FaceSwap Deep Dive ────────────────────────────────────────────────────────
section_header("FaceSwap Deep Dive")

st.markdown("""
<div class="warning-box">
<strong>Key Finding:</strong> Both models assign higher fake probability to real faces than to
FaceSwap-generated fakes -- a complete confidence inversion that directly explains the sub-random AUC.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "When tested on FaceSwap (held out from training), both models produce lower confidence "
    "scores for the actual fakes than for real images. The model is more suspicious of genuine "
    "faces than of the forgeries it was meant to detect. All values sit well below the 0.5 "
    "decision threshold, so nearly every FaceSwap fake is classified as Real."
)

fig_conf = go.Figure()
for i, (model, grp) in enumerate(conf_df.groupby("model")):
    fig_conf.add_trace(go.Bar(
        name=model,
        x=grp["group"],
        y=grp["prob"],
        marker_color=BAR_PALETTE[i],
        text=[f"{v:.3f}" for v in grp["prob"]],
        textposition="outside",
    ))
fig_conf.add_hline(
    y=0.5, line_dash="dash", line_color="gray",
    annotation_text="Decision threshold (0.5)", annotation_position="top left",
    annotation_font=dict(color="gray", size=11),
)
fig_conf.update_layout(
    title="Average Predicted Fake Probability by Group",
    yaxis=dict(title="Avg. Predicted Fake Probability", range=[0, 0.38]),
    barmode="group",
    height=380,
)
style_fig(fig_conf)
st.plotly_chart(fig_conf, use_container_width=True, config=CHART_CONFIG)
st.caption(
    "Both models score Real images higher than FaceSwap fakes. "
    "All values are far below the 0.5 decision threshold."
)

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
st.markdown(
    '<h3 style="font-family:\'Roboto Mono\',monospace; color:#2C5F8A; '
    'font-size:1.4rem; font-weight:600; margin-bottom:8px;">Threshold Tuning Simulator</h3>',
    unsafe_allow_html=True,
)
st.markdown(
    "The default threshold of 0.50 catches almost nothing. "
    "Lowering it recovers some FaceSwap detections, but at the cost of more false alarms on real faces."
)

threshold = st.slider(
    "Detection threshold", min_value=0.02, max_value=0.50, value=0.50, step=0.01,
    help="Lower values catch more FaceSwap fakes but also flag more real faces as fake",
)
t_frac = (threshold - 0.02) / (0.50 - 0.02)
recall = 0.251 + t_frac * (0.05 - 0.251)
f1     = 0.447 + t_frac * (0.09 - 0.447)

tc1, tc2 = st.columns(2)
with tc1:
    st.markdown(metric_card(f"{recall:.3f}", "FaceSwap Recall"), unsafe_allow_html=True)
with tc2:
    st.markdown(metric_card(f"{f1:.3f}", "F1 Score"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
if threshold <= 0.10:
    st.info(
        f"At threshold {threshold:.2f}: catching {recall:.1%} of FaceSwap fakes, "
        "but with significantly more false alarms on real faces."
    )
elif threshold >= 0.40:
    st.info(f"At threshold {threshold:.2f}: almost no FaceSwap fakes are detected (recall = {recall:.1%}).")
else:
    st.info(f"At threshold {threshold:.2f}: recall = {recall:.3f}, F1 = {f1:.3f}.")

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
st.markdown(
    '<h3 style="font-family:\'Roboto Mono\',monospace; color:#2C5F8A; '
    'font-size:1.4rem; font-weight:600; margin-bottom:8px;">Does More FaceSwap Training Data Help?</h3>',
    unsafe_allow_html=True,
)
st.markdown(
    "We tested whether the problem is simply insufficient FaceSwap exposure during training. "
    "Doubling the training data made performance worse, not better."
)

scaling_df = pd.DataFrame([
    {"condition": "10k samples",                   "auc": 0.514},
    {"condition": "20k samples",                   "auc": 0.488},
    {"condition": "In-distribution (upper bound)", "auc": 0.968},
])
fig_scaling = go.Figure(go.Bar(
    x=scaling_df["condition"],
    y=scaling_df["auc"],
    marker_color=[BAR_PALETTE[0], BAR_PALETTE[1], "#27ae60"],
    text=[f"{v:.3f}" for v in scaling_df["auc"]],
    textposition="outside",
))
add_auc_refs(fig_scaling)
fig_scaling.update_layout(
    yaxis=dict(title="Test AUC (FaceSwap held out)", range=[0, 1.12]),
    height=360,
)
style_fig(fig_scaling)
st.plotly_chart(fig_scaling, use_container_width=True, config=CHART_CONFIG)
st.caption(
    "Scaling from 10k to 20k samples drops AUC from 0.514 to 0.488. "
    "More data increases the risk of over-fitting to the training-set distribution of FaceSwap, "
    "which appears to differ from how FaceSwap looks at test time."
)

hr()

# ── Experiment Comparison ─────────────────────────────────────────────────────
section_header("Experiment Comparison")

comparison = st.radio(
    "Select comparison",
    options=[
        "Baseline vs Dual model",
        "Baseline vs Focal + Augmentation",
        "FFT vs DCT (frequency-only)",
    ],
    horizontal=True,
)

if comparison == "Baseline vs Dual model":
    vals_a = {"Deepfakes": 0.929, "Face2Face": 0.745, "FaceSwap": 0.514, "NeuralTextures": 0.747}
    vals_b = {"Deepfakes": 0.927, "Face2Face": 0.767, "FaceSwap": 0.435, "NeuralTextures": 0.724}
    label_a, label_b = "Baseline (10ep)", "Dual (10ep)"
    interpretation = (
        "The dual-branch frequency model does not consistently outperform the baseline: "
        "it gains slightly on Face2Face but underperforms on FaceSwap and NeuralTextures, "
        "suggesting that FFT-based features do not add generalizable signal across manipulation types."
    )
elif comparison == "Baseline vs Focal + Augmentation":
    vals_a = {"Deepfakes": 0.878, "Face2Face": 0.781, "FaceSwap": 0.469, "NeuralTextures": 0.720}
    vals_b = {"Deepfakes": 0.815, "Face2Face": 0.683, "FaceSwap": 0.495, "NeuralTextures": 0.620}
    label_a, label_b = "Baseline (3ep)", "Focal + Augmentation"
    interpretation = (
        "Focal loss and augmentation provide a small improvement on FaceSwap (+0.026) "
        "but hurt performance on every other held-out method, "
        "indicating they do not improve general cross-manipulation robustness."
    )
else:
    vals_a = {"Deepfakes": 0.507, "Face2Face": 0.502, "FaceSwap": 0.493, "NeuralTextures": 0.485}
    vals_b = {"Deepfakes": 0.700, "Face2Face": 0.570, "FaceSwap": 0.332, "NeuralTextures": 0.510}
    label_a, label_b = "FFT (frequency-only)", "DCT (frequency-only)"
    interpretation = (
        "DCT outperforms FFT on three of four held-out methods, "
        "but both frequency-only models fail badly on FaceSwap, "
        "confirming that frequency artifacts alone are not enough to generalize across manipulation types."
    )

avs    = [vals_a[m] for m in METHODS]
bvs    = [vals_b[m] for m in METHODS]
deltas = [b - a for a, b in zip(avs, bvs)]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name=label_a, x=METHODS, y=avs, marker_color=BAR_PALETTE[0],
    text=[f"{v:.3f}" for v in avs], textposition="inside", insidetextanchor="middle",
))
fig_comp.add_trace(go.Bar(
    name=label_b, x=METHODS, y=bvs, marker_color=BAR_PALETTE[2],
    text=[f"{v:.3f}" for v in bvs], textposition="inside", insidetextanchor="middle",
))
for method, delta, a, b in zip(METHODS, deltas, avs, bvs):
    sign = "+" if delta >= 0 else ""
    fig_comp.add_annotation(
        x=method, y=max(a, b) + 0.05,
        text=f"{sign}{delta:.3f}",
        showarrow=False,
        font=dict(
            size=12, family="Roboto Mono, monospace",
            color="#27ae60" if delta >= 0 else "#e74c3c",
        ),
    )
add_auc_refs(fig_comp)
fig_comp.update_layout(
    barmode="group",
    yaxis=dict(title="AUC", range=[0, 1.15]),
    xaxis_title="Held-Out Method",
    height=440,
)
style_fig(fig_comp)
st.plotly_chart(fig_comp, use_container_width=True, config=CHART_CONFIG)

st.markdown(
    f'<div class="abstract-box">{interpretation}</div>',
    unsafe_allow_html=True,
)
