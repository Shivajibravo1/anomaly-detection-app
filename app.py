# app.py — Streamlit Anomaly Detection (Robust Loader • Plotly • Safe Stats • AI Summary • Sanity Panel)

# --- NumPy 2.x compat shim (safe no-op on NumPy 1.x) ---
import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import csv
from io import StringIO
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# OpenAI SDK (Responses API) — requires openai>=1.0 in requirements.txt
from openai import OpenAI

st.set_page_config(page_title="Network Log Anomaly Detection", layout="wide")
st.title("🔍 Network Log Anomaly Detection")
st.caption("Robust loader, business charts, safe stats, downloads, and an OpenAI-generated executive summary.")

# --- Session state for run results & summaries ---
if "run_ready" not in st.session_state:
    st.session_state.run_ready = False
if "ctx" not in st.session_state:
    st.session_state.ctx = {}
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
POSSIBLE_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin1']
POSSIBLE_DELIMS = [',', ';', '\t', '|']

def robust_read(uploaded_file):
    """Read messy CSV/TSV/LOG with auto delimiter/encoding + skip bad lines."""
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    for enc in POSSIBLE_ENCODINGS:
        try:
            text = raw.decode(enc, errors='replace')
            try:
                dialect = csv.Sniffer().sniff(text[:20000])
                sep = dialect.delimiter
            except Exception:
                first = next((ln for ln in text.splitlines() if ln.strip()), '')
                sep = max(POSSIBLE_DELIMS, key=lambda d: len(first.split(d)))
            df = pd.read_csv(StringIO(text), sep=sep, engine='python', on_bad_lines='skip')
            if df.shape[1] == 1:
                df = pd.read_csv(StringIO(text), sep=r'[,\t;|]+', engine='python',
                                 on_bad_lines='skip', regex=True)
            return df, enc, sep
        except Exception:
            continue
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
    return df, 'unknown', 'auto'

def fig_bytes(fig) -> bytes:
    """Return PNG bytes from a Plotly figure (requires kaleido)."""
    return fig.to_image(format="png")

def build_business_summary(total, anomalies, pct, top_features=None, extra_notes=None):
    lines = []
    lines.append("# Anomaly Detection – Executive Summary\n")
    lines.append(f"- **Total records analyzed:** {total}")
    lines.append(f"- **Anomalies flagged:** {anomalies} (**{pct:.2f}%**)")
    if top_features is not None and not top_features.empty:
        top5 = top_features.head(5)
        feat_list = ", ".join([f"{i+1}. {name}" for i, name in enumerate(top5.index)])
        lines.append(f"- **Top indicative features:** {feat_list}")
    if extra_notes:
        lines.append(f"- **Notes:** {extra_notes}")
    lines.append("\n**Interpretation:** Lower (more negative) `anomaly_score` indicates a more suspicious record. Use distribution and trend charts to prioritize investigations.\n")
    return "\n".join(lines)

def make_ai_prompt(context: dict) -> str:
    """Construct an executive-ready prompt for OpenAI."""
    mode = context.get("mode")
    total = context.get("total")
    anomalies = context.get("anomalies")
    pct = context.get("pct")
    tf = context.get("top_features") or []
    tf_str = ", ".join(tf[:8]) if tf else "N/A"
    trends = "Yes" if context.get("has_trend") else "No"
    notes = context.get("notes") or ""
    return f"""
You are a senior cybersecurity analyst. Write a crisp, executive-ready summary (180-220 words) for business leaders.
Data context:
- Detection Mode: {mode}
- Total Records: {total}
- Anomalies Flagged: {anomalies} ({pct:.2f}%)
- Top Signals/Features: {tf_str}
- Time Trend Available: {trends}
- Analyst Notes: {notes}

Write in a confident, professional tone. Use **5 sections** with bold labels:

1) **Overview:** What was analyzed and how.
2) **Key Findings:** Quantify anomalies, notable spikes/trends, and signal drivers.
3) **Business Impact:** Implications for risk, operations, or compliance.
4) **Likely Causes:** Plausible explanations (e.g., misconfig, brute force, policy gaps).
5) **Immediate Mitigations & Next Steps:** 4–6 bullets, action-oriented and pragmatic.

Keep jargon minimal. Avoid code. No preambles or closings.
"""

def openai_summarize(api_key: str, model: str, prompt: str) -> str:
    """Call OpenAI Responses API and return plain text."""
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,  # e.g., "gpt-4o-mini"
        input=[{"role": "user", "content": prompt}],
    )
    text = getattr(resp, "output_text", None)
    if not text:
        try:
            text = resp.choices[0].message.content[0].text
        except Exception:
            text = "Unable to parse model output."
    return text

# --- Safe numeric summary helpers (prevents KeyError) ---
def safe_numeric_subset(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Return only existing numeric columns from cols; empty DF if none."""
    if not cols:
        return pd.DataFrame()
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    numeric = df[present].select_dtypes(include="number")
    return numeric if not numeric.empty else pd.DataFrame()

def concise_stats(df: pd.DataFrame, cols) -> pd.DataFrame:
    """describe().T + missing% on a safe subset (prevents KeyError)."""
    sub = safe_numeric_subset(df, cols)
    if sub.empty:
        return pd.DataFrame()
    stats = sub.describe().T
    stats["missing_%"] = (1 - sub.count() / len(sub)) * 100
    return stats

# ------------------------------------------------------------------------------------
# Sidebar: Upload & Preview
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Upload", help="Upload CSV/TSV/LOG. Auto-detects delimiter & encoding, skips bad lines.")
    up = st.file_uploader("Upload network log", type=["csv", "tsv", "log"],
                          help="Drag & drop or browse a log file from firewall/router/SIEM/etc.")
    preview_n = st.slider("Preview rows", 5, 100, 25, 5, help="How many rows of the raw data to preview.")

if not up:
    st.info("Please upload a file to begin.")
    st.stop()

try:
    df_raw, enc_used, sep_used = robust_read(up)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"Loaded: shape={df_raw.shape} • encoding={enc_used} • delimiter='{sep_used}'")
st.dataframe(df_raw.head(preview_n), use_container_width=True)

# ------------------------------------------------------------------------------------
# Normalize columns + detect time
# ------------------------------------------------------------------------------------
df = df_raw.copy()
df.columns = (df.columns.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(r'\s+', '_', regex=True)
              .str.replace(r'[^0-9a-z_]', '', regex=True))
st.caption(f"Detected columns: {list(df.columns)}")

event_time_dt = None
time_like_cols = [c for c in df.columns if any(k in c for k in ['time', 'date', 'timestamp'])]
for c in time_like_cols:
    dt = pd.to_datetime(df[c], errors='coerce')
    if dt.notna().any():
        event_time_dt = dt
        break

# ------------------------------------------------------------------------------------
# Quick stats selector (SAFE)
# ------------------------------------------------------------------------------------
numeric_options = sorted(df.select_dtypes(include="number").columns.tolist())

selected_num_cols = st.multiselect(
    "Columns to summarize (numeric only)",
    options=numeric_options,
    default=numeric_options[: min(6, len(numeric_options))],
    help="Pick numeric columns to see quick stats."
)

# Intersect selection with current numeric columns to avoid stale widget state
selected_num_cols = [c for c in selected_num_cols if c in numeric_options]

st.subheader("📋 Quick Numeric Summary")
stats_df = concise_stats(df, selected_num_cols)
if stats_df.empty:
    st.warning("No numeric columns available from the current selection to summarize.")
else:
    st.dataframe(stats_df, use_container_width=True)

# ------------------------------------------------------------------------------------
# Labels (optional)
# ------------------------------------------------------------------------------------
label_aliases = ['label', 'labels', 'class', 'attack', 'category', 'tag', 'malicious', 'target', 'outcome', 'is_anomaly']
autodetect = next((c for c in label_aliases if c in df.columns), None)

with st.sidebar:
    st.header("2) Labels (optional)", help="If your data has ground truth (e.g., BENIGN/ATTACK), pick it. Otherwise leave as None for unsupervised detection.")
    picked_label = st.selectbox(
        "Choose label column (optional):",
        ["<None>"] + list(df.columns),
        index=(1 + list(df.columns).index(autodetect)) if autodetect else 0,
        help="Select the column indicating normal vs malicious. If none, we’ll switch to unsupervised mode."
    )

if picked_label != "<None>":
    def to_anomaly(v):
        v = str(v).strip().upper()
        return 0 if v in {'BENIGN', '0', 'FALSE', 'NORMAL', 'CLEAN'} or 'BENIGN' in v else 1
    df['is_anomaly'] = df[picked_label].map(to_anomaly)
else:
    df['is_anomaly'] = np.nan

# ------------------------------------------------------------------------------------
# Preprocessing for modeling
# ------------------------------------------------------------------------------------
for c in time_like_cols:
    try:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    except:
        pass
for c in df.select_dtypes(include=['datetime64[ns]']).columns:
    df[c] = df[c].astype('int64') // 10**9

drop_like = ['ip', 'addr', 'mac', 'id', 'session', 'user', 'host', 'hostname', 'uri', 'path', 'src', 'dst']
to_drop = [c for c in df.columns if any(k in c for k in drop_like)]
df_model = df.drop(columns=to_drop, errors='ignore')

for c in df_model.columns:
    df_model[c] = pd.to_numeric(df_model[c], errors='coerce')
X = df_model.select_dtypes(include=['number']).fillna(0.0)

if X.empty or X.shape[1] == 0:
    st.error("No numeric features after preprocessing. Adjust drop rules or provide numeric columns.")
    st.stop()

st.write("**Modeling frame shape**:", X.shape)

# ------------------------------------------------------------------------------------
# Sidebar: Modeling controls
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("3) Modeling", help="Choose supervised (needs labels) or unsupervised. Tooltips explain each option.")
    supervised_possible = df['is_anomaly'].notna().any()
    mode = st.radio("Mode",
                    ["Supervised (RandomForest)", "Unsupervised (IsolationForest)"],
                    index=0 if supervised_possible else 1,
                    help="Supervised: uses labels to evaluate accuracy. Unsupervised: detects outliers without labels.")

    if mode == "Supervised (RandomForest)":
        n_estimators = st.slider("Trees (n_estimators)", 50, 500, 200, 50,
                                 help="More trees can improve stability (slower).")
        max_depth = st.slider("Max depth (0=None)", 0, 50, 0, 1,
                              help="Limit tree depth to reduce overfitting; 0 lets trees grow fully.")
        test_size = st.slider("Test size %", 10, 50, 20, 5,
                              help="Portion of data held out for testing.") / 100.0
        rs = st.number_input("Random state", 0, 10000, 42, 1,
                             help="Set for reproducible results.")
        run = st.button("Train Supervised Model", help="Train and show classification report, confusion matrix, and top features.")
    else:
        n_iso = st.slider("Trees (n_estimators)", 50, 600, 300, 50,
                          help="More trees can stabilize anomaly scoring.")
        contam_mode = st.radio("Contamination", ["auto", "manual"], index=0,
                               help="Expected anomaly fraction: 'auto' infers; 'manual' lets you pick.")
        contam = (st.slider("contamination", 0.001, 0.20, 0.05, 0.001,
                            help="Approximate fraction of anomalies.") if contam_mode == "manual" else 'auto')
        rs = st.number_input("Random state", 0, 10000, 42, 1,
                             help="Set for reproducible results.")
        run = st.button("Run Unsupervised Detection", help="Run IsolationForest to flag outliers without labels.")

# Prepare holders for outputs/downloads
png_buffers: Dict[str, bytes] = {}
business_summary_text = None
top_features_series = None  # populated in supervised path

# ------------------------------------------------------------------------------------
# Supervised path
# ------------------------------------------------------------------------------------
if mode == "Supervised (RandomForest)":
    if not supervised_possible:
        st.warning("No label provided/detected. Switch to Unsupervised.")
        st.stop()

    if run:
        y = df['is_anomaly'].astype(int)
        if y.nunique() < 2:
            st.error("Label column has only one class. Need at least two classes.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 0 else max_depth,
            random_state=rs,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        st.subheader("📊 Supervised Performance")
        st.code(classification_report(y_test, preds, digits=4), language="text")
        cm = confusion_matrix(y_test, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            st.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Feature importance chart
        try:
            top_features_series = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write("Top features (importance):")
            st.dataframe(top_features_series.head(20).to_frame("importance"))
            top10 = top_features_series.head(10).sort_values(ascending=True)
            fig = px.bar(top10, x=top10.values, y=top10.index, orientation="h",
                         title="Top 10 Features (Supervised)",
                         labels={"x": "Importance", "y": "Feature"})
            st.plotly_chart(fig, use_container_width=True)
            png_buffers["top_features_supervised.png"] = fig_bytes(fig)
        except Exception:
            pass

        # Business summary (classic)
        total = len(X_test)
        anomalies = int((preds == 1).sum())
        pct = (anomalies / total * 100) if total else 0
        business_summary_text = build_business_summary(total, anomalies, pct, top_features=top_features_series)

        # --- Mark app as ready and store context for AI summary ---
        st.session_state.run_ready = True
        st.session_state.ctx = {
            "mode": "Supervised (RandomForest)",
            "total": int(total),
            "anomalies": int(anomalies),
            "pct": float(pct),
            "top_features": list((top_features_series.head(8).index if top_features_series is not None else [])),
            "has_trend": event_time_dt is not None,
            "notes": "Charts exported; review flagged rows and high-importance signals with SecOps."
        }

# ------------------------------------------------------------------------------------
# Unsupervised path
# ------------------------------------------------------------------------------------
if mode == "Unsupervised (IsolationForest)" and run:
    iso = IsolationForest(
        n_estimators=n_iso,
        contamination=contam,
        random_state=rs,
        n_jobs=-1
    )
    iso.fit(X)
    df['anomaly_score'] = iso.decision_function(X)
    df['is_anomaly_pred'] = (iso.predict(X) == -1).astype(int)

    st.subheader("📈 Unsupervised Anomaly Summary")
    counts = df['is_anomaly_pred'].value_counts().rename({0: 'Normal', 1: 'Anomaly'})
    st.write(counts)

    total = len(df)
    anomalies = int(df['is_anomaly_pred'].sum())
    pct = (anomalies / total * 100) if total else 0
    st.markdown(f"**Summary:** {anomalies}/{total} rows flagged (**{pct:.2f}%**). Lower `anomaly_score` = more suspicious.")

    st.markdown("**Top 50 most suspicious rows (lowest scores):**")
    st.dataframe(df.sort_values('anomaly_score').head(50), use_container_width=True)

    flagged = df[df['is_anomaly_pred'] == 1]
    if not flagged.empty:
        st.download_button(
            "⬇️ Download anomalies (CSV)",
            data=flagged.to_csv(index=False).encode('utf-8'),
            file_name="flagged_anomalies.csv",
            mime="text/csv",
            help="Download only the rows flagged as anomalies."
        )
    else:
        st.info("No anomalies flagged with current settings.")

    # Charts
    counts_plot = df['is_anomaly_pred'].map({0: 'Normal', 1: 'Anomaly'}).value_counts()
    fig1 = px.bar(counts_plot, x=counts_plot.index, y=counts_plot.values,
                  title="Normal vs Anomaly (Count)",
                  labels={"x": "Class", "y": "Records"},
                  text=counts_plot.values)
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)
    png_buffers["anomaly_distribution.png"] = fig_bytes(fig1)

    fig2 = px.histogram(df, x="anomaly_score", nbins=40,
                        title="Anomaly Score Distribution",
                        labels={"anomaly_score": "anomaly_score (lower = more suspicious)", "count": "Frequency"})
    st.plotly_chart(fig2, use_container_width=True)
    png_buffers["anomaly_score_hist.png"] = fig_bytes(fig2)

    if event_time_dt is not None:
        trend = pd.DataFrame({
            'event_time': event_time_dt,
            'is_anomaly': df['is_anomaly_pred'].fillna(0).astype(int)
        }).dropna(subset=['event_time'])
        if not trend.empty:
            daily = trend.groupby(trend['event_time'].dt.floor('D'))['is_anomaly'].sum().reset_index()
            fig3 = px.line(daily, x='event_time', y='is_anomaly', markers=True,
                           title="Daily Anomalies",
                           labels={"event_time": "Day", "is_anomaly": "Anomaly Count"})
            st.plotly_chart(fig3, use_container_width=True)
            png_buffers["anomaly_trend_daily.png"] = fig_bytes(fig3)

    # Classic summary for unsupervised
    business_summary_text = build_business_summary(total, anomalies, pct, extra_notes="Unsupervised results (IsolationForest).")

    # --- Mark app as ready and store context for AI summary ---
    st.session_state.run_ready = True
    st.session_state.ctx = {
        "mode": "Unsupervised (IsolationForest)",
        "total": int(total),
        "anomalies": int(anomalies),
        "pct": float(pct),
        "top_features": [],
        "has_trend": event_time_dt is not None,
        "notes": "Unsupervised results; validate outliers and tune contamination if needed."
    }

# ------------------------------------------------------------------------------------
# SANITY PANEL (readiness & key figures)
# ------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("✅ Sanity Panel")
if st.session_state.run_ready and st.session_state.ctx:
    c = st.session_state.ctx
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mode", c.get("mode", "-"))
    col_b.metric("Total", f"{c.get('total', 0):,}")
    col_c.metric("Anomalies", f"{c.get('anomalies', 0):,}")
    col_d.metric("Anomaly Rate", f"{c.get('pct', 0.0)::.2f}%")
    st.success("Ready for AI summary and downloads.")
else:
    st.warning("Run a model first (Supervised or Unsupervised) to populate key figures.")

# ------------------------------------------------------------------------------------
# OpenAI Executive Summary (main page UI)
# ------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧠 Executive Summary (OpenAI)")

# Prefer secrets on Streamlit Cloud; fall back to UI
key_from_secret = st.secrets.get("OPENAI_API_KEY", None)
key_from_ui = st.text_input(
    "OpenAI API Key (optional if set in Secrets)",
    type="password",
    placeholder="sk-...",
    help="Used only to generate the summary. Not stored."
)
api_key = key_from_secret or key_from_ui

model_name = st.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="Pick a model for a crisp, board-ready summary."
)

gen_ai_col1, gen_ai_col2 = st.columns([1,3])
with gen_ai_col1:
    gen_ai_summary = st.button("Generate AI Summary")

if gen_ai_summary:
    if not api_key:
        st.error("Please enter your OpenAI API key (or set OPENAI_API_KEY in Secrets).")
    elif not st.session_state.run_ready:
        st.error("Run a model first (Supervised/Unsupervised), then generate the summary.")
    else:
        try:
            ctx = st.session_state.ctx
            prompt = make_ai_prompt(ctx)
            ai_text = openai_summarize(api_key, model_name, prompt)
            st.session_state.ai_summary = ai_text
            st.success("AI summary generated.")
        except Exception as e:
            st.error(f"OpenAI summary failed: {e}")

# Show AI summary + download if present
if st.session_state.ai_summary:
    st.markdown("#### 📄 AI-Generated Professional Summary")
    st.markdown(st.session_state.ai_summary)
    st.download_button(
        "⬇️ Download AI Summary (.md)",
        data=st.session_state.ai_summary.encode("utf-8"),
        file_name="ai_anomaly_summary.md",
        mime="text/markdown",
        help="Board-ready summary generated by the model."
    )

# ------------------------------------------------------------------------------------
# Downloads: classic summary + charts (only after a run)
# ------------------------------------------------------------------------------------
if st.session_state.run_ready and 'business_summary_text' in locals() and business_summary_text:
    colL, colR = st.columns(2)
    with colL:
        st.download_button(
            "⬇️ Download Business Summary (.md)",
            data=business_summary_text.encode("utf-8"),
            file_name="business_summary_classic.md",
            mime="text/markdown",
            help="One-page executive summary (non-AI)."
        )
    with colR:
        if 'png_buffers' in locals() and png_buffers:
            for name, data in png_buffers.items():
                st.download_button(
                    f"⬇️ Download {name}",
                    data=data,
                    file_name=name,
                    mime="image/png",
                    help="Save chart image for slides/reports."
                )

# ------------------------------------------------------------------------------------
# Help / UX notes
# ------------------------------------------------------------------------------------
with st.expander("What do these controls do?"):
    st.markdown("""
- **Upload**: Provide your exported log file. The app auto-detects delimiter/encoding and ignores corrupt rows.
- **Columns to summarize**: Pick numeric columns for quick descriptive stats; safe against missing columns.
- **Labels (optional)**: If your file has ground truth (e.g., `BENIGN`/`ATTACK`), pick it to train/evaluate a supervised model.
- **Supervised (RandomForest)**: Shows precision/recall/F1, confusion matrix, and top features driving decisions.
- **Unsupervised (IsolationForest)**: Flags outliers without labels. Tune **Contamination** to set expected anomaly rate.
- **Sanity Panel**: Confirms readiness and shows core figures (Total, Anomalies, Rate).
- **OpenAI Executive Summary**: Generates a polished, board-ready narrative with insights and a mitigation plan.
- **Downloads**: Anomalies CSV, PNG charts, and Markdown summaries.
""")

st.info("Tip: If too many rows are marked anomalous, reduce **contamination**. If none are flagged, increase it.")
