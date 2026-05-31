"""
OmniSim Dashboard — visualize conversation metrics and NLP comparison.

Usage:
    streamlit run dashboard.py
"""

import os
import re
import glob
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniSim Dashboard",
    page_icon="💬",
    layout="wide",
)

MODES = ["free", "static", "adaptive"]
MODE_COLORS = {"free": "#4C72B0", "static": "#55A868", "adaptive": "#C44E52"}
CHATS_ROOT = "chats/llama/imdb"
RESULTS_DIR = "results"

_STRATEGIES = r'(?:items|item_list|user_item_pairs|user_guest)'
_FILENAME_RE = re.compile(rf'^(?:{_STRATEGIES}-)?(.+?)-(\d+)-(\d+)-([01])-(\d+)\.txt$')
_REDIAL_RE = re.compile(r'^redial-(.+?)-(\d+)-(\d+)-([01])-(\d+)\.txt$')


# ─────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────
def parse_filename(fname):
    m = _REDIAL_RE.match(fname) or _FILENAME_RE.match(fname)
    if m:
        return {
            'item_id': m.group(1),
            'num_turns': int(m.group(2)),
            'rec_attempts': int(m.group(3)),
            'succeed': int(m.group(4)),
        }
    return None


def load_conv_stats(mode: str) -> pd.DataFrame | None:
    """Load per-conversation stats from the latest run folder."""
    base = os.path.join(CHATS_ROOT, mode, "user_item_pairs")
    if not os.path.isdir(base):
        return None
    runs = sorted(os.listdir(base), reverse=True)
    for run in runs:
        folder = os.path.join(base, run)
        rows = []
        for fname in os.listdir(folder):
            if not fname.endswith('.txt') or 'tmp' in fname:
                continue
            p = parse_filename(fname)
            if p:
                p['mode'] = mode
                rows.append(p)
        if rows:
            return pd.DataFrame(rows)
    return None


@st.cache_data
def load_all_conv_stats():
    dfs = []
    for m in MODES:
        df = load_conv_stats(m)
        if df is not None:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def load_nlp_metrics():
    rows = []
    for m in MODES:
        path = os.path.join(RESULTS_DIR, f"{m}_metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['mode'] = m
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _compute_nlp_metrics_live(folder: str, label: str, file_sample=None) -> dict | None:
    """Compute all NLP metrics directly from a chat folder (walks subdirs).
    If file_sample is provided, only read those specific files instead of walking folder.
    """
    import re as _re
    from collections import Counter as _Counter
    from math import comb as _comb

    def _tok(text):
        return _re.findall(r'\b[a-zA-Z]+\b', text.lower())

    utts = []
    file_list = file_sample if file_sample is not None else [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.endswith('.txt') and 'tmp' not in f
    ]
    for fpath in file_list:
        try:
            with open(fpath, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('Bot:') or line.startswith('User:'):
                        utts.append(line.split(':', 1)[1].strip())
        except Exception:
            pass
    if not utts:
        return None

    tokens = [t for u in utts for t in _tok(u)]
    MAX_TOKENS = 5_000
    tokens_ld = tokens[:MAX_TOKENS] if len(tokens) > MAX_TOKENS else tokens
    N = len(tokens)
    cnt = _Counter(tokens)

    # Distinct-1/2
    ng1 = [t for u in utts for t in _tok(u)]
    ng2 = [tuple(_tok(u)[i:i+2]) for u in utts for i in range(len(_tok(u))-1)]
    d1 = len(set(ng1)) / len(ng1) if ng1 else 0
    d2 = len(set(ng2)) / len(ng2) if ng2 else 0

    # HDD (D=42)
    D = 42
    hdd_val = sum(1.0 - _comb(N-n,D)/_comb(N,D) if n<=N-D else 1.0
                  for n in cnt.values()) if N >= D else 0.0

    # lexical_diversity metrics
    ld_vals = {}
    try:
        from lexical_diversity import lex_div as ld
        ld_vals = {
            'ttr':         ld.ttr(tokens_ld),
            'root_ttr':    ld.root_ttr(tokens_ld),
            'log_ttr':     ld.log_ttr(tokens_ld),
            'maas_ttr':    ld.maas_ttr(tokens_ld),
            'msttr':       ld.msttr(tokens_ld),
            'mattr':       ld.mattr(tokens_ld),
            'mtld':        ld.mtld(tokens_ld),
            'mtld_ma_w':   ld.mtld_ma_wrap(tokens_ld),
            'mtld_ma_bid': ld.mtld_ma_bid(tokens_ld),
        }
    except Exception:
        pass

    # Cosine Diversity (mean pairwise cosine distance between utterance embeddings)
    cosine_div = 0.0
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        sample = utts[:200]  # cap for speed
        embs = _model.encode(sample, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = np.dot(embs, embs.T)
        n = len(sample)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        cosine_div = float(1 - upper.mean())
    except Exception:
        pass

    # Item Mention Entropy
    item_entropy = 0.0
    try:
        import math as _math
        item_cnt = _Counter()
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith('.txt') and 'tmp' not in f:
                    parts = f.replace('.txt','').split('-')
                    if len(parts) >= 2:
                        item_cnt[parts[1]] += 1
        total_items = sum(item_cnt.values())
        if total_items > 0:
            item_entropy = -sum((c/total_items)*_math.log2(c/total_items) for c in item_cnt.values())
    except Exception:
        pass

    # success rate + average turns from filenames
    # pattern: ...-{turns}-{rec_attempts}-{succeed}-{timestamp}.txt
    import re as _re2
    _fn_pat = _re2.compile(r'-(\d+)-(\d+)-([01])-(\d{10,})\.txt$')
    stat_files = [os.path.basename(fp) for fp in file_list] if file_sample is not None else \
                 [f for _,_,fs in os.walk(folder) for f in fs if f.endswith('.txt') and 'tmp' not in f]
    total = len(stat_files)
    succ, turns_list = 0, []
    for f in stat_files:
        m2 = _fn_pat.search(f)
        if m2:
            turns_list.append(int(m2.group(1)))
            succ += int(m2.group(3))
    avg_turns = sum(turns_list)/len(turns_list) if turns_list else 0.0

    return {
        'label': label, 'folder': folder,
        'num_files': total,
        'num_utterances': len(utts),
        'success_rate': f"{succ/total:.0%}" if total else "—",
        'avg_turns': round(avg_turns, 1),
        'distinct_1': d1, 'distinct_2': d2,
        'hdd': hdd_val,
        'cosine_diversity': cosine_div,
        'item_entropy': item_entropy,
        **ld_vals,
    }


_MODEL_DIRS = {"llama", "gpt"}  # top-level dirs under chats/ that are model prefixes
_MODEL_LABELS = {
    "default": "Original",
    "gpt":     "GPT-4o-mini",
    "llama":   "Llama-3.1-70B",
}


def _model_label(key: str) -> str:
    return _MODEL_LABELS.get(key, key.upper())


def get_available_datasets() -> list[str]:
    """Return datasets found under chats/, handling both 3-level and 4-level structures."""
    chats_root = "chats"
    if not os.path.isdir(chats_root):
        return []
    found = set()
    for entry in os.listdir(chats_root):
        path = os.path.join(chats_root, entry)
        if not os.path.isdir(path):
            continue
        if entry in _MODEL_DIRS:
            # 4-level: chats/{model}/{dataset}/{mode}/
            for sub in os.listdir(path):
                if os.path.isdir(os.path.join(path, sub)):
                    found.add(sub)
        else:
            found.add(entry)
    datasets = sorted(found)
    if "redial" in datasets:
        datasets.remove("redial")
        datasets.append("redial")
    # Prefer imdb first
    if "imdb" in datasets:
        datasets.remove("imdb")
        datasets.insert(0, "imdb")
    return datasets


def _has_chat_txt(root: str) -> bool:
    for _r, _d, files in os.walk(root):
        if any(f.endswith('.txt') and 'tmp' not in f for f in files):
            return True
    return False


def get_available_models(dataset: str) -> list[str]:
    """Return model labels that actually have chat .txt files for the given dataset."""
    chats_root = "chats"
    models = []
    default_root = os.path.join(chats_root, dataset)
    if os.path.isdir(default_root) and _has_chat_txt(default_root):
        models.append("default")
    for entry in sorted(os.listdir(chats_root)):
        if entry in _MODEL_DIRS:
            model_root = os.path.join(chats_root, entry, dataset)
            if os.path.isdir(model_root) and _has_chat_txt(model_root):
                models.append(entry)
    return models


def get_available_modes(dataset: str, model: str = "default") -> list[str]:
    """Return modes that have conversation files for the given dataset + model."""
    base = os.path.join("chats", dataset) if model == "default" else os.path.join("chats", model, dataset)
    if not os.path.isdir(base):
        return []
    modes = []
    for mode in sorted(os.listdir(base)):
        mode_dir = os.path.join(base, mode)
        if not os.path.isdir(mode_dir):
            continue
        has_files = False
        for root, _, files in os.walk(mode_dir):
            if any(f.endswith('.txt') and 'tmp' not in f for f in files):
                has_files = True
                break
        if has_files:
            modes.append(mode)
    return modes


def _load_baseline_chats() -> list[dict]:
    """Load ReDial redial conversations from chats/redial/real/."""
    folder = os.path.join("chats", "redial", "real")
    if not os.path.isdir(folder):
        return []
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt') or 'tmp' in fname:
            continue
        p = parse_filename(fname)
        with open(os.path.join(folder, fname), encoding='utf-8') as f:
            content = f.read()
        results.append({
            'fname': fname,
            'content': content,
            'strategy': 'real',
            'run': 'redial',
            'item_id': p['item_id'] if p else '?',
            'num_turns': p['num_turns'] if p else 0,
            'rec_attempts': p['rec_attempts'] if p else 0,
            'succeed': p['succeed'] if p else 0,
        })
    return results


def load_chat_files(dataset: str, mode: str, model: str = "default") -> list[dict]:
    """Load all chat txt files for a dataset+mode+model, return list of {fname, content}."""
    if dataset == "redial":
        return _load_baseline_chats()
    base = os.path.join("chats", dataset, mode) if model == "default" else os.path.join("chats", model, dataset, mode)
    if not os.path.isdir(base):
        return []

    # Collect all strategy subfolders (user_item_pairs, item_list, etc.) or direct files
    results = []
    strategy_dirs = sorted(os.listdir(base), reverse=False)
    for strategy in strategy_dirs:
        strategy_path = os.path.join(base, strategy)
        if not os.path.isdir(strategy_path):
            continue
        # Each strategy has timestamped run folders
        run_dirs = sorted(os.listdir(strategy_path), reverse=True)
        for run in run_dirs:
            run_path = os.path.join(strategy_path, run)
            if not os.path.isdir(run_path):
                continue
            for fname in sorted(os.listdir(run_path)):
                if not fname.endswith('.txt') or 'tmp' in fname:
                    continue
                with open(os.path.join(run_path, fname), encoding='utf-8') as f:
                    content = f.read()
                p = parse_filename(fname)
                results.append({
                    'fname': fname,
                    'content': content,
                    'strategy': strategy,
                    'run': run,
                    'item_id': p['item_id'] if p else '?',
                    'num_turns': p['num_turns'] if p else 0,
                    'rec_attempts': p['rec_attempts'] if p else 0,
                    'succeed': p['succeed'] if p else 0,
                })
    return results


# ─────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


# ─────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────

def page_overview():
    st.title("📊 OmniSim — Overview")
    conv_df = load_all_conv_stats()
    if conv_df.empty:
        st.warning("No conversation data found.")
        return

    # Summary cards per mode
    st.subheader("Conversation Summary by Mode")
    cols = st.columns(3)
    for i, mode in enumerate(MODES):
        df = conv_df[conv_df['mode'] == mode]
        if df.empty:
            continue
        with cols[i]:
            st.markdown(f"**{mode.upper()}**")
            st.metric("Total", len(df))
            st.metric("Success Rate", f"{df['succeed'].mean():.1%}")
            st.metric("Avg Turns", f"{df['num_turns'].mean():.1f}")
            st.metric("Avg Rec Attempts", f"{df['rec_attempts'].mean():.1f}")

    st.divider()

    # Turns distribution
    st.subheader("Turns Distribution")
    fig = go.Figure()
    for mode in MODES:
        df = conv_df[conv_df['mode'] == mode]
        if df.empty:
            continue
        fig.add_trace(go.Box(
            y=df['num_turns'],
            name=mode.capitalize(),
            marker_color=MODE_COLORS[mode],
            boxmean=True,
        ))
    fig.update_layout(yaxis_title="Number of Turns", height=380)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Success rate bar
        st.subheader("Success Rate")
        rates = {m: conv_df[conv_df['mode'] == m]['succeed'].mean() for m in MODES if not conv_df[conv_df['mode'] == m].empty}
        fig2 = go.Figure(go.Bar(
            x=[m.capitalize() for m in rates],
            y=[v * 100 for v in rates.values()],
            marker_color=[MODE_COLORS[m] for m in rates],
            text=[f"{v:.1%}" for v in rates.values()],
            textposition='outside',
        ))
        fig2.update_layout(yaxis_title="Success Rate (%)", height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Rec attempts distribution
        st.subheader("Recommendation Attempts")
        fig3 = go.Figure()
        for mode in MODES:
            df = conv_df[conv_df['mode'] == mode]
            if df.empty:
                continue
            counts = df['rec_attempts'].value_counts().sort_index()
            fig3.add_trace(go.Bar(
                name=mode.capitalize(),
                x=counts.index.astype(str),
                y=counts.values,
                marker_color=MODE_COLORS[mode],
            ))
        fig3.update_layout(
            barmode='group',
            xaxis_title="Rec Attempts",
            yaxis_title="Count",
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)


def page_nlp_comparison():
    st.title("🔍 Eval: NLP Metrics — OmniSim vs. ReDial")
    st.info("💡 To understand the data behind this comparison, see **Eval: Data Statistics** in the sidebar.")

    # ── Metric Definitions ────────────────────────────────────
    with st.expander("📖 Metric Definitions", expanded=True):
        ref_df = pd.DataFrame([
            {"Metric": "Distinct-2",       "Full Name": "Distinct Bigrams",                     "Measures": "Unique word-pair ratio across all utterances",                          "↑/↓": "↑"},
            {"Metric": "Log TTR",          "Full Name": "Log Type-Token Ratio",                  "Measures": "log(types) / log(tokens) — length-robust vocabulary diversity",         "↑/↓": "↑"},
            {"Metric": "MTLD",             "Full Name": "Measure of Textual Lexical Diversity",  "Measures": "Mean length of word runs maintaining TTR ≥ 0.72",                       "↑/↓": "↑"},
            {"Metric": "HDD",              "Full Name": "Hypergeometric Distribution D",         "Measures": "Expected unique words in a random 42-token sample",                     "↑/↓": "↑"},
            {"Metric": "Cosine Diversity", "Full Name": "Sentence Embedding Cosine Diversity",   "Measures": "Mean pairwise semantic distance between utterance embeddings (sample 200)", "↑/↓": "↑"},
            {"Metric": "Item Entropy",     "Full Name": "Shannon Entropy of Item Distribution",  "Measures": "How evenly target items are spread across conversations",                "↑/↓": "↑"},
        ])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── CSV cache path ─────────────────────────────────────────
    NLP_CACHE_CSV = os.path.join(RESULTS_DIR, "nlp_metrics_cache.csv")

    def _save_results_to_csv(results: dict):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        rows = []
        for key, r in results.items():
            if r:
                row = {"key": key}
                row.update({k: v for k, v in r.items() if k != "key"})
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(NLP_CACHE_CSV, index=False)

    def _load_results_from_csv() -> dict | None:
        if not os.path.exists(NLP_CACHE_CSV):
            return None
        try:
            df = pd.read_csv(NLP_CACHE_CSV)
            results = {}
            key_col = "key" if "key" in df.columns else "label"
            for _, row in df.iterrows():
                key = row[key_col]
                results[key] = row.drop(key_col).to_dict()
            return results
        except Exception:
            return None

    def _run_compute() -> dict:
        import random as _random
        real_folder = os.path.join("chats", "redial", "real_low_turn")
        results = {}
        real_all = [
            os.path.join(root, f)
            for root, _, files in os.walk(real_folder)
            for f in files if f.endswith('.txt') and 'tmp' not in f
        ]
        # ReDial-500 sample
        results["ReDial"] = _compute_nlp_metrics_live(
            real_folder, "ReDial",
            file_sample=_random.sample(real_all, min(500, len(real_all)))
        )
        # LLama and GPT modes
        for model_prefix in ["llama", "gpt"]:
            for mode in MODES:
                mode_folder = os.path.join("chats", model_prefix, "imdb", mode)
                if not os.path.isdir(mode_folder):
                    continue
                all_files = [
                    os.path.join(root, f)
                    for root, _, files in os.walk(mode_folder)
                    for f in files if f.endswith('.txt') and 'tmp' not in f
                ]
                if not all_files:
                    continue
                key_full = f"{model_prefix}_{mode}"
                key_200  = f"{model_prefix}_{mode}_200"
                results[key_full] = _compute_nlp_metrics_live(mode_folder, key_full)
                results[key_200]  = _compute_nlp_metrics_live(
                    mode_folder, key_200,
                    file_sample=_random.sample(all_files, min(200, len(all_files)))
                )
        return results

    # ── Load from cache ────────────────────────────────────────
    if "nlp_live_results" not in st.session_state:
        cached = _load_results_from_csv()
        if cached:
            st.session_state["nlp_live_results"] = cached

    if "nlp_live_results" in st.session_state:
        mtime = os.path.getmtime(NLP_CACHE_CSV) if os.path.exists(NLP_CACHE_CSV) else None
        if mtime:
            import datetime as _dt
            ts = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"📂 Loaded from cache — last computed {ts}")
    else:
        st.warning("No cached results found. Run `python utils/compute_nlp_cache.py` to generate metrics.")
        return

    if "nlp_live_results" not in st.session_state:
        return

    results = st.session_state["nlp_live_results"]
    real_r = results.get("ReDial")
    if not real_r:
        st.warning("No ReDial data in cache.")
        return

    ALL_METRICS = [
        ("distinct_2",       "Distinct-2",   True,  3),
        ("log_ttr",          "Log TTR",      True,  3),
        ("mtld",             "MTLD",         True,  1),
        ("hdd",              "HDD",          True,  3),
        ("cosine_diversity", "Cosine Div.",  True,  3),
        ("item_entropy",     "Item Entropy", True,  2),
    ]
    metric_col_names = [m[1] for m in ALL_METRICS]

    def _fmt(val, dec):
        if val is None or (isinstance(val, float) and val != val):
            return "—"
        return f"{val:.{dec}f}"

    real_full_r = results.get("ReDial-Full")

    def _build_table(model_prefix: str, use_full_redial: bool = False) -> pd.DataFrame | None:
        """Build comparison table: Dataset + Mode columns, OmniSim modes + ReDial row."""
        ref_r = real_full_r if use_full_redial else real_r
        rows = []
        for mode in MODES:
            m_r = results.get(f"{model_prefix}_{mode}")
            if not m_r:
                continue
            row = {
                "Dataset": "IMDB",
                "Mode": mode.capitalize(),
                "N": str(int(m_r.get("num_files", 0) or 0)),
                "Succ%": m_r.get("success_rate", "—"),
                "Avg Turns": m_r.get("avg_turns", "—"),
            }
            for mkey, col, higher, dec in ALL_METRICS:
                row[col] = _fmt(m_r.get(mkey), dec)
            rows.append(row)
        if not rows or not ref_r:
            return None
        # ReDial row
        redial_label = "ReDial (Full)" if use_full_redial else "ReDial (Sample)"
        redial_row = {
            "Dataset": redial_label,
            "Mode": "",
            "N": str(int(ref_r.get("num_files", 0) or 0)),
            "Succ%": ref_r.get("success_rate", "—"),
            "Avg Turns": ref_r.get("avg_turns", "—"),
        }
        for mkey, col, higher, dec in ALL_METRICS:
            redial_row[col] = _fmt(ref_r.get(mkey), dec)
        rows.append(redial_row)
        return pd.DataFrame(rows)

    def _highlight_gap(df):
        # Highlight vs ReDial row (last row)
        real_idx = len(df) - 1
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i in range(real_idx):
            for col in metric_col_names:
                try:
                    sv = float(df.at[i, col]); rv = float(df.at[real_idx, col])
                except Exception:
                    continue
                rel = abs(sv - rv) / abs(rv) if rv else 0
                if rel <= 0.05:
                    styles.at[i, col] = "background-color:#d4edda;font-weight:bold"
                elif rel <= 0.20:
                    styles.at[i, col] = "background-color:#fff3cd"
                else:
                    styles.at[i, col] = "background-color:#f8d7da"
        # ReDial row — blue
        for col in df.columns:
            styles.at[real_idx, col] = "background-color:#e8f4fd;font-weight:bold"
        return styles

    def _render_table(model_prefix: str, model_label: str, use_full_redial: bool = False):
        tbl = _build_table(model_prefix, use_full_redial)
        if tbl is None:
            st.warning(f"No data found for {model_label} in cache.")
            return
        st.caption("🟢 Gap ≤5%  🟡 Gap ≤20%  🔴 Gap >20%  (vs ReDial)")
        st.dataframe(
            tbl.style.apply(_highlight_gap, axis=None),
            use_container_width=True, hide_index=True,
        )

    # ── Table 1 & 2 ──────────────────────────────────────────
    st.subheader("📊 ReDial (Sample) vs Simulations from IMDB")
    col1a, col1b = st.columns(2)
    with col1a:
        st.markdown("**LLaMA-3.1-70B**")
        _render_table("llama", "LLaMA-3.1-70B")
    with col1b:
        st.markdown("**GPT-4o-mini**")
        _render_table("gpt", "GPT-4o-mini")

    st.divider()

    # ── Table 3 ───────────────────────────────────────────────
    st.subheader("📊 ReDial (Full Data) vs Simulations from IMDB")
    st.caption("Comparison against full ReDial dataset (11,348 conversations, chats/redial/real_full/)")
    if not real_full_r:
        st.warning("ReDial Full data not found in cache. Please recompute.")
    else:
        col3a, col3b = st.columns(2)
        with col3a:
            st.markdown("**LLaMA-3.1-70B**")
            _render_table("llama", "LLaMA-3.1-70B", use_full_redial=True)
        with col3b:
            st.markdown("**GPT-4o-mini**")
            _render_table("gpt", "GPT-4o-mini", use_full_redial=True)


def page_conversation_browser():
    st.title("💬 Conversation Browser")

    # ── Dataset → Mode selection ─────────────────────────────
    datasets = get_available_datasets()
    if not datasets:
        st.warning("No conversation data found under chats/.")
        return

    col_ds, col_mdl, col_mode = st.columns(3)
    with col_ds:
        dataset = st.selectbox("Dataset", datasets, format_func=lambda x: "ReDial (Real Dialogues)" if x == "redial" else x.upper())
    with col_mdl:
        if dataset == "redial":
            st.markdown("**Model**")
            st.markdown("—")
            model = "default"
        else:
            available_models = get_available_models(dataset)
            model = st.selectbox("Model", available_models, format_func=_model_label)
    with col_mode:
        if dataset == "redial":
            st.markdown("**Mode**")
            st.markdown("real *(ReDial human conversations)*")
            mode = "real"
        else:
            available_modes = get_available_modes(dataset, model)
            if not available_modes:
                st.warning(f"No modes found for {dataset} / {model}.")
                return
            mode = st.selectbox("Mode", available_modes, format_func=str.capitalize)

    chats = load_chat_files(dataset, mode, model)
    if not chats:
        st.warning(f"No conversations found for {dataset} / {model} / {mode}.")
        return

    # ── Filters ──────────────────────────────────────────────
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        succeed_filter = st.selectbox("Outcome", ["All", "Success", "Failed"])
    with col2:
        strategies = sorted(set(c['strategy'] for c in chats))
        strategy_filter = st.selectbox("Strategy", ["All"] + strategies)
    with col3:
        min_turns = int(min(c['num_turns'] for c in chats))
        max_turns = int(max(c['num_turns'] for c in chats))
        if min_turns == max_turns:
            st.markdown(f"**Turns**  \n{min_turns} (all)")
            turns_range = (min_turns, max_turns)
        else:
            turns_range = st.slider("Turns range", min_turns, max_turns, (min_turns, max_turns))
    with col4:
        search = st.text_input("Search item_id or keyword")

    filtered = chats
    if succeed_filter == "Success":
        filtered = [c for c in filtered if c['succeed'] == 1]
    elif succeed_filter == "Failed":
        filtered = [c for c in filtered if c['succeed'] == 0]
    if strategy_filter != "All":
        filtered = [c for c in filtered if c['strategy'] == strategy_filter]
    filtered = [c for c in filtered if turns_range[0] <= c['num_turns'] <= turns_range[1]]
    if search:
        filtered = [c for c in filtered if search.lower() in c['fname'].lower() or search.lower() in c['content'].lower()]

    st.caption(f"Showing {len(filtered)} / {len(chats)} conversations  |  dataset: **{dataset}**  mode: **{mode}**")

    if not filtered:
        st.info("No conversations match the filters.")
        return

    # ── Table with pagination ─────────────────────────────────
    _pg_key = "conv_page"
    _ps_key = "conv_page_size"
    if _pg_key not in st.session_state:
        st.session_state[_pg_key] = 1
    if _ps_key not in st.session_state:
        st.session_state[_ps_key] = 20

    page_size = st.session_state[_ps_key]
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = min(st.session_state[_pg_key], total_pages)
    page_start = (page - 1) * page_size
    page_end = min(page_start + page_size, len(filtered))
    paged = filtered[page_start:page_end]

    _sel_key = f"conv_sel_{page}"
    cur_sel = st.session_state.get(_sel_key, 0)

    table_data = [{
        '☑': i == cur_sel,
        'Strategy': c['strategy'],
        'Item ID': c['item_id'],
        'Turns': c['num_turns'],
        'Rec': c['rec_attempts'],
        'OK': '✅' if c['succeed'] else '❌',
        'File': c['fname'],
    } for i, c in enumerate(paged)]

    edited = st.data_editor(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={'☑': st.column_config.CheckboxColumn('☑', width='small')},
        disabled=['Strategy', 'Item ID', 'Turns', 'Rec', 'OK', 'File'],
        key=f"conv_editor_{page}",
    )
    checked = edited['☑'].tolist()
    newly = [i for i, v in enumerate(checked) if v and i != cur_sel]
    sel_idx = newly[0] if newly else cur_sel
    if sel_idx != cur_sel:
        st.session_state[_sel_key] = sel_idx
        st.rerun()
    sel = [sel_idx]

    # ── Pagination bar (below table) ──────────────────────────
    def _page_nums(cur, total):
        """Return list of page numbers and '...' for ellipsis."""
        if total <= 7:
            return list(range(1, total + 1))
        pages = [1]
        if cur > 3:
            pages.append('...')
        for p in range(max(2, cur - 1), min(total, cur + 2)):
            pages.append(p)
        if cur < total - 2:
            pages.append('...')
        if total not in pages:
            pages.append(total)
        return pages

    btn_cols = st.columns([1, 1] + [1] * len(_page_nums(page, total_pages)) + [1, 2, 2])
    col_idx = 0

    # Prev button
    with btn_cols[col_idx]:
        if st.button("◀", disabled=(page == 1), key="_pg_prev"):
            st.session_state[_pg_key] = page - 1
            st.rerun()
    col_idx += 1

    # First page
    with btn_cols[col_idx]:
        pass
    col_idx += 1

    # Page number buttons
    for pn in _page_nums(page, total_pages):
        with btn_cols[col_idx]:
            if pn == '...':
                st.markdown("<div style='text-align:center;padding-top:6px'>…</div>", unsafe_allow_html=True)
            else:
                label = f"**{pn}**" if pn == page else str(pn)
                if st.button(label, key=f"_pg_{pn}"):
                    st.session_state[_pg_key] = pn
                    st.rerun()
        col_idx += 1

    # Next button
    with btn_cols[col_idx]:
        if st.button("▶", disabled=(page == total_pages), key="_pg_next"):
            st.session_state[_pg_key] = page + 1
            st.rerun()
    col_idx += 1

    # Jump to page
    with btn_cols[col_idx]:
        jump = st.number_input("跳转", min_value=1, max_value=total_pages,
                               value=page, step=1, key="_pg_jump",
                               label_visibility="collapsed")
        if jump != page:
            st.session_state[_pg_key] = jump
            st.rerun()
    col_idx += 1

    # Page size selector
    with btn_cols[col_idx]:
        new_ps = st.selectbox("Per page", [10, 20, 50, 100],
                              index=[10,20,50,100].index(st.session_state[_ps_key]),
                              key="_ps_select", label_visibility="collapsed")
        if new_ps != st.session_state[_ps_key]:
            st.session_state[_ps_key] = new_ps
            st.session_state[_pg_key] = 1
            st.rerun()

    st.caption(f"{len(filtered)} conversations  |  {page_start+1}–{page_end}  |  Page {page}/{total_pages}")

    # ── Conversation viewer ──────────────────────────────────
    if sel:
        chat = paged[sel[0]]
        st.divider()
        chat_col, _ = st.columns([3, 2])
        with chat_col:
            st.markdown(f"**📄 {chat['fname']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Turns", chat['num_turns'])
            m2.metric("Rec Attempts", chat['rec_attempts'])
            m3.metric("Outcome", "✅ Success" if chat['succeed'] else "❌ Failed")

            # Parse into turns; continuation lines and rec items attach to the last Bot bubble
            turns = []
            for line in chat['content'].split('\n'):
                if line.startswith('Bot:'):
                    turns.append({'type': 'bot', 'text': line[4:].strip(), 'items': []})
                elif line.startswith('User:'):
                    turns.append({'type': 'user', 'text': line[5:].strip()})
                elif line.startswith('System:'):
                    turns.append({'type': 'system', 'text': line[7:].strip()})
                elif line.strip().startswith('- ') and turns and turns[-1]['type'] == 'bot':
                    turns[-1]['items'].append(line.strip()[2:])
                elif line.strip() and turns and turns[-1]['type'] == 'bot':
                    turns[-1]['text'] += ' ' + line.strip()

            html_lines = []
            for turn in turns:
                if turn['type'] == 'bot':
                    text = turn['text']
                    items_html = ''.join(
                        f'<div style="padding:2px 0 2px 12px;color:#dde8ff;font-size:0.85em">• {r}</div>'
                        for r in turn['items']
                    )
                    html_lines.append(
                        f'<div style="display:flex;justify-content:flex-end;margin:4px 0">'
                        f'<div style="max-width:75%;background:#4C72B0;color:#fff;'
                        f'padding:8px 12px;border-radius:16px 16px 4px 16px;font-size:0.9em">'
                        f'<span style="font-size:0.75em;opacity:0.8;display:block;margin-bottom:2px">🤖 Bot</span>'
                        f'{text}{items_html}</div></div>'
                    )
                elif turn['type'] == 'user':
                    text = turn['text']
                    html_lines.append(
                        f'<div style="display:flex;justify-content:flex-start;margin:4px 0">'
                        f'<div style="max-width:75%;background:#F0FFF4;color:#1a3a1a;'
                        f'border:1px solid #55A868;'
                        f'padding:8px 12px;border-radius:16px 16px 16px 4px;font-size:0.9em">'
                        f'<span style="font-size:0.75em;color:#55A868;display:block;margin-bottom:2px">👤 User</span>'
                        f'{text}</div></div>'
                    )
                elif turn['type'] == 'system':
                    text = turn['text']
                    html_lines.append(
                        f'<div style="text-align:center;margin:8px 0">'
                        f'<span style="background:#FFF8E1;color:#7a6000;border:1px solid #FFC107;'
                        f'border-radius:12px;padding:3px 12px;font-size:0.8em">⚙️ {text}</span></div>'
                    )

            st.html(
                '<div style="background:#f5f7fa;border:1px solid #dde3ed;border-radius:10px;'
                'padding:12px 16px;max-height:600px;overflow-y:auto">'
                + '\n'.join(html_lines) + '</div>'
            )


def page_descriptive_stats():
    st.title("📈 Descriptive Statistics")
    conv_df = load_all_conv_stats()
    if conv_df.empty:
        st.warning("No conversation data found.")
        return

    mode = st.selectbox("Mode", MODES, format_func=str.capitalize)
    df = conv_df[conv_df['mode'] == mode]
    if df.empty:
        st.info(f"No data for {mode} mode.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Success Rate", f"{df['succeed'].mean():.1%}")
    col3.metric("Avg Turns", f"{df['num_turns'].mean():.2f}")
    col4.metric("Avg Rec Attempts", f"{df['rec_attempts'].mean():.2f}")

    st.divider()

    # Turns histogram
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Turns Distribution")
        fig = px.histogram(df, x='num_turns', nbins=15,
                           color_discrete_sequence=[MODE_COLORS[mode]])
        fig.update_layout(xaxis_title='Number of Turns', yaxis_title='Count', height=300)
        st.plotly_chart(fig, use_container_width=True)

        desc = df['num_turns'].describe()
        st.dataframe(pd.DataFrame({
            'Stat': ['mean', 'std', 'min', 'Q1 (25%)', 'median', 'Q3 (75%)', 'max'],
            'Value': [f"{desc['mean']:.2f}", f"{desc['std']:.2f}", f"{desc['min']:.0f}",
                      f"{desc['25%']:.0f}", f"{desc['50%']:.0f}", f"{desc['75%']:.0f}", f"{desc['max']:.0f}"]
        }), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Rec Attempts Distribution")
        fig2 = px.histogram(df, x='rec_attempts', nbins=10,
                            color_discrete_sequence=[MODE_COLORS[mode]])
        fig2.update_layout(xaxis_title='Rec Attempts', yaxis_title='Count', height=300)
        st.plotly_chart(fig2, use_container_width=True)

        desc2 = df['rec_attempts'].describe()
        st.dataframe(pd.DataFrame({
            'Stat': ['mean', 'std', 'min', 'Q1 (25%)', 'median', 'Q3 (75%)', 'max'],
            'Value': [f"{desc2['mean']:.2f}", f"{desc2['std']:.2f}", f"{desc2['min']:.0f}",
                      f"{desc2['25%']:.0f}", f"{desc2['50%']:.0f}", f"{desc2['75%']:.0f}", f"{desc2['max']:.0f}"]
        }), hide_index=True, use_container_width=True)


def page_dialogue_stats():
    st.title("📊 Dialogues Statistics")
    st.markdown("Select a dataset, mode, and run folder to compute conversation statistics.")

    datasets = get_available_datasets()
    if not datasets:
        st.warning("No conversation data found under `chats/`.")
        return

    col_ds, col_mdl, col_mode, col_folder = st.columns(4)
    with col_ds:
        dataset = st.selectbox("Dataset", datasets, format_func=str.upper, key="dstat_ds")

    if dataset == "redial":
        folder_path = os.path.join("chats", "redial", "real")
        with col_mdl:
            st.markdown("**Model:** —")
        with col_mode:
            st.markdown("**Mode:** real")
        with col_folder:
            st.markdown("**Folder:** chats/redial/real")
        rows = []
        if os.path.isdir(folder_path):
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith('.txt') or 'tmp' in fname:
                    continue
                p = parse_filename(fname)
                if p:
                    rows.append(p)
        df = pd.DataFrame(rows)
    else:
        with col_mdl:
            available_models = get_available_models(dataset)
            model = st.selectbox("Model", available_models, format_func=_model_label, key="dstat_mdl")
        available_modes = get_available_modes(dataset, model)
        if not available_modes:
            st.warning(f"No modes found for '{dataset}' / '{model}'.")
            return
        with col_mode:
            mode = st.selectbox("Mode", available_modes, format_func=str.capitalize, key="dstat_mode")

        base = os.path.join("chats", dataset, mode) if model == "default" else os.path.join("chats", model, dataset, mode)
        run_options = []
        if os.path.isdir(base):
            for strat in sorted(os.listdir(base)):
                strat_path = os.path.join(base, strat)
                if not os.path.isdir(strat_path):
                    continue
                for run in sorted(os.listdir(strat_path), reverse=True):
                    run_path = os.path.join(strat_path, run)
                    if os.path.isdir(run_path):
                        run_options.append(f"{strat}/{run}")

        if not run_options:
            st.warning("No run folders found.")
            return
        with col_folder:
            selected = st.selectbox("Folder", run_options, key="dstat_folder")

        strat_part, run_part = selected.split("/", 1)
        folder_path = os.path.join(base, strat_part, run_part)
        rows = []
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if not fname.endswith('.txt') or 'tmp' in fname:
                    continue
                p = parse_filename(fname)
                if p:
                    rows.append(p)
        df = pd.DataFrame(rows)

    if df.empty:
        st.info("No conversations found in the selected folder.")
        return

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("Success Rate", f"{df['succeed'].mean():.1%}")
    c3.metric("Avg Turns", f"{df['num_turns'].mean():.2f}")
    c4.metric("Avg Rec Attempts", f"{df['rec_attempts'].mean():.2f}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Turns Distribution")
        fig = px.histogram(df, x='num_turns', nbins=15, color_discrete_sequence=["#4C72B0"])
        fig.update_layout(xaxis_title='Number of Turns', yaxis_title='Count', height=300)
        st.plotly_chart(fig, use_container_width=True)
        desc = df['num_turns'].describe()
        st.dataframe(pd.DataFrame({
            'Stat': ['mean', 'std', 'min', 'Q1 (25%)', 'median', 'Q3 (75%)', 'max'],
            'Value': [f"{desc['mean']:.2f}", f"{desc['std']:.2f}", f"{desc['min']:.0f}",
                      f"{desc['25%']:.0f}", f"{desc['50%']:.0f}", f"{desc['75%']:.0f}", f"{desc['max']:.0f}"]
        }), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Rec Attempts Distribution")
        fig2 = px.histogram(df, x='rec_attempts', nbins=10, color_discrete_sequence=["#55A868"])
        fig2.update_layout(xaxis_title='Rec Attempts', yaxis_title='Count', height=300)
        st.plotly_chart(fig2, use_container_width=True)
        desc2 = df['rec_attempts'].describe()
        st.dataframe(pd.DataFrame({
            'Stat': ['mean', 'std', 'min', 'Q1 (25%)', 'median', 'Q3 (75%)', 'max'],
            'Value': [f"{desc2['mean']:.2f}", f"{desc2['std']:.2f}", f"{desc2['min']:.0f}",
                      f"{desc2['25%']:.0f}", f"{desc2['50%']:.0f}", f"{desc2['75%']:.0f}", f"{desc2['max']:.0f}"]
        }), hide_index=True, use_container_width=True)


def page_features():
    st.title("🛠️ OmniSim — Features & Documentation")

    # ── Hero ─────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#4C72B0,#2d4a8a);color:#fff;
padding:20px 24px;border-radius:12px;margin-bottom:8px">
<h3 style="margin:0 0 8px 0">What is OmniSim?</h3>
<p style="margin:0;font-size:1.05em;opacity:0.95">
OmniSim is an open-source library for generating <b>simulated conversational recommendation</b> data.
It helps enterprises and researchers bootstrap conversation datasets from scratch — no real user conversations required.
</p>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
**Background — Why do we need this?**
To build a Conversational Recommender System (CRS) or any conversational AI, you need conversation data to learn from:
*when to ask questions, what to ask, when to recommend, and what to recommend.*
Collecting real conversations is extremely time-consuming and expensive.
OmniSim solves the **cold-start problem** by automatically generating high-quality simulated conversations for any item domain.

**GitHub Repository:** https://github.com/irecsys/OmniSim
    """)

    # ── Architecture Diagrams ─────────────────────────────────
    import os as _os
    _img_path = _os.path.join(_os.path.dirname(__file__), "omnisim.png")
    if _os.path.exists(_img_path):
        _a1, _a2, _a3 = st.columns([0.10, 0.80, 0.10])
        with _a2:
            st.image(_img_path, caption="OmniSim System Architecture", use_container_width=True)
    _img_path1 = _os.path.join(_os.path.dirname(__file__), "omnisim1.png")
    if _os.path.exists(_img_path1):
        _c1, _c2, _c3 = st.columns([0.10, 0.80, 0.10])
        with _c2:
            st.image(_img_path1, use_container_width=True)

    st.divider()

    # ── How It Works ─────────────────────────────────────────
    st.subheader("⚙️ How It Works")
    st.markdown("""
**Core Retrieval Formula**

Every recommendation is driven by a hybrid scoring function:

$$score = \\alpha \\cdot cos(TargetItem,\\ UserPref\\_{dialogue}) + (1-\\alpha) \\cdot [(1-\\beta) \\cdot cos(TargetItem,\\ long\\_{term}\\_{pref}) + \\beta \\cdot cos(TargetItem,\\ short\\_{term}\\_{pref})]$$

Where `α` controls the weight between dialogue-extracted preference and user history, and `β` controls long-term vs short-term preference.
**It is optional to upload user rating history, thus long_term_pref and short_term_pref may be unavailable. In this case, alpha is set as 1.**
    """)

    steps = [
        ("1️⃣", "Prepare Data", "Provide item metadata CSV for any domain (movies, fashion, electronics…). Embeddings of item descriptions are computed using **Azure text-embedding-3-small** (or any OpenAI-compatible provider) and indexed into **Elasticsearch**. It is optional to provide users.csv (user demographic info) and interactions.csv (user ratings on the items). These two files can help our simulator to provide personalized outputs."),
        ("2️⃣", "Select Target Item", "For each conversation, OmniSim selects a **target item** and optionally a **user** based on the input strategy (**user-item pairs** / **item list** / **user list**)."),
        ("3️⃣", "Understand User Intent", "The chatbot asks the user about their preferences. Depending on the mode (Free / Static / Adaptive), it asks for open-ended descriptions or specific attribute preferences. <a href='#conversation-modes' style='color:#4C72B0'>(details ↓)</a>"),
        ("4️⃣", "Retrieve & Recommend", "User responses are embedded and used to query Elasticsearch for item retrieval. If rating histories are provided, a **hybrid score** as shown above is computed for personalized item retrieval and recommendations."),
        ("5️⃣", "Accept or Reject", "If the user rejects, they explain why (generated by comparing recommended vs target item metadata). The rejected items are excluded from future recommendations."),
        ("6️⃣", "Conversation Ends", "The conversation terminates when the target item appears in the recommendation list (success) or max attempts are reached (failure)."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0;
padding:10px 14px;background:#f8f9fb;border-radius:8px;border-left:3px solid #4C72B0">
<span style="font-size:1.4em">{icon}</span>
<div><b>{title}</b><br><span style="color:#444;font-size:0.92em">{desc}</span></div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── LLM Configuration ────────────────────────────────────
    st.subheader("🤖 LLM Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
OmniSim supports any **OpenAI-compatible API** — including OpenAI, Azure OpenAI, and self-hosted models (e.g. Llama via ThetaEdge or vLLM):
- Chatbot responses & questions
- Chit-chat generation
- Rejection explanation generation
- Adaptive / Static attribute extraction
- User response simulation

Each role uses an independently configurable temperature for fine-grained control over diversity.
        """)
    with col2:
        st.markdown("""
Configure in `configs/system/system.yaml`:
```yaml
# OpenAI native
openai_provider: openai
chat_model: gpt-4o

# Azure OpenAI
openai_provider: azure
azure_openai_endpoint: https://xxx.openai.azure.com/
azure_openai_api_version: 2024-02-01
chat_model: gpt-4o

# Self-hosted / third-party (OpenAI-compatible)
openai_provider: openai
openai_api_base: https://your-endpoint/v1
chat_model: meta-llama/Meta-Llama-3.1-70B-Instruct

# Temperature controls
temp_rephrase:    0.8   # attribute rephrasing
temp_rejection:   0.9   # rejection explanations
temp_rec_explain: 0.9   # recommendation explanations
temp_free_user:   1.0   # free-mode user simulation

# Ratio controls
rec_explanation_ratio: 0.5  # explanations of why recommend these items
```
        """)

    st.divider()

    # ── Input Requirements ────────────────────────────────────
    st.subheader("📥 Input Requirements")

    tab_req, tab_opt = st.tabs(["Required Inputs", "Optional Inputs (Personalisation)"])

    with tab_req:
        st.markdown("#### 1. Item Metadata CSV")
        st.markdown("""
Any domain is supported. The CSV must contain at minimum an item ID, title, and a descriptive text field.
Embeddings of the description field are computed and stored in Elasticsearch.

| Domain | Example Fields |
|--------|---------------|
| 🎬 Movies | `imdb_id`, `title`, `genre`, `overview`, `director`, `release_year` |
| 👗 Fashion | `product_id`, `title`, `brand`, `color`, `price`, `category`, `details` |
| 🛒 E-commerce | `product_id`, `title`, `brand`, `material`, `gender`, `availability` |
        """)

        st.markdown("#### 2. Conversation Input — 4 Strategies")
        st.markdown("""
| Strategy | Input File | Behaviour | `chats_per_entry` |
|----------|-----------|-----------|-------------------|
| **user_item_pairs** | `test_pairs.csv` (user_id, item_id) | 1 conversation per pair | N conversations per pair |
| **item_list** | `test_items.csv` (item_id) | N conversations per item | if users available → N random users; else anonymous |
| **user_list** | `test_users.csv` (user_id) | N conversations per user with random items | N random items per user |
| **random** | *(no file)* | Fully random: anonymous users + random items from ES | Controlled by `guestuser_randomitem_records` |

All strategies can run simultaneously in one execution.
        """)

    with tab_opt:
        st.markdown("""
Providing `users.csv` and `interactions.csv` unlocks **personalisation**:

| File | Required Fields | Purpose |
|------|----------------|---------|
| `users.csv` | `user_id`, demographic info (age, gender, occupation…) | Build user persona |
| `interactions.csv` | `user_id`, `item_id`, `rating` (+ optional `timestamp`) | Build preference history |

#### User Profile Fields Generated

| Field | Source | Description |
|-------|--------|-------------|
| `user_info` | `users.csv` demographics | Persona description (age, gender, occupation…) |
| `user_likes_long` | All ratings > 3 | Long-term preference summary from full history |
| `user_likes_short` | Ratings > 3 in last 30 days | Short-term preference (requires timestamp) |

> **Fallback rule:** If a user has no ratings in the last 30 days, `user_likes_short` copies `user_likes_long`.

#### How Profiles Are Used

**① Personalised Chit-Chat** — user profile is injected into the LLM prompt to generate context-aware small talk.

**② Hybrid Retrieval Score:**
```
score = α · cos(TargetItem, UserPref_dialogue) + (1-α) · [(1-β) · cos(TargetItem, long_term_pref) + β · cos(TargetItem, short_term_pref)]
```
- `α` (weight_es_score): balance between dialogue-extracted preference and user history
- `β` (weight_user_taste_short): balance between long-term and short-term preference
        """)

    st.divider()

    # ── Pre-Prepared Datasets ─────────────────────────────────
    st.subheader("📦 Pre-Prepared Datasets")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div style="border:1px solid #ddd;border-radius:10px;padding:14px">
<h4 style="margin-top:0">🎬 IMDB</h4>
<b>Domain:</b> Movies<br><br>

- IMDB movie metadata<br>
  (title, genre, overview, director, cast…)
- Real user ratings from IMDB
- Simulated demographic info for 20 users
- 200 test items, 200 user-item pairs

<br><code>ES index: imdb-2025</code><br>
<code>200 documents</code>
</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div style="border:1px solid #ddd;border-radius:10px;padding:14px">
<h4 style="margin-top:0">👗 H&M</h4>
<b>Domain:</b> Fashion / E-commerce<br><br>

- H&M product metadata<br>
  (brand, price, color, category, details…)
- Simulated users & interactions

<br><code>ES index: hm-2025</code><br>
<code>3246 documents</code>
</div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Conversation Modes ────────────────────────────────────
    st.subheader("💬 Conversation Modes")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div style="border-top:4px solid #4C72B0;border-radius:8px;padding:14px;background:#f0f4ff">
<h4 style="color:#4C72B0;margin-top:0">🟦 Free Mode</h4>

The chatbot asks the user to <b>freely describe</b> what they're looking for in natural language.
The description is embedded and used for vector search in Elasticsearch.

<br><b>Pros:</b>
<ul>
<li>Most open-ended and natural</li>
<li>No reliance on fixed attributes</li>
<li>Works for any domain</li>
</ul>

<b>Search:</b> Dense vector kNN search on user description embedding
</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div style="border-top:4px solid #55A868;border-radius:8px;padding:14px;background:#f0fff4">
<h4 style="color:#2d6a3f;margin-top:0">🟩 Static Mode</h4>

The chatbot asks about <b>specific metadata attributes</b> defined in the item schema (e.g., genre, color, brand).
User answers are used to build a structured query.

<br><b>Pros:</b>
<ul>
<li>Structured and predictable</li>
<li>Good for attribute-rich domains</li>
</ul>

<b>Limitation:</b> Limited to schema-defined fields only<br>
<b>Search:</b> Attribute-filtered vector search
</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div style="border-top:4px solid #C44E52;border-radius:8px;padding:14px;background:#fff0f0">
<h4 style="color:#8b2a2d;margin-top:0">🟥 Adaptive Mode</h4>

Like Static, but the chatbot <b>also queries the LLM</b> to discover additional relevant attributes beyond the schema.
Even if an attribute isn't in the data, the LLM extracts it from item descriptions.

<br><b>Example:</b> Buying a powerbank → LLM suggests asking about capacity, charging speed, portability

<br>Preference questions are <b>randomly varied</b> across 12+ templates
(e.g. <i>"Any must-haves for genre?"</i>, <i>"Does runtime matter to you?"</i>)
and no-preference replies use 15+ natural variants (<i>"Not fussed."</i>, <i>"Open to anything."</i>)
— significantly improving lexical diversity vs. fixed-phrase repetition.

<br><b>Pros:</b>
<ul>
<li>Most realistic & flexible</li>
<li>Highest NLP diversity scores (Distinct-1: 0.168, MTLD: 89)</li>
<li>Closest to real shopping conversations</li>
</ul>
<b>Search:</b> LLM-augmented attribute + vector search
</div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Conversation Quality Features ─────────────────────────
    st.subheader("✨ Conversation Quality Features")

    feat_cols = st.columns(2)
    features_left = [
        ("💬", "Chit-Chat",
         "Controlled by `chit_chat_ratio` (0–1) in config. Added randomly to bot turns. "
         "If a user profile is available, chit-chat is **personalised** — referencing the user's demographic info or preferences."),
        ("❌", "Rejection Explanation",
         "When the user rejects a recommendation, the LLM generates a natural explanation "
         "by **comparing the recommended items' metadata against the target item** — explaining what's different "
         "(e.g. wrong genre, too expensive, wrong brand). Controlled by `rejection_explanation_ratio`."),
        ("💡", "Recommendation Explanation",
         "When the bot recommends items, it can also explain **why** those items match the user's stated needs — "
         "connecting item metadata (genre, category, attributes) to preferences the user expressed earlier in the conversation. "
         "Controlled by `rec_explanation_ratio`. Example: *Since you mentioned enjoying atmospheric sci-fi, here are some picks...*"),
        ("🚫", "No Repeated Recommendations",
         "Once an item is rejected within a conversation, it is **never recommended again** in that session."),
        ("✂️", "Short Message Support",
         "Controlled by `short_reply_ratio` (default 0.45) — 45% of user turns are short natural replies (2–5 words). "
         "Bot also asks brief follow-up questions (`bot_short_followup_ratio`, default 0.4): *'What else?', 'Any dealbreakers?', 'Old or new?'* "
         "This brings short-message ratio close to real human conversations (~35%)."),
    ]
    features_right = [
        ("🔁", "No Repeated Chit-Chat / Rejections",
         "Used chit-chat phrases and rejection explanations are tracked within each conversation and **not reused**, "
         "preventing repetitive outputs."),
        ("🎲", "Diverse Fixed Phrases",
         "Greetings, recommendation intros, and post-rejection expressions are **randomly varied** "
         "from a pool of templates to avoid formulaic patterns."),
        ("1️⃣", "Single-Item vs. List Format",
         "When `rec_top_n = 1`, recommendations are presented as a **natural conversational sentence** "
         "(e.g. *How about Titanic? Since you enjoy romance, this fits perfectly.*) — "
         "mimicking real human recommendations like ReDial. When `rec_top_n > 1`, a bulleted list is shown with an optional explanation prefix."),
        ("📝", "Configurable Prompt Templates",
         "All LLM prompts are extracted to `configs/prompts/default.yaml` — fully editable without touching code. "
         "Override with `prompts_file: configs/prompts/my_prompts.yaml` to customise for a different domain or language."),
        ("⚙️", "YAML Configuration Control",
         "All behaviour is controlled via YAML config files: `chit_chat_ratio`, `rejection_explanation_ratio`, "
         "`rec_explanation_ratio`, `max_rec_attempts`, `rec_top_n`, `chats_per_entry`, `weight_es_score`, `weight_user_taste_short`, "
         "`short_reply_ratio`, `bot_short_followup_ratio`, `temp_rephrase`, `temp_free_user`, etc."),
    ]
    with feat_cols[0]:
        for icon, title, desc in features_left:
            st.markdown(f"""
<div style="background:#f8f9fb;border-radius:8px;padding:12px;margin-bottom:8px">
<b>{icon} {title}</b><br>
<span style="font-size:0.92em;color:#444">{desc}</span>
</div>""", unsafe_allow_html=True)
    with feat_cols[1]:
        for icon, title, desc in features_right:
            st.markdown(f"""
<div style="background:#f8f9fb;border-radius:8px;padding:12px;margin-bottom:8px">
<b>{icon} {title}</b><br>
<span style="font-size:0.92em;color:#444">{desc}</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Output File Format ────────────────────────────────────
    st.subheader("📁 Output File Format")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Directory structure:**
```
chats/
  {dataset}/
    {mode}/
      {strategy}/
        {run_timestamp}/
          {user_id}-{item_id}-{turns}-{attempts}-{succeed}-{ts}.txt
```
        """)
    with col2:
        st.markdown("""
**Filename fields:**

| Field | Description |
|-------|-------------|
| `user_id` | User ID, or `"user"` if anonymous |
| `item_id` | Target item ID |
| `turns` | Total dialogue turns |
| `attempts` | Recommendation rounds |
| `succeed` | `1` = accepted, `0` = failed |
| `ts` | Timestamp of conversation |

**Example:** `u1-tt0109830-6-2-1-20260317120000.txt`
        """)

    st.divider()

    # ── Evaluation ────────────────────────────────────────────
    st.subheader("📊 Evaluation")

    tab1, tab2, tab3 = st.tabs(["📈 Conversation Stats", "🔍 NLP Metrics vs Real Data", "🤖 LLM Quality Scores"])

    with tab1:
        st.markdown("""
For each run folder, **descriptive statistics** are computed across all conversations:

| Metric | Description | Stats Reported |
|--------|-------------|----------------|
| **Turns** | Number of dialogue turns per conversation | avg, std, min, Q1, median, Q3, max |
| **Rec Attempts** | How many recommendation rounds occurred | avg, std, min, Q1, median, Q3, max |
| **Succeed Rate** | Proportion of conversations where target item was accepted | % |

These statistics help understand the simulation behaviour across different modes and datasets.
        """)

    with tab2:
        st.markdown("""
Simulated conversations are compared to **real human conversations** (ReDial movie dataset, ~11,000 conversations)
using **corpus-level NLP metrics**. No 1-to-1 alignment is needed, so BLEU/ROUGE are not applicable.

**Baseline sampling:** For each mode, we sample N conversations from ReDial matching the succeed/fail ratio of the simulated set.

| Metric | Full Name | Measures | Ideal |
|--------|-----------|----------|-------|
| **Distinct-2** | Distinct Bigrams | Unique word-pair ratio across all conversations | ↑ Higher is better |
| **Log TTR** | Log Type-Token Ratio | log(types) / log(tokens) — length-robust vocabulary diversity | ↑ Higher is better |
| **MTLD** | Measure of Textual Lexical Diversity | Mean length of word runs maintaining TTR ≥ 0.72 | ↑ Higher is better |
| **HDD** | Hypergeometric Distribution D | Expected unique words in a random 42-token sample | ↑ Higher is better |
| **Cosine Diversity** | Sentence Embedding Cosine Diversity | Mean pairwise semantic distance between utterance embeddings | ↑ Higher is better |
| **Item Entropy** | Shannon Entropy of Item Distribution | How evenly target items are spread across conversations | ↑ Higher is better |

**Why these metrics?** They measure **language naturalness** independent of domain content —
a natural fashion conversation should have similar lexical diversity to a natural movie conversation.
        """)

    with tab3:
        st.markdown("""
An LLM acts as a judge, evaluating the **simulated user's behaviour** across **3 dimensions** (score 1–5 each),
with a Chain-of-Thought rationale written before each score.

| Dimension | What It Evaluates | Criteria |
|-----------|------------------|----------|
| **Language Fluency** | Does the user's language sound natural? | Grammar, Fluency, Naturalness, Tone |
| **Conversational Quality** | Does the user interact logically and consistently? | Response appropriateness, Proactivity, Multi-turn consistency |
| **Content Quality** | Is the user's content relevant and grounded? | Factuality, Coverage, Coherence, Relevance |

Results are saved to `results/judge_*.csv` with per-conversation rationale text.

**Run from the ⚖️ Eval: LLM-as-Judge page** — select dataset, mode, and folder directly in the UI,
then click **🚀 Run Evaluation**. A real-time progress bar shows how many conversations have been scored.

**Or run from command line:**
```bash
python evaluation/judge.py \\
    --folder chats/llama/imdb/adaptive/user_item_pairs/all \\
    --provider thetaedgecloud \\
    --limit 20 \\
    --output results/judge_adaptive.csv
```
        """)


def _get_run_folders(dataset: str, mode: str) -> list[str]:
    """Return sorted run timestamp folders for a given dataset+mode."""
    base = os.path.join("chats", dataset, mode, "user_item_pairs")
    if not os.path.isdir(base):
        return []
    return sorted(
        [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )


def _load_judge_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "language_fluency" in df.columns:
                return df
        except Exception:
            pass
    return None


def _desc_stats(series: pd.Series) -> dict:
    return {
        "N":      str(int(series.count())),
        "Mean":   round(series.mean(), 3),
        "Std":    round(series.std(), 3),
        "Min":    round(series.min(), 3),
        "Q1":     round(series.quantile(0.25), 3),
        "Median": round(series.median(), 3),
        "Q3":     round(series.quantile(0.75), 3),
        "Max":    round(series.max(), 3),
    }

def page_judge():
    st.title("⚖️ Eval: LLM-as-Judge Evaluation")
    st.info("💡 To understand the data used in this evaluation, see **Eval: Data Statistics** in the sidebar.")

    st.markdown("""
The **LLM-as-Judge** evaluator scores the **simulated user's behaviour** across three dimensions (1–5)
with a Chain-of-Thought rationale per conversation.

| Dimension | Criteria |
|-----------|----------|
| **Language Fluency** | Grammar, Fluency, Naturalness, Tone |
| **Conversational Quality** | Appropriateness, Proactivity, Multi-turn Consistency |
| **Content Quality** | Factuality, Coverage, Coherence, Relevance |

**Run batch evaluation (all 6 groups × 3 judges):**
```bash
python scripts/run_judge_all.py
python scripts/run_judge_all.py --limit 50          # quick test with 50 conversations
python scripts/run_judge_all.py --judges gpt deepseek  # specific judges only
```
Per-conversation results are saved to `<folder>/<judge>/eval_*.csv` automatically (resume supported).
    """)

    st.divider()

    # ── Configuration ─────────────────────────────────────────
    SIM_GROUPS = {
        "LLaMA / Free":     ("llama", "free"),
        "LLaMA / Static":   ("llama", "static"),
        "LLaMA / Adaptive": ("llama", "adaptive"),
        "GPT / Free":       ("gpt",   "free"),
        "GPT / Static":     ("gpt",   "static"),
        "GPT / Adaptive":   ("gpt",   "adaptive"),
        "ReDial":           ("redial", ""),
    }
    JUDGE_KEYS = {
        "GPT-4o-mini":   "gpt",
        "DeepSeek":      "deepseek",
        "LLaMA-3.1-70B": "llama",
        "Gemini":        "gemini",
    }
    DIM_COLS   = ["language_fluency", "conversational_quality", "content_quality", "overall_score"]
    DIM_LABELS = ["Lang. Fluency",    "Conv. Quality",          "Content Quality", "Overall"]

    # Load all available CSVs: results/judge_{sim}_{mode}_{judge}.csv
    # For ReDial: results/judge_redial_{judge}.csv
    data = {}   # (group_label, judge_label) -> DataFrame
    for group_label, (sim, mode) in SIM_GROUPS.items():
        for judge_label, judge_key in JUDGE_KEYS.items():
            if sim == "redial":
                csv_path = os.path.join(RESULTS_DIR, f"judge_redial_{judge_key}.csv")
            else:
                csv_path = os.path.join(RESULTS_DIR, f"judge_{sim}_{mode}_{judge_key}.csv")
            df = _load_judge_csv(csv_path)
            if df is not None:
                data[(group_label, judge_label)] = df

    if not data:
        st.info("No judge results found yet. Run:\n```bash\npython scripts/run_judge_all.py\n```")
        return

    available_judges = [j for j in JUDGE_KEYS if any(j == jl for (_, jl) in data)]
    available_groups = [g for g in SIM_GROUPS if any(g == gl for (gl, _) in data)]

    st.markdown(
        f"**Judges available:** {', '.join(available_judges)}  |  "
        f"**Groups available:** {', '.join(available_groups)}"
    )
    st.divider()

    def _fmt(df, col):
        if col not in df.columns:
            return "—"
        if col == "overall_score":
            return f"{df[col].mean():.2f}"
        return f"{df[col].mean():.2f} ± {df[col].std():.2f}"

    def _hl_best_row(s):
        """Highlight highest mean value across judge columns (green bold)."""
        try:
            vals = [float(v.split(" ")[0]) if isinstance(v, str) and v != "—" else -1 for v in s]
            mx = max(vals)
            return [
                "background-color:#d4edda;font-weight:bold" if v == mx and v > 0 else ""
                for v in vals
            ]
        except Exception:
            return [""] * len(s)

    # ── Helper: render 3 metric tables for one sim model ─────
    def _render_sim_section(sim_key: str, dataset_label: str):
        groups_for_sim = {
            g: (sim, mode)
            for g, (sim, mode) in SIM_GROUPS.items()
            if sim == sim_key
        }
        def _build_rows(dim_col):
            rows = []
            for group_label, (_, mode) in groups_for_sim.items():
                row = {"Dataset": dataset_label, "Mode": mode.capitalize(), "N": "500"}
                has_any = False
                for judge_label in available_judges:
                    df = data.get((group_label, judge_label))
                    if df is not None:
                        row[judge_label] = _fmt(df, dim_col)
                        has_any = True
                    else:
                        row[judge_label] = "—"
                if has_any:
                    rows.append(row)
            redial_row = {"Dataset": "ReDial (Sample)", "Mode": "", "N": "500"}
            redial_has = False
            for judge_label in available_judges:
                df = data.get(("ReDial", judge_label))
                if df is not None:
                    redial_row[judge_label] = _fmt(df, dim_col)
                    redial_has = True
                else:
                    redial_row[judge_label] = "—"
            if redial_has:
                rows.append(redial_row)
            return rows, redial_has

        def _render_table(col, rows, redial_has):
            if not rows:
                st.info("No data yet.")
                return
            tbl = pd.DataFrame(rows)
            judge_cols = [j for j in available_judges if j in tbl.columns]
            real_idx = len(tbl) - 1 if redial_has else None

            def _hl_table(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for jcol in judge_cols:
                    try:
                        n = real_idx if real_idx is not None else len(df)
                        sim_vals = [float(df.at[i, jcol].split(" ")[0]) if isinstance(df.at[i, jcol], str) and df.at[i, jcol] != "—" else -1 for i in range(n)]
                        mx = max(sim_vals)
                        for i, v in enumerate(sim_vals):
                            if v == mx and v > 0:
                                styles.at[i, jcol] = "background-color:#d4edda;font-weight:bold"
                    except Exception:
                        pass
                if real_idx is not None:
                    for c in df.columns:
                        styles.at[real_idx, c] = "background-color:#e8f4fd;font-weight:bold"
                return styles

            col.dataframe(tbl.style.apply(_hl_table, axis=None), use_container_width=True, hide_index=True)

        # Row 1: Lang Fluency | Conv Quality
        r1c1, r1c2 = st.columns(2)
        r1c1.markdown("**Lang. Fluency**")
        r1c2.markdown("**Conv. Quality**")
        rows0, rh0 = _build_rows("language_fluency")
        rows1, rh1 = _build_rows("conversational_quality")
        _render_table(r1c1, rows0, rh0)
        _render_table(r1c2, rows1, rh1)

        st.markdown("")

        # Row 2: Content Quality | Overall Score
        r2c1, r2c2 = st.columns(2)
        r2c1.markdown("**Content Quality**")
        r2c2.markdown("**Overall Score**")
        rows2, rh2 = _build_rows("content_quality")
        rows3, rh3 = _build_rows("overall_score")
        _render_table(r2c1, rows2, rh2)
        _render_table(r2c2, rows3, rh3)

        st.markdown("")

    # ── Part 1: LLaMA simulations ─────────────────────────────
    st.subheader("📊 Part 1: Evaluation on LLaMA-3.1-70B Simulations")
    st.caption("Columns = judge model. 🟢 = best among simulation modes. 🔵 = ReDial reference.")
    _render_sim_section("llama", "IMDB")

    st.divider()

    # ── Part 2: GPT simulations ───────────────────────────────
    st.subheader("📊 Part 2: Evaluation on GPT-4o-mini Simulations")
    st.caption("Columns = judge model. 🟢 = best among simulation modes. 🔵 = ReDial reference.")
    _render_sim_section("gpt", "IMDB")

    st.divider()


def page_run():
    import subprocess, sys, time, yaml

    st.title("▶️ Run Simulation")
    st.markdown("Configure and launch an OmniSim simulation directly from the dashboard.")

    # ── Dataset & Mode ────────────────────────────────────────
    st.subheader("1️⃣  Dataset & Mode")
    c1, c2 = st.columns(2)
    with c1:
        dataset = st.radio("Dataset", ["imdb", "hm"], horizontal=True, key="run_dataset",
                           format_func=lambda x: "🎬 IMDB (Movies)" if x == "imdb" else "👗 H&M (Fashion)")
    with c2:
        mode = st.radio("Simulation Mode", ["free", "static", "adaptive"], horizontal=True, key="run_mode",
                    help="free = LLM generates user freely | static = fixed attribute Q&A | adaptive = LLM discovers extra attributes")

    # ── Strategy ──────────────────────────────────────────────
    st.subheader("2️⃣  Strategy")

    inputs_dir = os.path.join("configs", dataset, "inputs")
    csv_files  = sorted(glob.glob(os.path.join(inputs_dir, "*.csv"))) if os.path.isdir(inputs_dir) else []
    csv_names  = [os.path.basename(f) for f in csv_files]

    strategy = st.radio(
        "Select strategy",
        ["user_item_pairs", "item_list", "user_guest", "random items"],
        horizontal=True,
        key="run_strategy",
        help="user_item_pairs: one conversation per pair | item_list: N conv per item | user_guest: N conv per user | random items: anonymous + random item from ES"
    )

    if strategy == "user_item_pairs":
        pairs_opts = [f for f in csv_names if "pair" in f] or csv_names or ["test_pairs.csv"]
        pairs_file = st.selectbox("Pairs CSV", pairs_opts, key="run_pairs_file")
    elif strategy == "item_list":
        items_opts = [f for f in csv_names if "item" in f] or csv_names or ["test_items.csv"]
        items_file = st.selectbox("Items CSV", items_opts, key="run_items_file")
    elif strategy == "user_guest":
        users_opts = [f for f in csv_names if "user" in f] or csv_names or ["test_users.csv"]
        users_file = st.selectbox("Users CSV", users_opts, key="run_users_file")
    elif strategy == "random items":
        random_n = st.number_input("# random conversations", min_value=1, max_value=500, value=10, key="run_random_n")

    # ── Simulation Parameters ─────────────────────────────────
    st.subheader("3️⃣  Parameters")

    with st.expander("Core parameters", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            chats_per_entry = st.number_input("chats_per_entry", min_value=1, max_value=10, value=1,
                                              help="Conversations generated per input pair/user/item")
        with p2:
            max_rec_attempts = st.number_input("max_rec_attempts", min_value=1, max_value=10, value=3,
                                               help="Max recommendation rounds before ending conversation")
        with p3:
            rec_top_n = st.number_input("rec_top_n", min_value=1, max_value=10, value=3,
                                        help="Number of items recommended per round")
        with p4:
            max_turns = st.number_input("max_turns", min_value=2, max_value=30, value=10,
                                        help="Max dialogue turns per conversation")

    with st.expander("Quality & diversity parameters"):
        q1, q2, q3 = st.columns(3)
        with q1:
            chit_chat_ratio        = st.slider("chit_chat_ratio", 0.0, 1.0, 0.6, 0.05)
            rejection_exp_ratio    = st.slider("rejection_explanation_ratio", 0.0, 1.0, 0.8, 0.05)
        with q2:
            rec_exp_ratio          = st.slider("rec_explanation_ratio", 0.0, 1.0, 0.5, 0.05)
            short_reply_ratio      = st.slider("short_reply_ratio", 0.0, 1.0, 0.45, 0.05)
        with q3:
            bot_short_ratio        = st.slider("bot_short_followup_ratio", 0.0, 1.0, 0.4, 0.05)
            weight_es              = st.slider("weight_es_score (α)", 0.0, 1.0, 0.5, 0.05)

    with st.expander("LLM temperature parameters"):
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            temp_rephrase    = st.slider("temp_rephrase", 0.0, 2.0, 0.8, 0.1)
        with t2:
            temp_rejection   = st.slider("temp_rejection", 0.0, 2.0, 0.9, 0.1)
        with t3:
            temp_rec_explain = st.slider("temp_rec_explain", 0.0, 2.0, 0.9, 0.1)
        with t4:
            temp_free_user   = st.slider("temp_free_user", 0.0, 2.0, 1.0, 0.1)

    # ── Build override YAML & launch ─────────────────────────
    st.subheader("4️⃣  Launch")

    run_btn = st.button("🚀 Start Simulation", type="primary", key="run_start_btn")

    if run_btn:
        # Write a temporary override YAML
        override = {
            "chats_per_entry":              int(chats_per_entry),
            "max_rec_attempts":             int(max_rec_attempts),
            "rec_top_n":                    int(rec_top_n),
            "max_turns":                    int(max_turns),
            "chit_chat_ratio":              float(chit_chat_ratio),
            "rejection_explanation_ratio":  float(rejection_exp_ratio),
            "rec_explanation_ratio":        float(rec_exp_ratio),
            "short_reply_ratio":            float(short_reply_ratio),
            "bot_short_followup_ratio":     float(bot_short_ratio),
            "weight_es_score":              float(weight_es),
            "temp_rephrase":                float(temp_rephrase),
            "temp_rejection":               float(temp_rejection),
            "temp_rec_explain":             float(temp_rec_explain),
            "temp_free_user":               float(temp_free_user),
        }
        if strategy == "user_item_pairs":
            override["input_pairs_file"] = os.path.join(inputs_dir, pairs_file)
        elif strategy == "item_list":
            override["input_items_file"] = os.path.join(inputs_dir, items_file)
        elif strategy == "user_guest":
            override["input_users_file"] = os.path.join(inputs_dir, users_file)
        elif strategy == "random items":
            override["guestuser_randomitem_records"] = int(random_n)

        override_path = "/tmp/run_override.yaml"
        with open(override_path, "w") as f:
            yaml.dump(override, f)

        dataset_yaml = os.path.join("configs", dataset, f"{dataset}.yaml")
        config_str   = f"configs/system/system.yaml {dataset_yaml} {override_path}"
        log_path     = f"/tmp/run_{dataset}_{mode}.log"

        cmd = [sys.executable, "run.py", "--config", config_str, "--mode", mode]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

        st.session_state["sim_pid"]      = proc.pid
        st.session_state["sim_log"]      = log_path
        st.session_state["sim_running"]  = True
        st.session_state["sim_dataset"]  = dataset
        st.session_state["sim_mode"]     = mode
        st.session_state["sim_start_ts"] = time.time()
        st.rerun()

    # ── Poll running simulation ───────────────────────────────
    if st.session_state.get("sim_running"):
        pid      = st.session_state["sim_pid"]
        log_path = st.session_state["sim_log"]
        ds       = st.session_state.get("sim_dataset", "")
        md       = st.session_state.get("sim_mode", "")
        elapsed  = int(time.time() - st.session_state.get("sim_start_ts", time.time()))

        still_running = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            still_running = False

        log_text = ""
        if os.path.exists(log_path):
            with open(log_path) as lf:
                log_text = lf.read()

        done  = log_text.count("Generated") + log_text.count("conversation written")
        total_match = __import__("re").search(r"(\d+) total", log_text)
        total = int(total_match.group(1)) if total_match else None

        st.info(f"🔄 Running **{ds}/{md}** simulation  |  elapsed **{elapsed}s**")
        if total:
            pct = min(done / total, 1.0)
            st.progress(pct, text=f"{done} / {total} conversations done")
        else:
            st.progress(0.0, text="Initialising…")

        # Last log line
        log_lines = [l for l in log_text.splitlines() if l.strip()]
        if log_lines:
            st.caption(f"Latest: {log_lines[-1].strip()}")

        if still_running:
            time.sleep(4)
            st.rerun()
        else:
            st.session_state["sim_running"] = False
            if "error" in log_text.lower() and "done" not in log_text.lower():
                st.error("❌ Simulation encountered errors.")
            else:
                st.success(f"✅ Simulation complete! Conversations saved to `chats/{ds}/{md}/`")
            with st.expander("📄 Full log"):
                st.code(log_text, language="text")
            st.rerun()

    st.divider()
    with st.expander("📋 Run Logs", expanded=False):
        # simplified log viewer
        log_path = st.session_state.get("sim_log", "")
        if log_path and os.path.exists(log_path):
            with open(log_path, encoding='utf-8', errors='replace') as f:
                log_text = f.read()
            log_lines = log_text.splitlines()
            n_lines = st.number_input("Last N lines", min_value=10, max_value=500, value=50, step=10, key="log_n_lines")
            tail = log_lines[-int(n_lines):]
            html_lines = []
            for line in tail:
                line_s = line.rstrip()
                if any(k in line_s for k in ['DONE', 'Done', 'complete', 'written']):
                    html_lines.append(f'<div style="background:#F0FFF4;color:#2d6a2d;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
                elif any(k in line_s for k in ['ERROR', 'Error', 'Traceback', 'WARN']):
                    html_lines.append(f'<div style="background:#FFF0F0;color:#a00;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
                elif any(k in line_s for k in ['Running', 'Strategy', 'INFO', '===']):
                    html_lines.append(f'<div style="background:#EEF4FF;color:#2244aa;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
                else:
                    html_lines.append(f'<div style="padding:1px 8px;font-family:monospace;font-size:0.82em;color:#444">{line_s}</div>')
            st.html('<div style="background:#fafafa;border:1px solid #ddd;border-radius:6px;padding:8px;max-height:400px;overflow-y:auto">' + '\n'.join(html_lines) + '</div>')
        else:
            # Show run status table
            st.markdown("**Run Status**")
            rows = []
            for ds in ['imdb', 'hm']:
                for md in ['free', 'static', 'adaptive']:
                    n_convs = len([
                        f for root, _, files in os.walk(f"chats/{ds}/{md}")
                        for f in files if f.endswith('.txt') and 'tmp' not in f
                    ]) if os.path.isdir(f"chats/{ds}/{md}") else 0
                    log = f"/tmp/run_{ds}_{md}.log"
                    if not os.path.exists(log):
                        status = "⏳ Pending"
                    else:
                        with open(log, errors='replace') as lf:
                            lc = lf.read()
                        if 'complete' in lc.lower() or 'conversations written' in lc.lower():
                            status = "✅ Done"
                        elif 'ERROR' in lc or 'Traceback' in lc:
                            status = "❌ Error"
                        else:
                            status = "🔄 Running"
                    rows.append({'Dataset': ds.upper(), 'Mode': md.capitalize(), 'Status': status, 'Conversations': n_convs})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_logs():
    st.title("📋 Run Logs")

    LOG_FILES = {
        "Overall Progress":    "/tmp/run_all.log",
        "IMDB / Free":         "/tmp/run_imdb_free.log",
        "IMDB / Static":       "/tmp/run_imdb_static.log",
        "IMDB / Adaptive":     "/tmp/run_imdb_adaptive.log",
        "HM / Free":           "/tmp/run_hm_free.log",
        "HM / Static":         "/tmp/run_hm_static.log",
        "HM / Adaptive":       "/tmp/run_hm_adaptive.log",
        "Streamlit":           "/tmp/streamlit.log",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Select log", list(LOG_FILES.keys()))
    with col2:
        n_lines = st.number_input("Last N lines", min_value=10, max_value=500, value=50, step=10)
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    if auto_refresh:
        import time
        st.caption("Auto-refreshing every 5 seconds…")
        time.sleep(5)
        st.rerun()

    log_path = LOG_FILES[selected]
    st.caption(f"`{log_path}`")

    if not os.path.exists(log_path):
        st.info(f"Log file not found: `{log_path}`")
    else:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total = len(lines)
        tail = lines[-n_lines:]
        content = ''.join(tail)

        st.caption(f"Total lines: {total}  |  Showing last {min(n_lines, total)}")

        # Colour-code key lines
        html_lines = []
        for line in tail:
            line_s = line.rstrip()
            if any(k in line_s for k in ['DONE', 'ALL DONE', 'Done', 'Inserted']):
                html_lines.append(f'<div style="background:#F0FFF4;color:#2d6a2d;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
            elif any(k in line_s for k in ['ERROR', 'Error', 'Traceback', 'Exception', 'WARN']):
                html_lines.append(f'<div style="background:#FFF0F0;color:#a00;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
            elif any(k in line_s for k in ['Running mode', 'Strategy', 'Strategies', 'INFO', '▶▶', '===']):
                html_lines.append(f'<div style="background:#EEF4FF;color:#2244aa;padding:2px 8px;font-family:monospace;font-size:0.85em">{line_s}</div>')
            else:
                html_lines.append(f'<div style="padding:1px 8px;font-family:monospace;font-size:0.82em;color:#444">{line_s}</div>')

        st.html(
            '<div style="background:#fafafa;border:1px solid #ddd;border-radius:6px;'
            'padding:8px;max-height:600px;overflow-y:auto">'
            + '\n'.join(html_lines)
            + '</div>'
        )

        st.divider()

    # ── Summary table: which runs have completed ──────────────
    st.subheader("Run Status")
    rows = []
    for dataset in ['imdb', 'hm']:
        for mode in ['free', 'static', 'adaptive']:
            log = f"/tmp/run_{dataset}_{mode}.log"
            n_convs = len([
                f for root, _, files in os.walk(f"chats/{dataset}/{mode}")
                for f in files
                if f.endswith('.txt') and 'tmp' not in f
            ]) if os.path.isdir(f"chats/{dataset}/{mode}") else 0
            if not os.path.exists(log):
                status = "⏳ Pending"
            else:
                with open(log, errors='replace') as lf:
                    content_check = lf.read()
                if 'Simulation complete' in content_check or 'All strategies done' in content_check or 'conversations written' in content_check.lower():
                    status = "✅ Done"
                elif 'ERROR' in content_check or 'Traceback' in content_check:
                    status = "❌ Error"
                else:
                    status = "🔄 Running"
            rows.append({'Dataset': dataset.upper(), 'Mode': mode.capitalize(),
                         'Status': status, 'Conversations': n_convs})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Page: ReDial Statistics
# ─────────────────────────────────────────────────────────────
def page_redial_stats():
    import json as _json, numpy as _np, re as _re

    st.title("📚 ReDial — Real Data Statistics")
    st.markdown(
        "Full statistics of the **ReDial dataset** (real human movie recommendation conversations). "
        "Use these distributions as the **reference target** when configuring simulation parameters."
    )

    REDIAL_JSONL = ["data/redial/train_data.jsonl", "data/redial/test_data.jsonl"]
    _RQ_RE = _re.compile(r'@(\d+)')

    rows = []
    for jsonl_path in REDIAL_JSONL:
        if not os.path.exists(jsonl_path):
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = _json.loads(line)
                msgs = d.get("messages", [])
                if not msgs:
                    continue
                rq = d.get("respondentQuestions", {})
                if not isinstance(rq, dict):
                    rq = {}
                suggested     = [mid for mid, q in rq.items() if q.get("suggested", 0) == 1]
                liked_sug     = [mid for mid in suggested if rq[mid].get("liked", 0) == 1]
                succeed       = 1 if liked_sug else 0
                rec_attempts  = len(suggested)

                init_id = d["initiatorWorkerId"]
                resp_id = d["respondentWorkerId"]
                bot_rec_turns = []
                for msg in msgs:
                    if msg.get("senderWorkerId") == resp_id:
                        refs = _RQ_RE.findall(msg.get("text", ""))
                        mentioned = [r for r in refs if r in suggested]
                        if mentioned:
                            bot_rec_turns.append(len(mentioned))
                items_per_rec = sum(bot_rec_turns) / len(bot_rec_turns) if bot_rec_turns else 0

                rows.append({
                    "num_turns":    len(msgs),
                    "rec_attempts": rec_attempts,
                    "succeed":      succeed,
                    "items_per_rec": items_per_rec,
                })

    if not rows:
        st.info("ReDial JSONL not found in `data/redial/`.")
        return

    df = pd.DataFrame(rows)

    # ── Key metrics ───────────────────────────────────────────
    st.subheader("📊 Key Metrics")
    c = st.columns(5)
    c[0].metric("Total Conversations", f"{len(df):,}")
    c[1].metric("Mean Turns",          f"{df.num_turns.mean():.1f} ± {df.num_turns.std():.1f}")
    c[2].metric("Mean Rec Attempts",   f"{df.rec_attempts.mean():.1f} ± {df.rec_attempts.std():.1f}")
    c[3].metric("Items per Rec Turn",  f"{df.items_per_rec.mean():.2f}")
    c[4].metric("Success Rate",        f"{df.succeed.mean():.1%}")

    st.divider()

    # ── Descriptive stats table ───────────────────────────────
    st.subheader("📋 Descriptive Statistics")
    desc_cols = {"num_turns": "Num Turns", "rec_attempts": "Rec Attempts", "items_per_rec": "Items/Rec"}
    desc_rows = []
    for col, label in desc_cols.items():
        s = df[col]
        desc_rows.append({
            "Metric":  label,
            "Mean":    round(s.mean(), 2),
            "Std":     round(s.std(), 2),
            "Min":     round(s.min(), 2),
            "Q1":      round(s.quantile(0.25), 2),
            "Median":  round(s.median(), 2),
            "Q3":      round(s.quantile(0.75), 2),
            "Max":     round(s.max(), 2),
        })
    st.dataframe(pd.DataFrame(desc_rows).set_index("Metric"), use_container_width=True)

    st.caption("💡 Use these values as targets when setting `rec_top_n`, `max_rec_attempts` in `configs/system/system.yaml`.")
    st.divider()

    # ── Distribution plots ────────────────────────────────────
    st.subheader("📈 Distributions")
    p1, p2, p3 = st.columns(3)

    for col_widget, col, label, bins in [
        (p1, "num_turns",    "Num Turns",    40),
        (p2, "rec_attempts", "Rec Attempts", 15),
        (p3, "items_per_rec","Items/Rec",    10),
    ]:
        fig = go.Figure(go.Histogram(
            x=df[col], nbinsx=bins,
            marker_color="#888888", opacity=0.8,
        ))
        fig.update_layout(
            title=label, height=300,
            margin=dict(t=40, b=20),
            xaxis_title=label, yaxis_title="Count",
        )
        col_widget.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Success rate breakdown ────────────────────────────────
    st.subheader("✅ Success Rate")
    succ_cols = st.columns(3)
    succ_cols[0].metric("Successful",    f"{df.succeed.sum():,} ({df.succeed.mean():.1%})")
    succ_cols[1].metric("Unsuccessful",  f"{(1-df.succeed).sum():,} ({1-df.succeed.mean():.1%})")
    succ_cols[2].metric("Avg Rec Attempts (success)", f"{df[df.succeed==1].rec_attempts.mean():.1f}")


def page_data_distribution():
    import re as _re

    st.title("📐 Data Distribution — Simulated vs Real")
    st.markdown(
        "Structural comparison of conversation statistics across all groups "
        "(**Num Turns**, **Rec Attempts**, **Items per Rec Turn**, **Success Rate**). "
        "This validates whether simulated and real conversations are structurally comparable "
        "before quality evaluation."
    )

    # ── Scan all folders ──────────────────────────────────────
    GROUPS = {
        "Free/LLama":     ("chats/llama/imdb/free/user_item_pairs",     True),
        "Static/LLama":   ("chats/llama/imdb/static/user_item_pairs",   True),
        "Adaptive/LLama": ("chats/llama/imdb/adaptive/user_item_pairs", True),
        "Free/GPT":       ("chats/gpt/imdb/free/user_item_pairs",       True),
        "Static/GPT":     ("chats/gpt/imdb/static/user_item_pairs",     True),
        "Adaptive/GPT":   ("chats/gpt/imdb/adaptive/user_item_pairs",   True),
        "ReDial (Real)":  ("chats/redial/real",                       False),
    }
    GROUP_COLORS = {
        "Free (OmniSim)":     "#4C72B0",
        "Static (OmniSim)":   "#55A868",
        "Adaptive (OmniSim)": "#C44E52",
        "ReDial (Real)":       "#888888",
    }
    _FNAME_RE   = _re.compile(r'.*?-(\d+)-(\d+)-([01])-\d+\.txt$')
    _QUOTE_RE   = _re.compile(r'"[^"]+"')          # ReDial: "Movie Title"
    _BULLET_RE  = _re.compile(r'^\s*-\s+.+\(tt\d+\)')  # Simulated: - Title (ttXXXX)

    def _latest_folder(path):
        if not os.path.isdir(path):
            return path
        subs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))], reverse=True)
        return os.path.join(path, subs[0]) if subs else path

    def _load_group(path, use_subdir=True):
        folder = _latest_folder(path) if use_subdir else path
        rows = []
        for fname in os.listdir(folder):
            if not fname.endswith('.txt'):
                continue
            m = _FNAME_RE.search(fname)
            if not m:
                continue
            turns, rec_att, succeed = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Count items per rec attempt
            # Simulated: bullet lines "- Title (ttXXXX)" grouped after each rec block
            # ReDial: quoted titles in Bot lines
            total_items = 0
            in_rec_block = False
            block_items = 0
            rec_blocks = 0
            try:
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    if _BULLET_RE.match(line):
                        # simulated bullet item
                        block_items += 1
                        in_rec_block = True
                    else:
                        if in_rec_block and block_items > 0:
                            total_items += block_items
                            rec_blocks += 1
                            block_items = 0
                        in_rec_block = False
                        if line.startswith('Bot:') and '"' in line:
                            # ReDial quoted titles
                            quoted = len(_QUOTE_RE.findall(line))
                            if quoted > 0:
                                total_items += quoted
                                rec_blocks += 1
                if in_rec_block and block_items > 0:
                    total_items += block_items
                    rec_blocks += 1
            except Exception:
                pass
            avg_items = total_items / rec_blocks if rec_blocks > 0 else 0
            rows.append({
                'num_turns':    turns,
                'rec_attempts': rec_att,
                'succeed':      succeed,
                'avg_items':    avg_items,
            })
        return pd.DataFrame(rows)

    group_dfs = {}
    for name, (path, use_subdir) in GROUPS.items():
        df = _load_group(path, use_subdir)
        if not df.empty:
            group_dfs[name] = df

    if not group_dfs:
        st.info("No conversation data found.")
        return

    # ── Summary table ─────────────────────────────────────────
    st.subheader("📋 Summary Statistics")
    sum_rows = []
    for name, df in group_dfs.items():
        sum_rows.append({
            "Group":        name,
            "N":            str(len(df)),
            "Turns (mean±std)":   f"{df['num_turns'].mean():.1f} ± {df['num_turns'].std():.1f}",
            "Rec Attempts (mean±std)": f"{df['rec_attempts'].mean():.1f} ± {df['rec_attempts'].std():.1f}",
            "Items/Rec Turn": f"{df['avg_items'].mean():.2f}",
            "Success Rate":  f"{df['succeed'].mean():.1%}",
        })
    st.dataframe(pd.DataFrame(sum_rows).set_index("Group"), use_container_width=True)

    st.divider()

    # ── Distribution plots ────────────────────────────────────
    st.subheader("📊 Distribution Plots")

    def _box(col, title, yaxis_title):
        fig = go.Figure()
        for name, df in group_dfs.items():
            if col not in df.columns:
                continue
            fig.add_trace(go.Box(
                y=df[col], name=name,
                marker_color=GROUP_COLORS.get(name, "#888"),
                boxpoints="outliers",
            ))
        fig.update_layout(
            title=title, height=360,
            margin=dict(t=40, b=20),
            yaxis_title=yaxis_title,
            showlegend=False,
        )
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(_box("num_turns",    "Num Turns",    "Turns"),           use_container_width=True)
    c2.plotly_chart(_box("rec_attempts", "Rec Attempts", "Rec Attempts"),    use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(_box("avg_items",    "Items per Rec Turn", "Items"),     use_container_width=True)

    # Success rate bar
    with c4:
        fig_s = go.Figure(go.Bar(
            x=list(group_dfs.keys()),
            y=[df['succeed'].mean() * 100 for df in group_dfs.values()],
            marker_color=[GROUP_COLORS.get(n, "#888") for n in group_dfs],
            text=[f"{df['succeed'].mean():.1%}" for df in group_dfs.values()],
            textposition="outside",
        ))
        fig_s.update_layout(
            title="Success Rate (%)", height=360,
            margin=dict(t=40, b=60),
            yaxis=dict(range=[0, 110], title="Success Rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.divider()

    # ── Histogram: turns ──────────────────────────────────────
    st.subheader("📉 Turns Distribution Histogram")
    fig_h = go.Figure()
    for name, df in group_dfs.items():
        fig_h.add_trace(go.Histogram(
            x=df['num_turns'], name=name,
            marker_color=GROUP_COLORS.get(name, "#888"),
            opacity=0.6, nbinsx=20,
        ))
    fig_h.update_layout(
        barmode="overlay", height=360,
        xaxis_title="Num Turns", yaxis_title="Count",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # ── KS test ───────────────────────────────────────────────
    from scipy import stats as _stats
    st.subheader("📐 Distribution Similarity (KS Test — Num Turns)")
    st.markdown("p > 0.05 = distributions are not significantly different (comparable).")
    ks_rows = []
    names = list(group_dfs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            ks, p = _stats.ks_2samp(group_dfs[a]['num_turns'], group_dfs[b]['num_turns'])
            ks_rows.append({
                "Group A": a, "Group B": b,
                "KS Stat": round(ks, 3),
                "p-value": round(p, 4),
                "Comparable?": "✅ Yes" if p > 0.05 else "❌ No",
            })
    st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)


def page_eval_data_stats():
    import json as _json, re as _re

    st.title("📈 Eval: Data Statistics")
    st.markdown(
        "Overview of conversation statistics across the **ReDial dataset**, "
        "**OmniSim generated dialogues** (IMDB, 3 modes), and the **matched ReDial redial** used for comparison."
    )

    # ── Section 1: ReDial Full Dataset ────────────────────────
    st.markdown("---")
    st.subheader("Section 1: Whole Redial Data (ReDial)")
    st.markdown("Full statistics of the **ReDial dataset** (~11K real human movie recommendation conversations).")

    REDIAL_JSONL = ["data/redial/train_data.jsonl", "data/redial/test_data.jsonl"]
    _RQ_RE = _re.compile(r'@(\d+)')

    rows = []
    for jsonl_path in REDIAL_JSONL:
        if not os.path.exists(jsonl_path):
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = _json.loads(line)
                msgs = d.get("messages", [])
                if not msgs:
                    continue
                rq = d.get("respondentQuestions", {})
                if not isinstance(rq, dict):
                    rq = {}
                suggested    = [mid for mid, q in rq.items() if q.get("suggested", 0) == 1]
                liked_sug    = [mid for mid in suggested if rq[mid].get("liked", 0) == 1]
                succeed      = 1 if liked_sug else 0
                rec_attempts = len(suggested)
                resp_id = d["respondentWorkerId"]
                bot_rec_turns = []
                for msg in msgs:
                    if msg.get("senderWorkerId") == resp_id:
                        refs = _RQ_RE.findall(msg.get("text", ""))
                        mentioned = [r for r in refs if r in suggested]
                        if mentioned:
                            bot_rec_turns.append(len(mentioned))
                items_per_rec = sum(bot_rec_turns) / len(bot_rec_turns) if bot_rec_turns else 0
                rows.append({"num_turns": len(msgs), "rec_attempts": rec_attempts, "succeed": succeed, "items_per_rec": items_per_rec})

    if rows:
        df = pd.DataFrame(rows)
        c = st.columns(5)
        c[0].metric("Total Conversations", f"{len(df):,}")
        c[1].metric("Mean Turns",          f"{df.num_turns.mean():.1f} ± {df.num_turns.std():.1f}")
        c[2].metric("Mean Rec Attempts",   f"{df.rec_attempts.mean():.1f} ± {df.rec_attempts.std():.1f}")
        c[3].metric("Items per Rec Turn",  f"{df.items_per_rec.mean():.2f}")
        c[4].metric("Success Rate",        f"{df.succeed.mean():.1%}")
        st.divider()
        desc_cols = {"num_turns": "Num Turns", "rec_attempts": "Rec Attempts", "items_per_rec": "Items/Rec"}
        desc_rows = []
        for col, label in desc_cols.items():
            s = df[col]
            desc_rows.append({"Metric": label, "Mean": round(s.mean(),2), "Std": round(s.std(),2), "Min": round(s.min(),2), "Q1": round(s.quantile(0.25),2), "Median": round(s.median(),2), "Q3": round(s.quantile(0.75),2), "Max": round(s.max(),2)})
        st.dataframe(pd.DataFrame(desc_rows).set_index("Metric"), use_container_width=True)
        p1, p2 = st.columns(2)
        for col_widget, col, label, bins in [(p1,"num_turns","Num Turns",40),(p2,"rec_attempts","Rec Attempts",15)]:
            fig = go.Figure(go.Histogram(x=df[col], nbinsx=bins, marker_color="#888888", opacity=0.8))
            fig.update_layout(title=label, height=280, margin=dict(t=40, b=20), xaxis_title=label, yaxis_title="Count")
            col_widget.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ReDial JSONL not found in `data/redial/`.")

    # ── Section 2: Matched ReDial Baseline ────────────────
    st.markdown("---")
    st.subheader("Section 2: Sampled ReDial Data (ReDial Sample)")
    st.markdown("Statistics for the ReDial redial sample (`chats/redial/real/`) used in NLP comparison.")

    baseline_folder = os.path.join("chats", "redial", "real")
    _BQUOTED_RE = re.compile(r'"[^"]+"')
    if os.path.isdir(baseline_folder):
        brows = []
        for fname in os.listdir(baseline_folder):
            if not fname.endswith('.txt') or 'tmp' in fname:
                continue
            p = parse_filename(fname)
            if not p:
                continue
                p['items_per_rec'] = 1
            brows.append(p)
        if brows:
            bdf = pd.DataFrame(brows)
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total", len(bdf))
            bc2.metric("Success Rate", f"{bdf['succeed'].mean():.1%}")
            bc3.metric("Avg Turns", f"{bdf['num_turns'].mean():.1f}")
            bc4.metric("Avg Rec Attempts", f"{bdf['rec_attempts'].mean():.1f}")
            st.dataframe(pd.DataFrame([{
                "Group": "ReDial Sample", "N": str(len(bdf)),
                "Success Rate": f"{bdf['succeed'].mean():.1%}",
                "Turns (mean±std)": f"{bdf['num_turns'].mean():.1f} ± {bdf['num_turns'].std():.1f}",
                "Rec Attempts (mean±std)": f"{bdf['rec_attempts'].mean():.1f} ± {bdf['rec_attempts'].std():.1f}",
                "Items/Rec (mean)": "1",
            }]).set_index("Group"), use_container_width=True)
        else:
            st.info("No redial conversation files found.")
    else:
        st.info("Baseline folder `chats/redial/real/` not found.")

    # ── Section 3: OmniSim Generated Dialogues ────────────
    st.markdown("---")
    st.subheader("Section 3: Simulated Dialogues by OmniSim (IMDB, 3 Modes)")
    st.markdown("Statistics for the latest run of each mode under `chats/llama/imdb/`.")

    def _load_mode_df(mode):
        path = os.path.join("chats", "llama", "imdb", mode, "user_item_pairs")
        if not os.path.isdir(path):
            return None
        runs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))], reverse=True)
        # Items are recommended as quoted titles in Bot: lines, e.g. "Movie Title"
        _QUOTED_ITEM_RE = re.compile(r'"[^"]+"')
        for run in runs:
            folder = os.path.join(path, run)
            mrows = []
            for fname in os.listdir(folder):
                if not fname.endswith('.txt') or 'tmp' in fname:
                    continue
                p = parse_filename(fname)
                if not p:
                    continue
                p['items_per_rec'] = 1
                mrows.append(p)
            if mrows:
                return pd.DataFrame(mrows)
        return None

    mode_dfs = {mode: _load_mode_df(mode) for mode in MODES}
    mode_dfs = {k: v for k, v in mode_dfs.items() if v is not None}

    if mode_dfs:
        sum_rows3 = []
        for mode, mdf in mode_dfs.items():
            sum_rows3.append({
                "Mode": mode.capitalize(), "N": str(len(mdf)),
                "Success Rate": f"{mdf['succeed'].mean():.1%}",
                "Turns (mean±std)": f"{mdf['num_turns'].mean():.1f} ± {mdf['num_turns'].std():.1f}",
                "Rec Attempts (mean±std)": f"{mdf['rec_attempts'].mean():.1f} ± {mdf['rec_attempts'].std():.1f}",
                "Items/Rec (mean)": "1",
            })
        st.dataframe(pd.DataFrame(sum_rows3).set_index("Mode"), use_container_width=True)
        mc1, mc2, mc3 = st.columns(3)
        for col_widget, col, label in [(mc1,"num_turns","Num Turns"), (mc2,"rec_attempts","Rec Attempts"), (mc3,"items_per_rec","Items/Rec")]:
            fig = go.Figure()
            for mode, mdf in mode_dfs.items():
                fig.add_trace(go.Box(y=mdf[col], name=mode.capitalize(), marker_color=MODE_COLORS[mode], boxpoints="outliers"))
            fig.update_layout(title=label, height=300, margin=dict(t=40, b=20), yaxis_title=label)
            col_widget.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No OmniSim conversations found. Run simulations first.")


# ─────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("OmniSim")
    st.caption("LLM-Powered Conversational Recommendation Dialogue Simulator")
    st.divider()
    page = st.radio("Navigation", [
        "🛠️ Features",
        "💬 Conversation Browser",
        "📈 Eval: Data Statistics",
        "🔍 Eval: NLP Metrics",
        "⚖️ Eval: LLM-as-Judge",
    ])

if page == "🛠️ Features":
    page_features()
elif page == "💬 Conversation Browser":
    page_conversation_browser()
elif page == "📈 Eval: Data Statistics":
    page_eval_data_stats()
elif page == "🔍 Eval: NLP Metrics":
    page_nlp_comparison()
elif page == "⚖️ Eval: LLM-as-Judge":
    page_judge()
