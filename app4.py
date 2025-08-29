cd /Users/miyuanqi/Desktop/redcap_rds_app

git init
git branch -M main
git remote add origin https://github.com/Yuanqi-Mi/RDS-Automatic.git


git fetch origin
git pull origin main --allow-unrelated-histories


git add app.py compute_rds_weights.R requirements.txt .gitignore README.md
git commit -m "Add RDS Tidy Wave app"


git push -u origin main

# app.py — REDCap RDS Tree Automata (Tidy Wave Only, English UI)
import io
import os
import tempfile
import subprocess

from collections import defaultdict, deque
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
import requests
import streamlit as st

st.set_page_config(page_title="REDCap RDS Tree Automata", layout="wide")


# ==============================
# Helpers
# ==============================
def fmt_id(n) -> str:
    """Format node ID as an integer-like string (strip trailing .0 etc.)."""
    s = str(n).strip()
    if s.endswith(".0"):
        p = s[:-2]
        if p.replace("-", "").isdigit():
            return p
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def to_downloadable_html(fig, filename_label: str):
    """Render a Plotly figure as downloadable HTML."""
    html = to_html(fig, include_plotlyjs="cdn", full_html=True)
    st.download_button(
        f"Download {filename_label} (HTML)",
        data=html.encode("utf-8"),
        file_name=f"{filename_label.replace(' ', '_').lower()}.html",
        mime="text/html",
    )


# ==============================
# A) Upload + schema inference
# ==============================
def infer_schema(df: pd.DataFrame):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    def pick_one(cands):
        for pat in cands:
            for lc, orig in lower.items():
                if pat in lc:
                    return orig
        return None

    in_field = pick_one(
        [
            "inpon_number",
            "inpon",
            "inc",
            "incoming",
            "coupon_in",
            "in_coupon",
            "recruit_id",
            "participant_id",
        ]
    )
    seed_field = pick_one(["seed_id", "seed"])
    networksize_field = pick_one(
        ["networksize", "network_size", "degree", "net_size", "personal_network"]
    )

    out_fields = []
    for c in cols:
        lc = c.lower()
        if any(
            lc.startswith(p)
            for p in ["outpons_", "outpon", "out_coupon", "coupon_out", "recruit", "out_"]
        ):
            out_fields.append(c)
    if not out_fields:
        out_fields = [
            c
            for c in cols
            if any(x in c.lower() for x in ["out1", "out2", "out3", "out4", "out5", "out6"])
        ]
    return {
        "in_field": in_field,
        "seed_field": seed_field,
        "out_fields": out_fields[:10],
        "networksize_field": networksize_field,
    }


def read_uploaded_file(uploaded_file, sep=None, sheet=None):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded_file)
        use_sheet = xls.sheet_names[0] if sheet is None else sheet
        return pd.read_excel(xls, use_sheet)
    content = uploaded_file.read()
    sample = content[:10000].decode("utf-8", errors="ignore")
    sep = "\t" if "\t" in sample and sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(io.BytesIO(content), sep=sep)


# ==============================
# B) Build graph / attributes
# ==============================
def fetch_redcap(api_url: str, token: str) -> pd.DataFrame:
    payload = {"token": token, "content": "record", "format": "json", "type": "flat"}
    r = requests.post(api_url, data=payload)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def build_graph(df: pd.DataFrame, in_field: str, seed_field: str, out_fields: list):
    """
    Build a directed recruitment graph.

    CRITICAL RULE:
    - A "person/node" must appear in incoming coupon (incoupon) OR as a seed.
    - Coupons that appear only in outpon* but never in incoupon/seed are NOT persons.
    """
    df = df.copy()
    df["inc_code"] = df.apply(
        lambda row: row[in_field]
        if pd.notna(row[in_field]) and str(row[in_field]).strip() != ""
        else row[seed_field],
        axis=1,
    ).astype(str)

    edges = []
    for out in out_fields:
        if out not in df.columns:
            continue
        valid = df[df[out].notna() & (df[out].astype(str).str.strip() != "")]
        edges += [(row["inc_code"], str(row[out])) for _, row in valid.iterrows()]

    valid_nodes = set(df["inc_code"].astype(str))  # ONLY real persons

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G.subgraph(valid_nodes).copy()  # drop edges to non-person coupons


def compute_network_size(
    G: nx.DiGraph, df: pd.DataFrame, seed_field: str, in_field: str, networksize_field: str
):
    if networksize_field not in df.columns:
        raise ValueError(f"networksize field '{networksize_field}' not in dataframe.")

    reported_map = (
        df.assign(
            inc_code=lambda d: d[in_field].where(
                d[in_field].notna() & (d[in_field].astype(str).str.strip() != ""),
                d[seed_field],
            )
        )
        .astype({"inc_code": str})
        .set_index("inc_code")[networksize_field]
        .apply(pd.to_numeric, errors="coerce")
        .to_dict()
    )

    seeds_in_df = set(df[seed_field].dropna().astype(str)) if seed_field in df.columns else set()

    ns = {}
    for node in G.nodes():
        out_deg = G.out_degree(node)  # only counts recruits who became real persons
        networksize_in_tree = out_deg + 1
        rep = reported_map.get(node)
        reported_networksize = int(rep) if pd.notna(rep) else networksize_in_tree
        ns[node] = {
            "is_seed": node in seeds_in_df,
            "networksize_in_tree": networksize_in_tree,
            "reported_networksize": reported_networksize,
        }
    return ns


def compute_wave(G, seed):
    wave = {seed: 0}
    for child in nx.bfs_tree(G, seed):
        if child == seed:
            continue
        parents = list(G.predecessors(child))
        wave[child] = wave[parents[0]] + 1 if parents else 0
    return wave


def bfs_subgraph_upto_wave(G, seed, max_wave=None):
    if max_wave is None:
        nodes = list(nx.bfs_tree(G, seed).nodes())
    else:
        nodes = [
            n
            for n, d in nx.single_source_shortest_path_length(G, seed).items()
            if d <= max_wave
        ]
    return G.subgraph(nodes).copy()


def seeds_from_prefix(df: pd.DataFrame, seed_field: str, site_prefix: str) -> set:
    if seed_field not in df.columns:
        return set()
    return set(
        df[df[seed_field].astype(str).str.startswith(site_prefix)][seed_field].astype(str)
    )


def max_wave_of_graph(G: nx.DiGraph, seeds) -> int:
    maxw = 0
    for s in seeds:
        if s in G:
            depths = nx.single_source_shortest_path_length(G, s)
            if depths:
                maxw = max(maxw, max(depths.values()))
    return maxw


# ==============================
# C) Layout: Layered (Tidy wave)
# ==============================
def _tree_children(G: nx.DiGraph, seed):
    T = nx.bfs_tree(G, seed)
    ch = {n: [] for n in T.nodes()}
    for u, v in T.edges():
        ch[u].append(v)
    return ch, T.nodes()


def _subtree_size(children, node):
    if not children.get(node):
        return 1
    return 1 + sum(_subtree_size(children, c) for c in children[node])


def _tidy_assign_x(children, node, depth, x_cursor, x_map, y_map, order_by="size"):
    kids = children.get(node, [])
    if order_by == "size":
        kids = sorted(kids, key=lambda k: (-_subtree_size(children, k), str(k)))
    else:
        kids = sorted(kids, key=lambda k: str(k))

    if not kids:
        x_map[node] = x_cursor[0]
        y_map[node] = depth
        x_cursor[0] += 1
    else:
        xs = []
        for k in kids:
            _tidy_assign_x(children, k, depth + 1, x_cursor, x_map, y_map, order_by=order_by)
            xs.append(x_map[k])
        x_map[node] = float(sum(xs)) / len(xs)
        y_map[node] = depth


def _layout_tidy_one_tree(G, seed, node_gap=1.0, layer_gap=1.0):
    children, nodes = _tree_children(G, seed)
    depth = nx.single_source_shortest_path_length(G, seed)

    x_map, y_map, x_cursor = {}, {}, [0]
    _tidy_assign_x(children, seed, 0, x_cursor, x_map, y_map)

    min_x = min(x_map.values()) if x_map else 0
    pos = {}
    for n in nodes:
        x = (x_map[n] - min_x) * node_gap
        y = -float(y_map.get(n, depth.get(n, 0))) * layer_gap  # seed at top (y=0)
        pos[n] = np.array([x, y], dtype=float)
    width = (max(x_map.values()) - min_x + 1) * node_gap if x_map else 0
    return pos, width


def layout_layered_tidy_forest(G, seeds, node_gap=1.2, layer_gap=1.0, tree_gap=4.0):
    if not seeds:
        seeds = [n for n in G.nodes() if G.in_degree(n) == 0] or list(G.nodes())[:1]

    pos_all = {}
    x_offset = 0.0
    for s in seeds:
        pos_t, w = _layout_tidy_one_tree(G, s, node_gap=node_gap, layer_gap=layer_gap)
        for n, (x, y) in pos_t.items():
            pos_all[n] = np.array([x + x_offset, y], dtype=float)
        x_offset += w + tree_gap
    return pos_all


def plot_graph_layered_tidy(
    G,
    ns,
    seed_info=None,
    title="Recruitment Tree (Layered · Tidy)",
    node_gap=1.2,
    layer_gap=1.0,
    tree_gap=4.0,
    edge_alpha=0.6,
    show_edges=True,
    jitter=0.0,
):
    seeds = [n for n in G.nodes() if ns.get(n, {}).get("is_seed")]
    pos = layout_layered_tidy_forest(
        G, seeds, node_gap=node_gap, layer_gap=layer_gap, tree_gap=tree_gap
    )

    if jitter and jitter > 0:
        for n in pos:
            pos[n][0] += np.random.uniform(-jitter, jitter)

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(
            size=9,
            color=["#d62728" if ns[n]["is_seed"] else "#1f77b4" for n in G.nodes()],
            line=dict(width=1, color="white"),
        ),
        text=[fmt_id(n) for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=[
            "ID: "
            + fmt_id(n)
            + f"<br>is_seed: {ns[n]['is_seed']}"
            + f"<br>networksize_in_tree: {ns[n]['networksize_in_tree']}"
            + f"<br>reported_networksize: {ns[n]['reported_networksize']}"
            + (
                f"<br><b>Seed</b>: Wave={seed_info[n]['wave']} N={seed_info[n]['n']}"
                if seed_info and ns[n]["is_seed"] and n in seed_info
                else ""
            )
            for n in G.nodes()
        ],
    )

    traces = []
    if show_edges and G.number_of_edges() > 0:
        traces.append(
            go.Scatter(
                x=sum([[pos[u][0], pos[v][0], None] for u, v in G.edges()], []),
                y=sum([[pos[u][1], pos[v][1], None] for u, v in G.edges()], []),
                mode="lines",
                line=dict(width=1, color=f"rgba(120,120,120,{edge_alpha})"),
                hoverinfo="none",
            )
        )
    traces.append(node_trace)

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


# ==============================
# D) UI (English, Tidy-only)
# ==============================
st.title("REDCap RDS Tree Automata")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["REDCap API", "Upload file"], index=1)

    st.header("Layout")
    st.caption("Only Layered (Tidy wave) is available.")
    jitter_on = st.checkbox("Reduce overlap with small jitter", value=True)
    jitter_val = st.slider("Jitter amount (x-axis)", 0.0, 0.5, 0.15, 0.01)
    show_edges_tidy = st.checkbox("Show edges", value=True, key="show_edges_tidy")
    edge_alpha = st.slider("Edge opacity", 0.05, 1.0, 0.60, 0.05)

if "site_trees" not in st.session_state:
    st.session_state.site_trees = {}

default_in_field = "inpon_number"
default_seed_field = "seed_id"
default_out_fields = [f"outpons_{i}" for i in range(1, 6)]
default_networksize_field = "networksize"

# Source: REDCap
if source == "REDCap API":
    api_url = st.text_input(
        "REDCap API URL", value="https://mrprcbcw.hosts.jhmi.edu/redcap/api/"
    )
    api_token = st.text_input("API Token", value="", type="password")

    with st.expander("Field mapping (optional)"):
        in_field = st.text_input("Incoming coupon field", value=default_in_field)
        seed_field = st.text_input("Seed field", value=default_seed_field)
        out_fields_text = st.text_input(
            "Recruitment out fields (comma separated)", value=",".join(default_out_fields)
        )
        networksize_field = st.text_input(
            "Network size field", value=default_networksize_field
        )
        out_fields = [c.strip() for c in out_fields_text.split(",") if c.strip()]

    if st.button("Fetch from REDCap"):
        try:
            df = fetch_redcap(api_url, api_token)
            st.success(f"Fetched {len(df)} rows from REDCap")
            st.session_state.df = df
            st.session_state.mapping = {
                "in_field": in_field,
                "seed_field": seed_field,
                "out_fields": out_fields,
                "networksize_field": networksize_field,
            }
        except Exception as e:
            st.error(f"Failed to fetch from REDCap: {e}")

# Source: Upload
else:
    uploaded = st.file_uploader("Upload CSV/TSV/XLSX", type=["csv", "tsv", "xlsx"])
    if uploaded:
        try:
            df_up = read_uploaded_file(uploaded)
            st.write("Preview", df_up.head())

            schema = infer_schema(df_up)
            in_field = st.selectbox(
                "Incoming coupon field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["in_field"])
                    if schema["in_field"] in df_up.columns
                    else 0
                ),
            )
            seed_field = st.selectbox(
                "Seed field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["seed_field"])
                    if schema["seed_field"] in df_up.columns
                    else 0
                ),
            )
            networksize_field = st.selectbox(
                "Network size field",
                df_up.columns,
                index=(
                    df_up.columns.get_loc(schema["networksize_field"])
                    if schema["networksize_field"] in df_up.columns
                    else 0
                ),
            )
            out_fields = st.multiselect(
                "Recruitment out fields",
                list(df_up.columns),
                default=[c for c in schema["out_fields"] if c in df_up.columns],
            )

            if st.button("Use this uploaded file"):
                st.session_state.df = df_up
                st.session_state.mapping = {
                    "in_field": in_field,
                    "seed_field": seed_field,
                    "out_fields": out_fields,
                    "networksize_field": networksize_field,
                }
                st.success("Data and field mapping saved (used by Draw)")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

# Site options
add_site = st.checkbox("Add site-level recruitment")
if add_site:
    prefix_len = st.number_input(
        "Leading digits length for site code", min_value=1, max_value=10, value=3
    )
    site_prefix = st.text_input("Site prefix digits (e.g., 150):")
    site_name = st.text_input("Site name (e.g., Site A):")

# Draw
if st.button("Draw"):
    if "df" not in st.session_state or "mapping" not in st.session_state:
        st.error("Fetch from REDCap or upload a file and finish field mapping first.")
        st.stop()

    df = st.session_state.df
    in_field = st.session_state.mapping["in_field"]
    seed_field = st.session_state.mapping["seed_field"]
    out_fields = st.session_state.mapping["out_fields"]
    networksize_field = st.session_state.mapping["networksize_field"]

    needed = set([in_field, seed_field, networksize_field] + list(out_fields))
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Missing fields: {', '.join(missing)}")
        st.stop()

    try:
        G_all = build_graph(df, in_field, seed_field, out_fields)
        ns_all = compute_network_size(G_all, df, seed_field, in_field, networksize_field)
    except Exception as e:
        st.error(f"Graph construction or network size computation failed: {e}")
        st.stop()

    seeds = [n for n in G_all.nodes() if ns_all[n]["is_seed"]]
    seed_info_all = {}
    for node in seeds:
        wave_map = compute_wave(G_all, node)
        seed_info_all[node] = {
            "wave": (max(wave_map.values()) if len(wave_map) else 0),
            "n": (len(wave_map) if len(wave_map) else 1),
        }
    st.session_state.G_all = G_all
    st.session_state.ns_all = ns_all
    st.session_state.seed_info_all = seed_info_all

    # Build site-level subtrees
    if add_site and site_prefix and site_name:
        try:
            site_seeds = seeds_from_prefix(df, seed_field, site_prefix)
            nodes_site = set()
            for s in site_seeds:
                if s in G_all:
                    nodes_site |= set(nx.bfs_tree(G_all, s).nodes())

            if nodes_site:
                G_site = G_all.subgraph(nodes_site).copy()
                ns_site = compute_network_size(G_site, df, seed_field, in_field, networksize_field)
                for n in G_site.nodes():
                    ns_site[n]["is_seed"] = (n in site_seeds)

                seed_info_site = {}
                for node in site_seeds:
                    if node in G_site:
                        wave_map = compute_wave(G_site, node)
                        seed_info_site[node] = {
                            "wave": (max(wave_map.values()) if wave_map else 0),
                            "n": (len(wave_map) if wave_map else 1),
                        }

                st.session_state.site_trees[site_name] = {
                    "graph": G_site,
                    "networksize": ns_site,
                    "seed_info": seed_info_site,
                }
            else:
                st.warning(
                    f"No seeds starting with prefix '{site_prefix}' were found in the data."
                )
        except Exception as e:
            st.warning(f"Failed to build site subtree: {e}")

# Show FULL tree
if "G_all" in st.session_state:
    G_all = st.session_state.G_all
    ns_all = st.session_state.ns_all
    seed_info_all = st.session_state.seed_info_all

    seeds_all = sorted([n for n in G_all.nodes() if ns_all[n]["is_seed"]], key=lambda x: fmt_id(x))
    focus_seed = st.selectbox(
        "Focus on seed (optional)", ["<All seeds>"] + [fmt_id(s) for s in seeds_all], index=0
    )

    fmt_to_raw = {fmt_id(s): s for s in seeds_all}
    G_view = G_all
    ns_view = ns_all
    if focus_seed != "<All seeds>":
        raw_seed = fmt_to_raw[focus_seed]
        max_allowed = seed_info_all.get(raw_seed, {}).get("wave", 0)
        max_wave = st.slider("Show up to wave (inclusive)", 0, int(max_allowed), int(max_allowed))
        G_view = bfs_subgraph_upto_wave(G_all, raw_seed, max_wave)
        ns_view = {n: ns_all[n] for n in G_view.nodes()}

    fig = plot_graph_layered_tidy(
        G_view,
        ns_view,
        seed_info_all,
        title="Full Recruitment (Layered · Tidy)",
        node_gap=1.2,
        layer_gap=1.0,
        tree_gap=4.0,
        edge_alpha=edge_alpha,
        show_edges=show_edges_tidy,
        jitter=(jitter_val if jitter_on else 0.0),
    )
    st.plotly_chart(fig, use_container_width=True)
    to_downloadable_html(fig, "full_tree_tidy")

    # Cleaning / export (current view)
    under_ids = [
        n
        for n in G_view.nodes()
        if ns_view[n]["networksize_in_tree"] > ns_view[n]["reported_networksize"]
    ]
    st.write(f"Underreported count: **{len(under_ids)}**")
    if under_ids:
        st.write("Underreported coupon IDs:", ", ".join(fmt_id(x) for x in under_ids))
    fix_under = st.checkbox("Fix underreported networksize", key="fix_under")
    if fix_under and not under_ids:
        st.warning("No records to fix underreported.")
        fix_under = False

    reported_s = pd.Series(
        [ns_view[n]["reported_networksize"] for n in G_view.nodes()],
        index=list(G_view.nodes()),
    )
    in_s = pd.Series(
        [ns_view[n]["networksize_in_tree"] for n in G_view.nodes()],
        index=list(G_view.nodes()),
    )
    fixed = in_s.where(in_s > reported_s, reported_s)

    vals = [ns_view[n]["reported_networksize"] for n in ns_view]
    pct = pd.Series(vals).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("reported_networksize")
    pct.index = [f"{int(q * 100)}%" for q in pct.index]
    st.write("Percentiles of reported_networksize")
    st.dataframe(pct)

    show_vals = fixed if fix_under else reported_s
    median_val = float(pd.Series(show_vals).replace(0, np.nan).dropna().median())
    impute_na0 = st.checkbox("Impute NA and 0", key="impute_na0")
    impute_val = st.number_input(
        "Imputation value for NA/0", min_value=1.0, step=1.0, value=median_val, format="%.2f", key="impute_val"
    )
    st.caption(f"Median networksize (current): {median_val:.2f}")

    cap = st.number_input("Set networksize cap:", min_value=1, step=1, format="%d", key="cap")
    apply_cap = st.checkbox("Apply cap to reported_networksize", key="apply_cap")

    capped = reported_s.where(reported_s <= cap, cap)
    fixed_then = fixed.where(fixed <= cap, cap)

    if fix_under and apply_cap and impute_na0:
        clean_tmp = fixed_then.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif fix_under and impute_na0:
        clean_tmp = fixed.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif apply_cap and impute_na0:
        clean_tmp = capped.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif impute_na0:
        clean_tmp = reported_s.copy().where(lambda s: (s.notna()) & (s != 0), impute_val)
        networksize_clean = clean_tmp
    elif fix_under and apply_cap:
        networksize_clean = fixed_then
    elif fix_under:
        networksize_clean = fixed
    elif apply_cap:
        networksize_clean = capped
    else:
        networksize_clean = reported_s

    networksize_clean = networksize_clean.clip(lower=1.0).fillna(1.0)

    df_clean = pd.DataFrame(
        {"id": [fmt_id(x) for x in networksize_clean.index], "networksize_clean": networksize_clean.values}
    )
    st.download_button(
        "Export cleaned networksize (current view)",
        data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="networksize_clean_current_view.csv",
        mime="text/csv",
    )

# ==============================
# E) Site-level trees (Tidy-only, with weights)
# ==============================
if st.session_state.site_trees:
    st.subheader("Site-Level Trees")
    to_del = []

    for site, info in st.session_state.site_trees.items():
        with st.expander(site):
            G_site = info["graph"]
            ns_site = info["networksize"]
            seed_info_site = info["seed_info"]

            seeds_site_all = sorted(
                [n for n in G_site.nodes() if ns_site[n]["is_seed"]], key=lambda x: fmt_id(x)
            )
            focus_seed_site = st.selectbox(
                f"[{site}] Focus on seed",
                ["<All seeds>"] + [fmt_id(s) for s in seeds_site_all],
                index=0,
                key=f"seed_{site}",
            )

            fmt_to_raw_site = {fmt_id(s): s for s in seeds_site_all}
            Gs, nss = G_site, ns_site
            seeds_for_h = seeds_site_all

            if focus_seed_site != "<All seeds>":
                raw_seed = fmt_to_raw_site[focus_seed_site]
                max_allowed = max_wave_of_graph(G_site, [raw_seed])
                max_wave_site = st.slider(
                    f"[{site}] Show up to wave (inclusive)",
                    0,
                    int(max_allowed),
                    int(max_allowed),
                    key=f"wave_{site}",
                )
                Gs = bfs_subgraph_upto_wave(G_site, raw_seed, max_wave_site)
                nss = {n: ns_site[n] for n in Gs.nodes()}
                seeds_for_h = [raw_seed]

            fig_site = plot_graph_layered_tidy(
                Gs,
                nss,
                seed_info_site,
                title=f"Tree: {site} (Layered · Tidy)",
                node_gap=1.2,
                layer_gap=1.0,
                tree_gap=4.0,
                edge_alpha=edge_alpha,
                show_edges=show_edges_tidy,
                jitter=(jitter_val if jitter_on else 0.0),
            )
            st.plotly_chart(fig_site, use_container_width=True)
            to_downloadable_html(fig_site, f"{site}_tree_tidy")

            st.markdown(f"- Participants: {Gs.number_of_nodes()}")
            st.markdown(f"- Seeds: {sum(nss[n]['is_seed'] for n in Gs.nodes())}")
            st.markdown(f"- Max wave: {max_wave_of_graph(Gs, seeds_for_h)}")

            # Cleaning (site)
            under_ids_site = [
                n
                for n in Gs.nodes()
                if nss[n]["networksize_in_tree"] > nss[n]["reported_networksize"]
            ]
            st.write(f"Underreported count: **{len(under_ids_site)}**")
            if under_ids_site:
                st.write("Underreported coupon IDs:", ", ".join(fmt_id(x) for x in under_ids_site))

            fix_under_site = st.checkbox(f"[{site}] Fix underreported networksize", key=f"fix_under_{site}")
            if fix_under_site and not under_ids_site:
                st.warning("No records to fix underreported.")
                fix_under_site = False

            reported_s_site = pd.Series(
                [nss[n]["reported_networksize"] for n in Gs.nodes()],
                index=list(Gs.nodes()),
            )
            in_s_site = pd.Series(
                [nss[n]["networksize_in_tree"] for n in Gs.nodes()],
                index=list(Gs.nodes()),
            )
            fixed_site = in_s_site.where(in_s_site > reported_s_site, reported_s_site)

            vals_site = [nss[n]["reported_networksize"] for n in nss]
            pct_site = pd.Series(vals_site).quantile([0, 0.25, 0.5, 0.75, 1.0]).to_frame("reported_networksize")
            pct_site.index = [f"{int(q * 100)}%" for q in pct_site.index]
            st.write("Percentiles of reported_networksize")
            st.dataframe(pct_site)

            show_vals_site = fixed_site if fix_under_site else reported_s_site
            median_site_val = float(pd.Series(show_vals_site).replace(0, np.nan).dropna().median())
            impute_na0_site = st.checkbox(f"[{site}] Impute NA and 0", key=f"impute_{site}")
            impute_val_site = st.number_input(
                f"[{site}] Imputation value for NA/0",
                min_value=1.0,
                step=1.0,
                value=median_site_val,
                format="%.2f",
                key=f"impute_val_{site}",
            )
            st.caption(f"[{site}] Median networksize (current): {median_site_val:.2f}")

            cap_site = st.number_input(f"[{site}] Set networksize cap:", min_value=1, step=1, format="%d", key=f"cap_{site}")
            apply_cap_site = st.checkbox(f"[{site}] Apply cap to reported_networksize", key=f"apply_cap_{site}")

            capped_site = reported_s_site.where(reported_s_site <= cap_site, cap_site)
            fixed_then_site = fixed_site.where(fixed_site <= cap_site, cap_site)

            if fix_under_site and apply_cap_site and impute_na0_site:
                clean_tmp = fixed_then_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif fix_under_site and impute_na0_site:
                clean_tmp = fixed_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif apply_cap_site and impute_na0_site:
                clean_tmp = capped_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif impute_na0_site:
                clean_tmp = reported_s_site.copy().where(lambda s: (s.notna()) & (s != 0), impute_val_site)
                networksize_clean_site = clean_tmp
            elif fix_under_site and apply_cap_site:
                networksize_clean_site = fixed_then_site
            elif fix_under_site:
                networksize_clean_site = fixed_site
            elif apply_cap_site:
                networksize_clean_site = capped_site
            else:
                networksize_clean_site = reported_s_site

            networksize_clean_site = networksize_clean_site.clip(lower=1.0).fillna(1.0)

            df_clean_site = pd.DataFrame(
                {"id": [fmt_id(x) for x in networksize_clean_site.index], "networksize_clean": networksize_clean_site.values}
            )
            st.download_button(
                f"[{site}] Export cleaned networksize",
                data=df_clean_site.to_csv(index=False).encode("utf-8"),
                file_name=f"{site}_networksize_clean.csv",
                mime="text/csv",
            )

            # Weights (site-level only)
            N_est_site = st.number_input(f"[{site}] Population estimate:", min_value=1, step=1, key=f"N_{site}")
            if st.button(f"[{site}] Run Gile's Weights", key=f"run_weight_{site}"):
                try:
                    id_list = list(Gs.nodes())
                    recruiter_lst = [
                        list(Gs.predecessors(n))[0] if list(Gs.predecessors(n)) else "0"
                        for n in id_list
                    ]
                    ns_int = [int(round(float(x))) for x in networksize_clean_site.reindex(id_list).tolist()]

                    df_w = pd.DataFrame({"id": id_list, "recruiter": recruiter_lst, "network.size": ns_int})
                    df_w["network.size"] = df_w["network.size"].astype(int)

                    st.write("network.size to R:", df_w["network.size"].tolist())

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as in_csv, \
                         tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as out_csv:
                        df_w.to_csv(in_csv.name, index=False)
                        cmd = [
                            "Rscript",
                            os.path.join(os.path.dirname(__file__), "compute_rds_weights.R"),
                            in_csv.name,
                            out_csv.name,
                            str(int(N_est_site)),
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True)

                        if res.returncode != 0:
                            st.error("R script failed")
                            st.text(res.stderr)
                        else:
                            out_df = pd.read_csv(out_csv.name)
                            st.success("Weights computed")
                            st.dataframe(out_df)
                            st.download_button(
                                f"[{site}] Download weights",
                                data=out_df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{site}_weights.csv",
                                mime="text/csv",
                            )
                except FileNotFoundError as fe:
                    st.error(f"compute_rds_weights.R not found or file error: {fe}")
                except Exception as e:
                    st.error(f"Failed to compute weights via R: {e}")

            if st.button(f"Delete {site}", key=f"del_{site}"):
                to_del.append(site)

    for s in to_del:
        del st.session_state.site_trees[s]




        


#streamlit run /Users/miyuanqi/Desktop/redcap_rds_app/app4.py --server.port 8504