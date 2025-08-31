# app.py — Fish Cage Production Analysis (Biomass, ABW, eFCR) + Transfers + Period Metrics
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# =====================
# Helpers
# =====================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase headers, trim, collapse spaces; normalize a few common variants if needed."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Coerce cage labels like 'C2'/'Cage 2'/2 → 2."""
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates: list[str], fuzzy_hint: str | None = None) -> str | None:
    """Find a column by preferred names; fallback to substring match on fuzzy_hint."""
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

def to_number(x):
    """Extract numeric from strings like '1,234.5 g' → 1234.5."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# =====================
# Load data
# =====================

def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file is not None else None

    # coerce cage columns
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    if transfers is not None:
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # standardize transfer weight col if needed
        wcol = find_col(
            transfers,
            ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"],
            fuzzy_hint="WEIGHT",
        )
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# =====================
# Preprocess Cage 2 (stocking + harvest + transfers)
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Build the per-sampling timeline for Cage 2, with cumulative harvest/transfer series
    aligned to sampling dates, and standing fish counts. The first inbound transfer
    (stocking event) is excluded.
    """
    cage_number = 2

    # Filter to cage 2
    feeding_c2  = feeding[feeding.get("CAGE NUMBER").astype("Int64") == cage_number].copy() if "CAGE NUMBER" in feeding.columns else feeding.copy()
    harvest_c2  = harvest[harvest.get("CAGE NUMBER").astype("Int64") == cage_number].copy() if "CAGE NUMBER" in harvest.columns else harvest.copy()
    sampling_c2 = sampling[sampling.get("CAGE NUMBER").astype("Int64") == cage_number].copy() if "CAGE NUMBER" in sampling.columns else sampling.copy()

    # Stocking event
    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw_g = 7902, 0.7
    stocking_row = pd.DataFrame([{
        "DATE": stocking_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g
    }])

    # Build base timeline (one row per sampling, incl. stocking)
    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    base = sampling_c2.copy()
    base["STOCKED"] = stocked_fish

    # Initialise cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ---------- Harvest cumulatives (fish & kg) ----------
    # After normalize_columns, typical names are:
    #   NUMBER OF FISH, TOTAL WEIGHT [KG]  (brackets uppercased)
    h_fish_col = "NUMBER OF FISH" if "NUMBER OF FISH" in harvest_c2.columns else None
    h_kg_col   = "TOTAL WEIGHT [KG]" if "TOTAL WEIGHT [KG]" in harvest_c2.columns else None

    if (h_fish_col or h_kg_col) and not harvest_c2.empty:
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col],   errors="coerce").fillna(0) if h_kg_col   else 0
        h = h.sort_values("DATE")
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        mh = pd.merge_asof(
            base[["DATE"]].sort_values("DATE"),
            h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]],
            on="DATE", direction="backward"
        )
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ---------- Transfers cumulatives (exclude stocking instant) ----------
    if transfers is not None and not transfers.empty:
        t = transfers.copy()

        # Exclude the stocking event (any transfers at or before stocking timestamp)
        t = t[t["DATE"] > stocking_date]

        # Expected columns post-normalize:
        #   ORIGIN CAGE, DESTINATION CAGE, NUMBER OF FISH, TOTAL WEIGHT [KG]
        t_fish_col = "NUMBER OF FISH"     if "NUMBER OF FISH" in t.columns     else None
        t_kg_col   = "TOTAL WEIGHT [KG]"  if "TOTAL WEIGHT [KG]" in t.columns  else None

        # Outgoing: origin = cage 2
        if "ORIGIN CAGE" in t.columns:
            tout = t[t["ORIGIN CAGE"].apply(to_int_cage) == cage_number].sort_values("DATE")
            if not tout.empty:
                tout["OUT_FISH_CUM"] = pd.to_numeric(tout[t_fish_col], errors="coerce").fillna(0).cumsum() if t_fish_col else 0
                tout["OUT_KG_CUM"]   = pd.to_numeric(tout[t_kg_col],   errors="coerce").fillna(0).cumsum() if t_kg_col   else 0
                mo = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]],
                    on="DATE", direction="backward"
                )
                base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
                base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming: destination = cage 2
        if "DESTINATION CAGE" in t.columns:
            tin = t[t["DESTINATION CAGE"].apply(to_int_cage) == cage_number].sort_values("DATE")
            if not tin.empty:
                tin["IN_FISH_CUM"] = pd.to_numeric(tin[t_fish_col], errors="coerce").fillna(0).cumsum() if t_fish_col else 0
                tin["IN_KG_CUM"]   = pd.to_numeric(tin[t_kg_col],   errors="coerce").fillna(0).cumsum() if t_kg_col   else 0
                mi = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]],
                    on="DATE", direction="backward"
                )
                base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
                base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # ---------- Standing fish for each sampling date ----------
    base["FISH_ALIVE"] = (
        base["STOCKED"]
        - base["HARV_FISH_CUM"]
        + base["IN_FISH_CUM"]
        - base["OUT_FISH_CUM"]
    ).clip(lower=0)

    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Compute summary (period metrics + transfer-aware growth)
# =====================

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2  = feeding_c2.copy()
    s           = sampling_c2.copy().sort_values("DATE")

    # Resolve needed columns
    feed_col = "FEED AMOUNT (Kg)" if "FEED AMOUNT (Kg)" in feeding_c2.columns else None
    abw_col  = "AVERAGE BODY WEIGHT(G)" if "AVERAGE BODY WEIGHT(G)" in s.columns else None
    if not feed_col or not abw_col:
        return s  # fail-safe

    # Cum feed → align to sampling rows
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")

    # Standing biomass from ABW × fish
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (pd.to_numeric(summary["FISH_ALIVE"], errors="coerce").fillna(0)
                             * summary["ABW_G"].fillna(0) / 1000.0)

    # Period feed & Δ standing biomass (assigned to current row t_i)
    summary["FEED_PERIOD_KG"]     = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]        = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"]  = summary["BIOMASS_KG"].diff()

    # Period logistics (kg): diffs of the cumulative columns if present
    for cum_col, per_col in [("IN_KG_CUM","TRANSFER_IN_KG"),
                             ("OUT_KG_CUM","TRANSFER_OUT_KG"),
                             ("HARV_KG_CUM","HARVEST_KG")]:
        if cum_col in summary.columns:
            summary[per_col] = summary[cum_col].diff()
        else:
            summary[per_col] = np.nan

    # Produced growth in the period
    summary["GROWTH_KG"] = (summary["ΔBIOMASS_STANDING"]
                            + summary["HARVEST_KG"].fillna(0)
                            + summary["TRANSFER_OUT_KG"].fillna(0)
                            - summary["TRANSFER_IN_KG"].fillna(0))

    # Period eFCR and Aggregated eFCR with NA when not computable
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"] > 0,
                                      summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"],
                                      np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0,
                                          summary["FEED_AGG_KG"] / growth_cum,
                                          np.nan)

    # First row (stocking) → NA for period metrics
    first_idx = summary.index.min()
    summary.loc[first_idx, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING",
                            "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
                            "GROWTH_KG","PERIOD_eFCR"]] = np.nan
    return summary

# =====================
# UI
# =====================

st.title("Fish Cage Production Analysis (with Transfers)")
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file  = st.sidebar.file_uploader("Feeding Record",    type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest",      type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling",     type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary (period-based)")
    show_cols = [
        "DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]])

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])

    if selected_kpi == "Biomass":
        fig = px.line(
            summary_c2.dropna(subset=["BIOMASS_KG"]),
            x="DATE", y="BIOMASS_KG", markers=True,
            title="Cage 2: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_kpi == "ABW":
        fig = px.line(
            summary_c2.dropna(subset=["ABW_G"]),
            x="DATE", y="ABW_G", markers=True,
            title="Cage 2: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # eFCR
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(
            dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
            title="Cage 2: eFCR Over Time", labels={"AGGREGATED_eFCR":"Aggregated eFCR"}
        )
        fig.update_traces(showlegend=True, name="Aggregated eFCR")
        fig.add_scatter(
            x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers",
            name="Period eFCR", showlegend=True, line=dict(dash="dash")
        )
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
