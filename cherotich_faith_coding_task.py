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
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    ren = {
        "CAGE": "CAGE NUMBER",
        "CAGE_NO": "CAGE NUMBER",
        "CAGE ID": "CAGE NUMBER",
        "TOTAL WEIGHT (KG)": "TOTAL WEIGHT [KG]",
        "TOTAL WEIGHT  [KG]": "TOTAL WEIGHT [KG]",
        "ABW(G)": "ABW(G)",
        "ABW [G]": "ABW(G)",
        "AVERAGE BODYWEIGHT (G)": "AVERAGE BODY WEIGHT(G)",
        "ORIGIN": "ORIGIN CAGE",
        "DEST": "DESTINATION CAGE",
        "DESTINATION": "DESTINATION CAGE",
    }
    df.rename(columns={k: v for k, v in ren.items() if k in df.columns}, inplace=True)
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    def _coerce(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates: list[str], fuzzy_hint: str | None = None) -> str | None:
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

    # --- Coerce cage labels to integers (C2 → 2) ---
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    if transfers is not None:
        # common origin/destination headers → integers
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # normalize transfer weight column
        wcol = find_col(transfers,
                        ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"],
                        fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # --- Parse dates robustly ---
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


# =====================
# Preprocess Cage 2
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    feeding_c2  = feeding[feeding["CAGE NUMBER"] == cage_number].copy()
    harvest_c2  = harvest[harvest["CAGE NUMBER"] == cage_number].copy() if "CAGE NUMBER" in harvest.columns else harvest.copy()
    sampling_c2 = sampling[sampling["CAGE NUMBER"] == cage_number].copy()

    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame([{
        "DATE": stocking_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw
    }])

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE")

    # Base timeline
    base = sampling_c2.sort_values("DATE").copy()
    base["STOCKED"] = stocked_fish
    # Initialise cumulative trackers to zero to avoid KeyErrors downstream
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ---- Harvest cumulatives (fish & kg) aligned to sampling rows
    fish_col_h = find_col(harvest_c2, ["NUMBER OF FISH","NUMBER OF FISH "], fuzzy_hint="FISH")
    wcol_h     = find_col(harvest_c2, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], fuzzy_hint="WEIGHT")
    if fish_col_h or wcol_h:
        h = harvest_c2.copy()
        if fish_col_h: h["H_FISH"] = pd.to_numeric(h[fish_col_h], errors="coerce").fillna(0)
        else:          h["H_FISH"] = 0
        if wcol_h:     h["H_KG"]   = pd.to_numeric(h[wcol_h],   errors="coerce").fillna(0)
        else:          h["H_KG"]   = 0
        h = h.sort_values("DATE")
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ---- Transfers cumulatives (fish & kg), excluding initial stocking instant
    if transfers is not None and not transfers.empty:
        t = transfers.copy()
        t = t[t["DATE"] > stocking_date]  # exclude stocking moment
        ncol = find_col(t, ["NUMBER OF FISH","N_FISH"], fuzzy_hint="FISH")
        wcol = find_col(t, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], fuzzy_hint="WEIGHT")
        if ncol is None: t["T_FISH"] = 0
        else:            t["T_FISH"] = pd.to_numeric(t[ncol], errors="coerce").fillna(0)
        if wcol is None: t["T_KG"] = 0
        else:            t["T_KG"] = pd.to_numeric(t[wcol], errors="coerce").fillna(0)

        # Outgoing = origin cage 2
        if "ORIGIN CAGE" in t.columns:
            tout = t[t["ORIGIN CAGE"] == cage_number].sort_values("DATE").copy()
            if not tout.empty:
                tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
                mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
                base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
                base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming = destination cage 2
        if "DESTINATION CAGE" in t.columns:
            tin = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE").copy()
            if not tin.empty:
                tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
                mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
                base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
                base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish (used later to compute standing biomass)
    base["FISH_ALIVE"]      = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"]  = base["FISH_ALIVE"].astype(int)
    return feeding_c2, harvest_c2, base

# =====================
# Compute summary
# =====================

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2  = feeding_c2.copy()
    s           = sampling_c2.copy().sort_values("DATE")

    # Resolve columns
    feed_col = find_col(feeding_c2,
        ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED AMOUNT [KG]","FEED (KG)","FEED KG","FEED_AMOUNT","FEED"],
        fuzzy_hint="FEED")
    abw_col = find_col(s, ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW(G)","ABW [G]","ABW"],
        fuzzy_hint="ABW")
    if not feed_col or not abw_col:
        return s  # fail-safe

    # Cumulative feed (daily) → align to sampling
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")

    # Standing biomass from ABW×fish
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (
        pd.to_numeric(summary["FISH_ALIVE"], errors="coerce").fillna(0) * summary["ABW_G"].fillna(0) / 1000.0
    )

    # Period deltas (assign to current row)
    summary["FEED_PERIOD_KG"]   = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]      = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Transfers/harvest period deltas
    for cum_col, per_col in [
        ("IN_KG_CUM",  "TRANSFER_IN_KG"),
        ("OUT_KG_CUM", "TRANSFER_OUT_KG"),
        ("HARV_KG_CUM","HARVEST_KG"),
    ]:
        if cum_col in summary.columns:
            summary[per_col] = summary[cum_col].diff()
        else:
            summary[per_col] = np.nan

    # Growth produced in the period (exclude logistics effects)
    summary["GROWTH_KG"] = (
        summary["ΔBIOMASS_STANDING"]
        + summary["HARVEST_KG"].fillna(0)
        + summary["TRANSFER_OUT_KG"].fillna(0)
        - summary["TRANSFER_IN_KG"].fillna(0)
    )

    # Period eFCR and Aggregated eFCR
    eps = 1e-9
    summary["PERIOD_eFCR"] = summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"].where(lambda x: x.abs()>eps)
    summary["GROWTH_CUM_KG"] = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["AGGREGATED_eFCR"] = summary["FEED_AGG_KG"] / summary["GROWTH_CUM_KG"].where(lambda x: x.abs()>eps)

    # First row (stocking) should not show period metrics
    first_idx = summary.index.min()
    summary.loc[first_idx, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","TRANSFER_IN_KG","TRANSFER_OUT_KG",
                            "HARVEST_KG","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    # Final tidy columns for the table
    cols = [
        "DATE","CAGE NUMBER","NUMBER OF FISH","ABW_G",
        "BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    return summary[[c for c in cols if c in summary.columns]]


# =====================
# UI
# =====================

st.title("Fish Cage Production Analysis (with Transfers)")
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader(f"Cage 2 – Production Summary")
    show_cols = ["DATE","NUMBER OF FISH","ABW_G","TOTAL_WEIGHT_KG","AGGREGATED_eFCR","PERIOD_eFCR"]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]])

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR"]) 

    if selected_kpi == "Biomass" and "TOTAL_WEIGHT_KG" in summary_c2.columns:
        fig = px.line(summary_c2.dropna(subset=["TOTAL_WEIGHT_KG"]), x="DATE", y="TOTAL_WEIGHT_KG", markers=True,
                      title="Cage 2: Biomass Over Time", labels={"TOTAL_WEIGHT_KG": "Total Biomass (kg)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "ABW" and "ABW_G" in summary_c2.columns:
        fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title="Cage 2: Average Body Weight Over Time", labels={"ABW_G": "ABW (g)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "eFCR" and {"AGGREGATED_eFCR","PERIOD_eFCR"}.issubset(summary_c2.columns):
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                      title="Cage 2: eFCR Over Time", labels={"AGGREGATED_eFCR": "Aggregated eFCR"})
        fig.update_traces(showlegend=True, name="Aggregated eFCR")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", showlegend=True, line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
