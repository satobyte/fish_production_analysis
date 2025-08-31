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
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = None
    if transfer_file is not None:
        transfers = normalize_columns(pd.read_excel(transfer_file))

    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# =====================
# Preprocess Cage 2
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    feeding_c2 = feeding[feeding["CAGE NUMBER"] == cage_number].copy()
    harvest_c2 = harvest[harvest["CAGE NUMBER"] == cage_number].copy()
    sampling_c2 = sampling[sampling["CAGE NUMBER"] == cage_number].copy()

    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame([
        {"DATE": stocking_date, "CAGE NUMBER": cage_number, "AVERAGE BODY WEIGHT(G)": initial_abw}
    ])

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE")

    base = sampling_c2.sort_values("DATE").copy()
    base["STOCKED"] = stocked_fish

    for col in ["HARV_CUM_FISH", "IN_FISH_CUM", "OUT_FISH_CUM", "IN_KG_CUM", "OUT_KG_CUM"]:
        base[col] = 0.0

    # Exclude first stocking event from transfers to avoid double counting
    if transfers is not None and not transfers.empty:
        transfers = transfers[transfers["DATE"] > stocking_date]

    # Standing fish
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_CUM_FISH"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Compute summary
# =====================

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()

    feed_col = find_col(feeding_c2,["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED"], fuzzy_hint="FEED")
    abw_col = find_col(sampling_c2,["AVERAGE BODY WEIGHT(G)","ABW(G)","ABW"], fuzzy_hint="ABW")
    if not feed_col or not abw_col:
        return sampling_c2

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[abw_col].map(to_number), errors="coerce")
    sampling_c2["TOTAL_WEIGHT_KG"] = (pd.to_numeric(sampling_c2["FISH_ALIVE"], errors="coerce").fillna(0) * sampling_c2["ABW_G"].fillna(0) / 1000.0)

    summary = pd.merge_asof(
        sampling_c2.sort_values("DATE"),
        feeding_c2.sort_values("DATE")["DATE"].to_frame().assign(CUM_FEED=feeding_c2["CUM_FEED"].values),
        on="DATE",
        direction="backward",
    )

    tw = summary["TOTAL_WEIGHT_KG"].replace(0, np.nan)
    summary["AGGREGATED_eFCR"] = summary["CUM_FEED"] / tw
    summary["PERIOD_WEIGHT_GAIN"] = summary["TOTAL_WEIGHT_KG"].diff().fillna(summary["TOTAL_WEIGHT_KG"])
    summary["PERIOD_FEED"] = summary["CUM_FEED"].diff().fillna(summary["CUM_FEED"])
    summary["PERIOD_eFCR"] = summary["PERIOD_FEED"] / summary["PERIOD_WEIGHT_GAIN"].replace(0, np.nan)

    return summary

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
