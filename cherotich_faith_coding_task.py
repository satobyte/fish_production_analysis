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
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def to_number(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# =====================
# Preprocess Cage 2 (stocking, harvest, transfers)
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    feeding_c2 = feeding[feeding["CAGE NUMBER"] == cage_number].copy()
    harvest_c2 = harvest[harvest["CAGE NUMBER"] == cage_number].copy()
    sampling_c2 = sampling[sampling["CAGE NUMBER"] == cage_number].copy()

    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame([{ "DATE": stocking_date, "CAGE NUMBER": cage_number, "AVERAGE BODY WEIGHT(G)": initial_abw }])

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE")

    base = sampling_c2.sort_values("DATE").copy()
    base["STOCKED"] = stocked_fish

    # Initialize columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # TODO: Add harvest + transfers merge here if available (similar to previous version)

    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Compute summary (period metrics assigned to row after period)
# =====================

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    summary = sampling_c2.copy().sort_values("DATE")

    if "FEED AMOUNT (Kg)" not in feeding_c2.columns or "AVERAGE BODY WEIGHT(G)" not in summary.columns:
        return summary

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2["FEED AMOUNT (Kg)"], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(summary, feeding_c2[["DATE","CUM_FEED"]].sort_values("DATE"), on="DATE", direction="backward")

    summary["ABW_G"] = pd.to_numeric(summary["AVERAGE BODY WEIGHT(G)"].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (summary["FISH_ALIVE"] * summary["ABW_G"]) / 1000.0

    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"] = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Period production growth (include harvest/transfers if added)
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"]

    summary["PERIOD_eFCR"] = summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"].replace(0, np.nan)
    summary["AGGREGATED_eFCR"] = summary["CUM_FEED"] / summary["GROWTH_KG"].cumsum(skipna=True)

    # Blank out first row period metrics
    first_idx = summary.index.min()
    summary.loc[first_idx,["FEED_PERIOD_KG","ΔBIOMASS_STANDING","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    return summary

# =====================
# UI
# =====================

st.title("Fish Cage Production Analysis (with Transfers)")
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary")
    show_cols = ["DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG","FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG","AGGREGATED_eFCR","PERIOD_eFCR"]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]])

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi == "Biomass":
        fig = px.line(summary_c2.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "ABW":
        fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True, labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.update_traces(showlegend=True, name="Aggregated eFCR")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", showlegend=True, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
