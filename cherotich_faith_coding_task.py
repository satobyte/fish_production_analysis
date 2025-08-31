# This is a Python file (Streamlit app). Save it as app.py in your repo.
# The canvas type is set to code/react only for preview; the code itself is Python.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# ---------- helpers ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    ren = {
        "CAGE": "CAGE NUMBER",
        "CAGE_NO": "CAGE NUMBER",
        "CAGE ID": "CAGE NUMBER",
        "TOTAL WEIGHT (KG)": "TOTAL WEIGHT [kg]",
        "ABW(G)": "ABW(G)",
        "ABW [G]": "ABW(G)",
        "AVERAGE BODYWEIGHT (G)": "AVERAGE BODY WEIGHT(G)",
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


# ---------- 1) load data ----------

def load_data(feeding_file, harvest_file, sampling_file):
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))

    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    for df in (feeding, harvest, sampling):
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling


# ---------- 2) preprocess cage 2 ----------

def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2
    feeding_c2 = feeding[feeding["CAGE NUMBER"] == cage_number].copy()
    harvest_c2 = harvest[harvest["CAGE NUMBER"] == cage_number].copy()
    sampling_c2 = sampling[sampling["CAGE NUMBER"] == cage_number].copy()

    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame(
        [
            {
                "DATE": stocking_date,
                "CAGE NUMBER": cage_number,
                "AVERAGE BODY WEIGHT(G)": initial_abw,
            }
        ]
    )

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE")

    start_date, end_date = stocking_date, pd.to_datetime("2025-06-30")
    sampling_c2 = sampling_c2[(sampling_c2["DATE"] >= start_date) & (sampling_c2["DATE"] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2["DATE"] >= start_date) & (feeding_c2["DATE"] <= end_date)]
    harvest_c2 = harvest_c2[(harvest_c2["DATE"] >= start_date) & (harvest_c2["DATE"] <= end_date)]

    # standing fish = stocked – cumulative harvested
    if "NUMBER OF FISH" in harvest_c2.columns or "NUMBER OF FISH " in harvest_c2.columns:
        fish_col = find_col(harvest_c2, ["NUMBER OF FISH", "NUMBER OF FISH "], fuzzy_hint="FISH")
        h = harvest_c2[["DATE", fish_col]].dropna().copy()
        h["HARV_CUM"] = h[fish_col].cumsum()
    else:
        h = pd.DataFrame({"DATE": [], "HARV_CUM": []})

    sampling_c2["STOCKED"] = stocked_fish
    if not h.empty:
        sampling_c2 = pd.merge_asof(
            sampling_c2.sort_values("DATE"),
            h[["DATE", "HARV_CUM"]].sort_values("DATE"),
            on="DATE",
            direction="backward",
        )
        sampling_c2["HARV_CUM"] = sampling_c2["HARV_CUM"].fillna(0).astype(int)
    else:
        sampling_c2["HARV_CUM"] = 0

    sampling_c2["FISH_ALIVE"] = (sampling_c2["STOCKED"] - sampling_c2["HARV_CUM"]).clip(lower=0)
    sampling_c2["NUMBER OF FISH"] = sampling_c2["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, sampling_c2


# ---------- 3) compute summary ----------

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()

    feed_col = find_col(
        feeding_c2,
        [
            "FEED AMOUNT (KG)",
            "FEED AMOUNT (Kg)",
            "FEED AMOUNT [KG]",
            "FEED (KG)",
            "FEED KG",
            "FEED_AMOUNT",
            "FEED",
        ],
        fuzzy_hint="FEED",
    )
    if not feed_col:
        st.error(f"Could not find feed column. Available: {list(feeding_c2.columns)}")
        st.stop()

    abw_col = find_col(
        sampling_c2,
        ["AVERAGE BODY WEIGHT(G)", "AVERAGE BODY WEIGHT (G)", "ABW(G)", "ABW [G]", "ABW"],
        fuzzy_hint="ABW",
    )
    if not abw_col:
        st.error(f"Could not find ABW column. Available: {list(sampling_c2.columns)}")
        st.stop()

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # ABW and biomass
    sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[abw_col].map(to_number), errors="coerce")
    sampling_c2["TOTAL_WEIGHT_KG"] = (
        pd.to_numeric(sampling_c2["FISH_ALIVE"], errors="coerce").fillna(0)
        * sampling_c2["ABW_G"].fillna(0)
        / 1000.0
    )

    # align cum feed to sampling
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

    keep = [
        "DATE",
        "CAGE NUMBER",
        "NUMBER OF FISH",
        "ABW_G",
        "TOTAL_WEIGHT_KG",
        "CUM_FEED",
        "AGGREGATED_eFCR",
        "PERIOD_eFCR",
    ]
    return summary[[c for c in keep if c in summary.columns]]


# ---------- 4) mock cages ----------

def create_mock_cage_data(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock["CAGE NUMBER"] = cage_id
        rng = np.random.default_rng(cage_id)
        mock["TOTAL_WEIGHT_KG"] = mock["TOTAL_WEIGHT_KG"] * rng.normal(1, 0.05, size=len(mock))
        mock["NUMBER OF FISH"] = (
            mock["NUMBER OF FISH"].astype(int) + rng.integers(-50, 51, size=len(mock))
        ).clip(lower=1)
        mock["CUM_FEED"] = mock["CUM_FEED"] * rng.normal(1, 0.10, size=len(mock))
        tw = mock["TOTAL_WEIGHT_KG"].replace(0, np.nan)
        mock["AGGREGATED_eFCR"] = mock["CUM_FEED"] / tw
        mock["PERIOD_WEIGHT_GAIN"] = mock["TOTAL_WEIGHT_KG"].diff().fillna(mock["TOTAL_WEIGHT_KG"])
        mock["PERIOD_FEED"] = mock["CUM_FEED"].diff().fillna(mock["CUM_FEED"])
        mock["PERIOD_eFCR"] = mock["PERIOD_FEED"] / mock["PERIOD_WEIGHT_GAIN"].replace(0, np.nan)
        mock_summaries[cage_id] = mock
    return mock_summaries


# ---------- 5) UI ----------

st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"]) 
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"]) 
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"]) 

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling = load_data(feeding_file, harvest_file, sampling_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR"])  # renamed + ABW added

    df = all_cages[selected_cage].copy()

    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[[
        "DATE",
        "NUMBER OF FISH",
        "ABW_G",
        "TOTAL_WEIGHT_KG",
        "AGGREGATED_eFCR",
        "PERIOD_eFCR",
    ]])

    if selected_kpi == "Biomass":
        df = df.dropna(subset=["TOTAL_WEIGHT_KG"])
        fig = px.line(
            df,
            x="DATE",
            y="TOTAL_WEIGHT_KG",
            markers=True,
            title=f"Cage {selected_cage}: Biomass Over Time",
            labels={"TOTAL_WEIGHT_KG": "Total Biomass (kg)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_kpi == "ABW":
        df = df.dropna(subset=["ABW_G"])
        fig = px.line(
            df,
            x="DATE",
            y="ABW_G",
            markers=True,
            title=f"Cage {selected_cage}: Average Body Weight Over Time",
            labels={"ABW_G": "ABW (g)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # eFCR
        df = df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
        fig = px.line(
            df,
            x="DATE",
            y="AGGREGATED_eFCR",
            markers=True,
            title=f"Cage {selected_cage}: eFCR Over Time",
            labels={"AGGREGATED_eFCR": "Aggregated eFCR"},
        )
        # Ensure both lines show up with clear legend entries
        fig.update_traces(name="Aggregated eFCR", selector=dict(name="AGGREGATED_eFCR"))
        fig.add_scatter(x=df["DATE"], y=df["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR")
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the three Excel files to begin.")

