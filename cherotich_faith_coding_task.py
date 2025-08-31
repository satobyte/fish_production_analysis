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
        "ORIGIN CAGE NUMBER": "ORIGIN CAGE",
        "DESTINATION CAGE NUMBER": "DESTINATION CAGE",
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
        # Try to coerce origin/destination
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # Standardize possible weight column name
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


# =====================
# Preprocess Cage 2 (+ transfers)
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    feeding_c2 = feeding[feeding["CAGE NUMBER"] == cage_number].copy()
    harvest_c2 = harvest[harvest["CAGE NUMBER"] == cage_number].copy()
    sampling_c2 = sampling[sampling["CAGE NUMBER"] == cage_number].copy()

    # Stocking
    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame([
        {"DATE": stocking_date, "CAGE NUMBER": cage_number, "AVERAGE BODY WEIGHT(G)": initial_abw}
    ])

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=["DATE"]).sort_values("DATE")

    # Window
    start_date, end_date = stocking_date, pd.to_datetime("2025-06-30")
    sampling_c2 = sampling_c2[(sampling_c2["DATE"] >= start_date) & (sampling_c2["DATE"] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2["DATE"] >= start_date) & (feeding_c2["DATE"] <= end_date)]
    harvest_c2 = harvest_c2[(harvest_c2["DATE"] >= start_date) & (harvest_c2["DATE"] <= end_date)]

    # Cumulative harvested fish
    fish_col_h = find_col(harvest_c2, ["NUMBER OF FISH", "NUMBER OF FISH ", "N_FISH"], fuzzy_hint="FISH")
    if fish_col_h:
        h = harvest_c2[["DATE", fish_col_h]].dropna().copy()
        h["HARV_CUM_FISH"] = h[fish_col_h].astype(float).cumsum()
    else:
        h = pd.DataFrame({"DATE": [], "HARV_CUM_FISH": []})

    sampling_c2["STOCKED"] = stocked_fish

    # Transfers (in/out) for cage 2
    tin, tout = None, None
    if transfers is not None:
        # ensure required cols
        num_col_t = find_col(transfers, ["NUMBER OF FISH", "N_FISH"], fuzzy_hint="FISH")
        w_col_t = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]"], fuzzy_hint="WEIGHT")
        # outgoing (origin = cage 2)
        t_out = transfers[(transfers.get("ORIGIN CAGE") == cage_number)].copy() if "ORIGIN CAGE" in transfers.columns else pd.DataFrame()
        if not t_out.empty and num_col_t:
            t_out["OUT_FISH_CUM"] = pd.to_numeric(t_out[num_col_t], errors="coerce").fillna(0).cumsum()
        if not t_out.empty and w_col_t:
            t_out["OUT_KG_CUM"] = pd.to_numeric(t_out[w_col_t], errors="coerce").fillna(0).cumsum()
        # incoming (dest = cage 2)
        t_in = transfers[(transfers.get("DESTINATION CAGE") == cage_number)].copy() if "DESTINATION CAGE" in transfers.columns else pd.DataFrame()
        if not t_in.empty and num_col_t:
            t_in["IN_FISH_CUM"] = pd.to_numeric(t_in[num_col_t], errors="coerce").fillna(0).cumsum()
        if not t_in.empty and w_col_t:
            t_in["IN_KG_CUM"] = pd.to_numeric(t_in[w_col_t], errors="coerce").fillna(0).cumsum()

        tin, tout = (t_in if not t_in.empty else None), (t_out if not t_out.empty else None)

    # Merge cumulative harvest + transfers to sampling dates
    base = sampling_c2.sort_values("DATE").copy()
    base["HARV_CUM_FISH"] = 0.0
    if not h.empty:
        base = pd.merge_asof(base, h[["DATE", "HARV_CUM_FISH"]].sort_values("DATE"), on="DATE", direction="backward")
        base["HARV_CUM_FISH"] = base["HARV_CUM_FISH"].fillna(0)

    # Add cumulative transfer fish & kg
    for df_t, fish_cum_col, kg_cum_col, prefix in [
        (tin, "IN_FISH_CUM", "IN_KG_CUM", "IN"),
        (tout, "OUT_FISH_CUM", "OUT_KG_CUM", "OUT"),
    ]:
        if df_t is not None:
            cols = ["DATE"]
            if fish_cum_col in df_t.columns:
                cols.append(fish_cum_col)
            if kg_cum_col in df_t.columns:
                cols.append(kg_cum_col)
            tmp = df_t[cols].sort_values("DATE").copy()
            base = pd.merge_asof(base, tmp, on="DATE", direction="backward")
            if fish_cum_col in base.columns:
                base[fish_cum_col] = base[fish_cum_col].fillna(0)
            if kg_cum_col in base.columns:
                base[kg_cum_col] = base[kg_cum_col].fillna(0)
        else:
            base[f"{prefix}_FISH_CUM"] = 0.0
            base[f"{prefix}_KG_CUM"] = 0.0

    # Standing fish = stocked - harvested + in - out
    base["FISH_ALIVE"] = (
        base["STOCKED"]
        - base.get("HARV_CUM_FISH", 0)
        + base.get("IN_FISH_CUM", 0)
        - base.get("OUT_FISH_CUM", 0)
    ).clip(lower=0)

    # Keep for downstream
    sampling_c2 = base
    sampling_c2["NUMBER OF FISH"] = sampling_c2["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, sampling_c2


# =====================
# Compute summary (adds transfer in/out per period)
# =====================

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()

    # Feed column
    feed_col = find_col(
        feeding_c2,
        ["FEED AMOUNT (KG)", "FEED AMOUNT (Kg)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED KG", "FEED_AMOUNT", "FEED"],
        fuzzy_hint="FEED",
    )
    if not feed_col:
        st.error(f"Could not find feed column. Available: {list(feeding_c2.columns)}")
        st.stop()

    # ABW column
    abw_col = find_col(
        sampling_c2,
        ["AVERAGE BODY WEIGHT(G)", "AVERAGE BODY WEIGHT (G)", "ABW(G)", "ABW [G]", "ABW"],
        fuzzy_hint="ABW",
    )
    if not abw_col:
        st.error(f"Could not find ABW column. Available: {list(sampling_c2.columns)}")
        st.stop()

    # cumulative feed
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # ABW & biomass
    sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[abw_col].map(to_number), errors="coerce")
    sampling_c2["TOTAL_WEIGHT_KG"] = (
        pd.to_numeric(sampling_c2["FISH_ALIVE"], errors="coerce").fillna(0)
        * sampling_c2["ABW_G"].fillna(0)
        / 1000.0
    )

    # Align cumulative feed to sampling dates
    summary = pd.merge_asof(
        sampling_c2.sort_values("DATE"),
        feeding_c2.sort_values("DATE")["DATE"].to_frame().assign(CUM_FEED=feeding_c2["CUM_FEED"].values),
        on="DATE",
        direction="backward",
    )

    # Period deltas
    tw = summary["TOTAL_WEIGHT_KG"].replace(0, np.nan)
    summary["AGGREGATED_eFCR"] = summary["CUM_FEED"] / tw
    summary["PERIOD_WEIGHT_GAIN"] = summary["TOTAL_WEIGHT_KG"].diff().fillna(summary["TOTAL_WEIGHT_KG"])
    summary["PERIOD_FEED"] = summary["CUM_FEED"].diff().fillna(summary["CUM_FEED"])
    summary["PERIOD_eFCR"] = summary["PERIOD_FEED"] / summary["PERIOD_WEIGHT_GAIN"].replace(0, np.nan)

    # Transfers per period (kg and fish)
    for cum_col, per_col in [
        ("IN_KG_CUM", "PERIOD_TRANSFER_IN_KG"),
        ("OUT_KG_CUM", "PERIOD_TRANSFER_OUT_KG"),
        ("IN_FISH_CUM", "PERIOD_TRANSFER_IN_FISH"),
        ("OUT_FISH_CUM", "PERIOD_TRANSFER_OUT_FISH"),
    ]:
        if cum_col in summary.columns:
            summary[per_col] = summary[cum_col].diff().fillna(summary[cum_col])
        else:
            summary[per_col] = 0.0

    summary["NET_TRANSFER_KG"] = summary["PERIOD_TRANSFER_IN_KG"] - summary["PERIOD_TRANSFER_OUT_KG"]
    summary["NET_TRANSFER_FISH"] = summary["PERIOD_TRANSFER_IN_FISH"] - summary["PERIOD_TRANSFER_OUT_FISH"]

    # Keep tidy
    keep = [
        "DATE",
        "CAGE NUMBER",
        "NUMBER OF FISH",
        "ABW_G",
        "TOTAL_WEIGHT_KG",
        "CUM_FEED",
        "AGGREGATED_eFCR",
        "PERIOD_eFCR",
        "PERIOD_TRANSFER_IN_KG",
        "PERIOD_TRANSFER_OUT_KG",
        "NET_TRANSFER_KG",
    ]
    return summary[[c for c in keep if c in summary.columns]]


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

    # Mock cages still work (no transfers simulated there)
    def create_mock_cage_data(summary_c2):
        mock_summaries = {}
        for cage_id in range(3, 8):
            mock = summary_c2.copy()
            mock["CAGE NUMBER"] = cage_id
            rng = np.random.default_rng(cage_id)
            for col, sd in [("TOTAL_WEIGHT_KG", 0.05), ("CUM_FEED", 0.10)]:
                if col in mock.columns:
                    mock[col] = mock[col] * rng.normal(1, sd, size=len(mock))
            if "NUMBER OF FISH" in mock.columns:
                mock["NUMBER OF FISH"] = (mock["NUMBER OF FISH"].astype(int) + rng.integers(-50, 51, size=len(mock))).clip(lower=1)
            # recompute eFCR
            if {"CUM_FEED", "TOTAL_WEIGHT_KG"}.issubset(mock.columns):
                tw = mock["TOTAL_WEIGHT_KG"].replace(0, np.nan)
                mock["AGGREGATED_eFCR"] = mock["CUM_FEED"] / tw
                mock["PERIOD_WEIGHT_GAIN"] = mock["TOTAL_WEIGHT_KG"].diff().fillna(mock["TOTAL_WEIGHT_KG"])
                mock["PERIOD_FEED"] = mock["CUM_FEED"].diff().fillna(mock["CUM_FEED"])
                mock["PERIOD_eFCR"] = mock["PERIOD_FEED"] / mock["PERIOD_WEIGHT_GAIN"].replace(0, np.nan)
            mock_summaries[cage_id] = mock
        return mock_summaries

    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR"])  

    df = all_cages[selected_cage].copy()

    st.subheader(f"Cage {selected_cage} – Production Summary")
    show_cols = [
        "DATE","NUMBER OF FISH","ABW_G","TOTAL_WEIGHT_KG","AGGREGATED_eFCR","PERIOD_eFCR",
        "PERIOD_TRANSFER_IN_KG","PERIOD_TRANSFER_OUT_KG","NET_TRANSFER_KG"
    ]
    st.dataframe(df[[c for c in show_cols if c in df.columns]])

    if selected_kpi == "Biomass":
        if "TOTAL_WEIGHT_KG" in df.columns:
            fig = px.line(df.dropna(subset=["TOTAL_WEIGHT_KG"]), x="DATE", y="TOTAL_WEIGHT_KG", markers=True,
                          title=f"Cage {selected_cage}: Biomass Over Time",
                          labels={"TOTAL_WEIGHT_KG": "Total Biomass (kg)"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Biomass not available for this cage.")

    elif selected_kpi == "ABW":
        if "ABW_G" in df.columns:
            fig = px.line(df.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                          title=f"Cage {selected_cage}: Average Body Weight Over Time",
                          labels={"ABW_G": "ABW (g)"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ABW not available for this cage.")

    else:  # eFCR
        if {"AGGREGATED_eFCR","PERIOD_eFCR"}.issubset(df.columns):
            dff = df.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"]) 
            fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                          title=f"Cage {selected_cage}: eFCR Over Time",
                          labels={"AGGREGATED_eFCR": "Aggregated eFCR"})
            fig.update_traces(showlegend=True, name="Aggregated eFCR")
            fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", showlegend=True, line=dict(dash="dash"))
            fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("eFCR metrics not available for this cage.")
else:
    st.info("Upload the three Excel files to begin.")
