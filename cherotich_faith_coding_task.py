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
st.info("Upload the three Excel files to begin.")
