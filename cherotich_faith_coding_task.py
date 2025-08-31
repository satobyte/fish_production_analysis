# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

def normalize_columns(df):
    # upper-case, strip, collapse spaces
    df = df.copy()
    df.columns = [re.sub(r'\s+', ' ', c.strip().upper()) for c in df.columns]
    ren = {
        'CAGE': 'CAGE NUMBER',
        'CAGE_NO': 'CAGE NUMBER',
        'CAGE ID': 'CAGE NUMBER',
        'TOTAL WEIGHT (KG)': 'TOTAL WEIGHT [kg]',
        'ABW(G)': 'AVERAGE BODY WEIGHT (g)',
        'ABW [G]': 'AVERAGE BODY WEIGHT (g)',
        'AVERAGE BODYWEIGHT (G)': 'AVERAGE BODY WEIGHT (g)',
    }
    df.rename(columns={k:v for k,v in ren.items() if k in df.columns}, inplace=True)
    return df

def to_int_cage(series):
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r'(\d+)', str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)


# 1. Load data
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))

    # coerce cage columns
    if 'CAGE NUMBER' in feeding.columns:
        feeding['CAGE NUMBER'] = to_int_cage(feeding['CAGE NUMBER'])
    if 'CAGE NUMBER' in sampling.columns:
        sampling['CAGE NUMBER'] = to_int_cage(sampling['CAGE NUMBER'])
    if 'CAGE NUMBER' in harvest.columns:
        harvest['CAGE NUMBER'] = to_int_cage(harvest['CAGE NUMBER'])
    elif 'CAGE' in harvest.columns:
        harvest['CAGE NUMBER'] = to_int_cage(harvest['CAGE'])

    # dates
    for df in (feeding, harvest, sampling):
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    return feeding, harvest, sampling


# 2. Preprocess Cage 2
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2
    feeding_c2  = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2  = harvest[harvest['CAGE NUMBER'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish, initial_abw = 7902, 0.7
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])

    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    sampling_c2 = sampling_c2.dropna(subset=['DATE']).sort_values('DATE')

    start_date, end_date = stocking_date, pd.to_datetime("2025-06-30")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2  = feeding_c2[(feeding_c2['DATE']  >= start_date) & (feeding_c2['DATE']  <= end_date)]
    harvest_c2  = harvest_c2[(harvest_c2['DATE']  >= start_date) & (harvest_c2['DATE']  <= end_date)]

    # compute standing fish (stocked – harvested)
    if 'NUMBER OF FISH' in harvest_c2.columns:
        h = harvest_c2[['DATE','NUMBER OF FISH']].dropna().copy()
        h['HARV_CUM'] = h['NUMBER OF FISH'].cumsum()
    else:
        h = pd.DataFrame({'DATE':[], 'HARV_CUM':[]})

    sampling_c2['STOCKED'] = stocked_fish
    if not h.empty:
        sampling_c2 = pd.merge_asof(
            sampling_c2.sort_values('DATE'),
            h[['DATE','HARV_CUM']].sort_values('DATE'),
            on='DATE', direction='backward'
        )
        sampling_c2['HARV_CUM'] = sampling_c2['HARV_CUM'].fillna(0).astype(int)
    else:
        sampling_c2['HARV_CUM'] = 0

    sampling_c2['FISH_ALIVE'] = (sampling_c2['STOCKED'] - sampling_c2['HARV_CUM']).clip(lower=0)
    return feeding_c2, harvest_c2, sampling_c2

# 3. Compute production summary
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].astype(float).cumsum()

    sampling_c2['TOTAL_WEIGHT_KG'] = (
        sampling_c2['FISH_ALIVE'].astype(float) *
        sampling_c2['AVERAGE BODY WEIGHT (g)'].astype(float) / 1000.0
    )

    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE', direction='backward'
    )

    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG'].replace(0, np.nan)
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN'].replace(0, np.nan)

    return summary


# 4. Create mock cages (3-7)
def create_mock_cage_data(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id

        # Randomize weights ±5%, number of fish ±50, feed ±10%
        mock['TOTAL_WEIGHT_KG'] *= np.random.normal(1, 0.05, size=len(mock))
        mock['NUMBER OF FISH'] = mock['NUMBER OF FISH'] + np.random.randint(-50, 50, size=len(mock))
        mock['CUM_FEED'] *= np.random.normal(1, 0.1, size=len(mock))

        # recompute eFCR
        mock['AGGREGATED_eFCR'] = mock['CUM_FEED'] / mock['TOTAL_WEIGHT_KG']
        mock['PERIOD_WEIGHT_GAIN'] = mock['TOTAL_WEIGHT_KG'].diff().fillna(mock['TOTAL_WEIGHT_KG'])
        mock['PERIOD_FEED'] = mock['CUM_FEED'].diff().fillna(mock['CUM_FEED'])
        mock['PERIOD_eFCR'] = mock['PERIOD_FEED'] / mock['PERIOD_WEIGHT_GAIN']

        mock_summaries[cage_id] = mock
    return mock_summaries

# 5. Streamlit Interface
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling = load_data(feeding_file, harvest_file, sampling_file)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Generate mock cages
    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar selectors
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage]

    # Display production summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']])

    # Plot graphs
    if selected_kpi == "Growth":
        df['TOTAL_WEIGHT_KG'] = pd.to_numeric(df['TOTAL_WEIGHT_KG'], errors='coerce')
        df = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        df['AGGREGATED_eFCR'] = pd.to_numeric(df['AGGREGATED_eFCR'], errors='coerce')
        df['PERIOD_eFCR'] = pd.to_numeric(df['PERIOD_eFCR'], errors='coerce')
        df = df.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'])
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
