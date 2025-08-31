# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load data function
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    
    # Strip column names and remove spaces
    feeding.columns = feeding.columns.str.strip().str.upper()
    harvest.columns = harvest.columns.str.strip().str.upper()
    sampling.columns = sampling.columns.str.strip().str.upper()
    
    # Ensure DATE column is datetime
    feeding['DATE'] = pd.to_datetime(feeding['DATE'], dayfirst=True, errors='coerce')
    harvest['DATE'] = pd.to_datetime(harvest['DATE'], dayfirst=True, errors='coerce')
    sampling['DATE'] = pd.to_datetime(sampling['DATE'], dayfirst=True, errors='coerce')
    
    return feeding, harvest, sampling

# -------------------------------
# 2. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2

    if cage_number not in feeding['CAGE NUMBER'].unique():
        st.warning("Cage 2 data not found in feeding record!")
        return None, None, None

    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Add manual stocking
    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish = 7902
    initial_abw = 0.7
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values('DATE')

    # Filter timeframe
    start_date = pd.to_datetime("2024-07-16")
    end_date = pd.to_datetime("2025-06-30")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 3. Compute production summary
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    if feeding_c2.empty or sampling_c2.empty:
        return None

    feeding_c2 = feeding_c2.sort_values('DATE')
    sampling_c2 = sampling_c2.sort_values('DATE')

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (KG)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (G)'] / 1000

    # Merge feed to sampling by nearest past date
    summary = pd.merge_asof(
        sampling_c2,
        feeding_c2[['DATE', 'CUM_FEED']],
        on='DATE',
        direction='backward'
    )

    # Compute eFCRs
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# -------------------------------
# 4. Create mock cages (3-7)
# -------------------------------
def create_mock_cages(summary_c2):
    if summary_c2 is None or summary_c2.empty:
        return {}

    mock_cages = {}
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

        mock_cages[cage_id] = mock

    return mock_cages

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling = load_data(feeding_file, harvest_file, sampling_file)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)
    if feeding_c2 is None or sampling_c2 is None or feeding_c2.empty:
        st.error("Cage 2 data is empty. Please check your Excel files.")
    else:
        summary_c2 = compute_summary(feeding_c2, sampling_c2)
        mock_cages = create_mock_cages(summary_c2)
        all_cages = {2: summary_c2, **mock_cages}

        st.sidebar.header("Select Options")
        selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
        selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

        df = all_cages[selected_cage]

        # Production summary table
        st.subheader(f"Cage {selected_cage} Production Summary")
        st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']])

        # Graphs
        if selected_kpi == "Growth":
            fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                          title=f'Cage {selected_cage}: Growth Over Time',
                          labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
            st.plotly_chart(fig)
        else:
            fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)
            fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
            fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
            st.plotly_chart(fig)
