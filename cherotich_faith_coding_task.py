# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load data
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    return feeding, harvest, sampling

# -------------------------------
# 2. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2

    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Add stocking manually
    stocking_date = pd.to_datetime("2024-07-16")
    stocked_fish = 7902
    initial_abw = 0.7
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE')

    # Limit timeframe
    start_date = pd.to_datetime("2024-07-16")
    end_date = pd.to_datetime("2025-06-30")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 3. Compute production summary
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    # Merge cumulative feed to sampling points
    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    # Aggregated eFCR
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']

    # Period eFCR
    period_feed = []
    period_gain = []
    for i in range(len(summary)):
        if i == 0:
            period_feed.append(np.nan)
            period_gain.append(np.nan)
        else:
            period_feed.append(summary.loc[i, 'CUM_FEED'] - summary.loc[i-1, 'CUM_FEED'])
            period_gain.append(summary.loc[i, 'TOTAL_WEIGHT_KG'] - summary.loc[i-1, 'TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = period_feed
    summary['PERIOD_WEIGHT_GAIN'] = period_gain
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# -------------------------------
# 4. Create mock cages (3-7)
# -------------------------------
def create_mock_cages(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id

        # Randomize weights ±5%, number of fish ±50, feed ±10%
        mock['TOTAL_WEIGHT_KG'] *= np.random.normal(1, 0.05, size=len(mock))
        mock['NUMBER OF FISH'] = mock['NUMBER OF FISH'] + np.random.randint(-50, 50, size=len(mock))
        mock['CUM_FEED'] *= np.random.normal(1, 0.1, size=len(mock))

        # Recompute eFCR
        mock['AGGREGATED_eFCR'] = mock['CUM_FEED'] / mock['TOTAL_WEIGHT_KG']

        period_feed = []
        period_gain = []
        for i in range(len(mock)):
            if i == 0:
                period_feed.append(np.nan)
                period_gain.append(np.nan)
            else:
                period_feed.append(mock.loc[i, 'CUM_FEED'] - mock.loc[i-1, 'CUM_FEED'])
                period_gain.append(mock.loc[i, 'TOTAL_WEIGHT_KG'] - mock.loc[i-1, 'TOTAL_WEIGHT_KG'])
        mock['PERIOD_FEED'] = period_feed
        mock['PERIOD_WEIGHT_GAIN'] = period_gain
        mock['PERIOD_eFCR'] = mock['PERIOD_FEED'] / mock['PERIOD_WEIGHT_GAIN']

        mock_summaries[cage_id] = mock

    return mock_summaries

# -------------------------------
# 5. Streamlit interface
# -------------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling = load_data(feeding_file, harvest_file, sampling_file)

    # Process cage 2
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Generate mock cages
    mock_cages = create_mock_cages(summary_c2)
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
else:
    st.info("Please upload all three Excel files: Feeding Record, Fish Harvest, and Fish Sampling.")
