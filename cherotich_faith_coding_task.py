#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Load data 
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    return feeding, harvest, sampling


# 2. Preprocess Cage 2
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

    # Ensure dates are datetime
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'], errors='coerce')
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'], errors='coerce')
    feeding_c2 = feeding_c2.dropna(subset=['DATE'])
    sampling_c2 = sampling_c2.dropna(subset=['DATE'])

    return feeding_c2, harvest_c2, sampling_c2

# 3. Compute production summary
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# 4. Create mock cages (3-7)
def create_mock_cage_data(summary_c2, feeding_c2):
    mock_summaries = {}

    if feeding_c2.empty:
        st.warning("Feeding data is empty. Cannot generate mock cages.")
        return {}

    for cage_id in range(3, 8):
        # Generate daily feed
        date_range = pd.date_range(start=feeding_c2['DATE'].min(), end=feeding_c2['DATE'].max(), freq='D')
        daily_feed = np.random.normal(
            feeding_c2['FEED AMOUNT (Kg)'].mean(),
            0.1 * feeding_c2['FEED AMOUNT (Kg)'].mean(),
            len(date_range)
        )
        feeding_mock = pd.DataFrame({
            'DATE': date_range,
            'CAGE NUMBER': cage_id,
            'FEED AMOUNT (Kg)': daily_feed
        })
        feeding_mock['CUM_FEED'] = feeding_mock['FEED AMOUNT (Kg)'].cumsum()

        # Sampling mock
        sampling_mock = summary_c2[['DATE','NUMBER OF FISH','AVERAGE BODY WEIGHT (g)']].copy()
        sampling_mock['CAGE NUMBER'] = cage_id
        sampling_mock['AVERAGE BODY WEIGHT (g)'] *= np.random.normal(1, 0.05, len(sampling_mock))
        sampling_mock['NUMBER OF FISH'] += np.random.randint(-50, 50, len(sampling_mock))
        sampling_mock['TOTAL_WEIGHT_KG'] = sampling_mock['NUMBER OF FISH'] * sampling_mock['AVERAGE BODY WEIGHT (g)'] / 1000

        # Merge and compute eFCR
        summary_mock = pd.merge_asof(
            sampling_mock.sort_values('DATE'),
            feeding_mock[['DATE','CUM_FEED']].sort_values('DATE'),
            on='DATE'
        )
        summary_mock['AGGREGATED_eFCR'] = summary_mock['CUM_FEED'] / summary_mock['TOTAL_WEIGHT_KG']
        summary_mock['PERIOD_WEIGHT_GAIN'] = summary_mock['TOTAL_WEIGHT_KG'].diff().fillna(summary_mock['TOTAL_WEIGHT_KG'])
        summary_mock['PERIOD_FEED'] = summary_mock['CUM_FEED'].diff().fillna(summary_mock['CUM_FEED'])
        summary_mock['PERIOD_eFCR'] = summary_mock['PERIOD_FEED'] / summary_mock['PERIOD_WEIGHT_GAIN']

        mock_summaries[cage_id] = summary_mock

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
    mock_cages = create_mock_cage_data(summary_c2, feeding_c2)
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
        df = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        # Plot eFCR
        df = df.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'])
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)  # remove name argument
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)

