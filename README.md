
# Fish Cage Production Analysis

A **Streamlit** application for analyzing fish cage production, including growth and feed conversion efficiency (eFCR) for Cage 2 and simulated data for additional cages.

---

## Features

- Upload Excel files for **Feeding Records**, **Fish Harvest**, and **Fish Sampling**.
- Preprocess and clean data for Cage 2.
- Compute key production metrics:
  - **Total biomass (kg)**
  - **Cumulative feed (kg)**
  - **Aggregated eFCR**
  - **Period eFCR**
- Generate **mock data** for additional cages (3–7) based on Cage 2.
- Interactive visualization of:
  - **Growth over time**
  - **eFCR trends**
- Sidebar options to select **cage** and **KPI**.

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
````

2. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
streamlit
pandas
numpy
plotly
openpyxl
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run main_app.py
```

2. In the sidebar, upload the following Excel files:

   * Feeding Record (`feeding_file.xlsx`)
   * Fish Harvest (`harvest_file.xlsx`)
   * Fish Sampling (`sampling_file.xlsx`)

3. Select a **Cage** (2–7) and **KPI** (`Growth` or `eFCR`) from the sidebar.

4. View:

   * Production summary table
   * Interactive growth or eFCR plots

---

## Data Requirements

* **Feeding Record**: `DATE`, `CAGE NUMBER`, `FEED AMOUNT (Kg)`
* **Fish Harvest**: `DATE`, `CAGE`, `NUMBER OF FISH`, `AVERAGE BODY WEIGHT (g)` (if applicable)
* **Fish Sampling**: `DATE`, `CAGE NUMBER`, `NUMBER OF FISH`, `AVERAGE BODY WEIGHT (g)`

> If feeding data is missing for Cage 2, the app generates synthetic daily feed data.

---

## How It Works

1. **Data Loading**: Reads user-uploaded Excel files.
2. **Preprocessing**: Filters Cage 2, adds initial stocking data, and ensures date range consistency.
3. **Summary Computation**: Calculates total biomass, cumulative feed, aggregated and period eFCR.
4. **Mock Cage Simulation**: Generates synthetic production data for cages 3–7 based on Cage 2 trends.
5. **Visualization**: Interactive plots for growth and eFCR using Plotly.

---

## Example Visualizations

* Growth over time per cage
* Aggregated and period eFCR trends

---

## License

MIT License

---

## Author

**Cherotich Faith** – Data Scientist 

Do you want me to do that?
```
